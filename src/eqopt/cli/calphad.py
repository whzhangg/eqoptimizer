import argparse
import dataclasses
import random
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
import yaml

from ..config import OptimizationConfig
from ..models import CEFConfig, EnsembleSystem, SymbolicCEFParser
from ..optimize import optimize_thermodynamic_parameters
from ..phase import PhaseID
from ..training_data import phase_equilibria_from_csv
from ..models.calphad.cef_parser import _tdb_commands


def _read_config(filename: str | Path | None) -> dict:
    if filename is None:
        return {}
    values = yaml.safe_load(Path(filename).read_text())
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError("The YAML configuration must contain a mapping at its top level.")
    return dict(values)


def _pop_section(values: dict, name: str) -> dict:
    section = values.pop(name, {})
    if not isinstance(section, Mapping):
        raise TypeError(f"Configuration section {name!r} must be a mapping.")
    return dict(section)


def _optimization_config(values: Mapping) -> OptimizationConfig:
    options = dict(values)
    field_names = {field.name for field in dataclasses.fields(OptimizationConfig)}
    unknown = set(options) - field_names
    if unknown:
        names = ", ".join(sorted(unknown))
        raise TypeError(f"Unknown optimization configuration option(s): {names}")

    optimizer = options.get("optimizer_cls")
    if isinstance(optimizer, str):
        try:
            options["optimizer_cls"] = getattr(torch.optim, optimizer)
        except AttributeError:
            raise ValueError(f"Unknown torch optimizer {optimizer!r}.") from None
    return OptimizationConfig(**options)


def _optimized_tdb_string(
    parser: SymbolicCEFParser,
    system: EnsembleSystem,
) -> str:
    retained_commands = []
    for command in _tdb_commands(parser.tdb_string):
        upper_command = command.upper()
        if upper_command.startswith(("PARAMETER", "PARA ", "PAR ", "FUNCTION", "FUN ")):
            continue
        retained_commands.append(f"{command} !")
    function_commands = [
        f"FUNCTION {name} {function.get_tdb_expression()} !"
        for name, function in parser.functions.items()
    ]
    parameter_commands = [
        command
        for phase_id in system.phase_ids
        for command in _tdb_commands(system.get_model_by_phase_id(phase_id).get_tdb_str())
        if command.upper().startswith(("PARAMETER", "PARA ", "PAR "))
    ]
    commands = (
        *retained_commands,
        *function_commands,
        *(f"{command} !" for command in parameter_commands),
    )
    return "\n".join(commands) + "\n"


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eqopt-calphad",
        description="Optimize parameterized CALPHAD models against phase equilibria.",
    )
    parser.add_argument("tdb", type=Path, help="Parameterized input TDB file.")
    parser.add_argument("equilibria", type=Path, help="Phase-equilibrium training CSV file.")
    parser.add_argument("--config", type=Path, help="Optional YAML configuration file.")
    parser.add_argument(
        "--seed", type=int,
        help="Random seed for initialization and sampling, overriding general.seed in YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory, overriding general.output_dir from the YAML file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _argument_parser().parse_args(argv)
    values = _read_config(arguments.config)
    general_options = _pop_section(values, "general")
    optimization_options = _pop_section(values, "optimization")
    cef_options = _pop_section(values, "cef")
    if values:
        names = ", ".join(sorted(values))
        raise TypeError(f"Unknown top-level configuration section(s): {names}")

    initial_parameter_scale = float(general_options.pop("initial_parameter_scale", 0.1))
    config_output_dir = Path(general_options.pop("output_dir", "results"))
    output_dir = arguments.output_dir or config_output_dir
    config_seed = general_options.pop("seed", None)
    seed = arguments.seed if arguments.seed is not None else config_seed
    record_every = int(general_options.pop("record_every", 10))
    print_final_results = bool(general_options.pop("print_final_results", True))
    if general_options:
        names = ", ".join(sorted(general_options))
        raise TypeError(f"Unknown general configuration option(s): {names}")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # load the thermodynamic models
    parser = SymbolicCEFParser(
        arguments.tdb,
        initial_parameter_scale=initial_parameter_scale,
    )
    parsed_models = parser.get_models(config=CEFConfig(**cef_options))
    models = {}
    for phase_name, model in parsed_models.items():
        phase_id = PhaseID(name=phase_name, elements=tuple(model.elements))
        models[phase_id] = model
    excluded_phase_ids = {
        phase_id
        for phase_id in models
        if phase_id.name in parser.disordered_reference_phase_names
    }
    system = EnsembleSystem(models, excluded_phase_ids=excluded_phase_ids)

    # load the equilibrium
    equilibria = phase_equilibria_from_csv(arguments.equilibria, system)

    # create optimization configuration
    config = _optimization_config(optimization_options)

    output_dir.mkdir(parents=True, exist_ok=True)
    optimized_system, _, _ = optimize_thermodynamic_parameters(
        system,
        config,
        equilibria=equilibria,
        checkpoint_dir=output_dir / "checkpoint",
        record_every=record_every,
        print_final_results=print_final_results,
    )
    output_tdb = output_dir / "optimized.tdb"
    output_tdb.write_text(_optimized_tdb_string(parser, optimized_system))
    print(f"Wrote optimized TDB to {output_tdb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
