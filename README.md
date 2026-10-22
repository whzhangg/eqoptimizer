# eqopt: differentiable thermodynamic optimization in PyTorch

This package supports optimization of CALPHAD-type and non-CALPHAD-type thermodynamic models by PyTorch. CALPHAD-type CEF models are implemented, and new model can be defined according to need of user as a PyTorch `nn.Module` and `ThermodynamicModel`.

## Quick Start

### Installation

This project depends on the following packages: `torch`, `numpy`, `scipy`, `sympy`, `rich`,
`pandas`, and `pyyaml`. `ase` is needed for the use of potential. Optionally, `pycalphad` is used
for reference equilibrium calculation.

To install the code, simply install dependencies through `pip` and install directly. `conda-forge` can be used to manage the environment:

```bash
conda create -n eqopt python=3.12
conda activate eqopt
conda install torch numpy scipy sympy rich pandas pyyaml ase pycalphad
pip install -e .
```

### Usage

A few optimization examples including binary and ternary have been included in the examples folder. Optimization following the general pseudo-code:

```python
equilibria: Sequence[PhaseEquilibrium] = list(datas)

system = EnsembleSystem({
    phase_id: thermodynamic_model,
    # ...
})

config = OptimizationConfig(
    epochs=1000, 
    lr=100, 
    regularization_weight=1e-12
)

optimize_thermodynamic_parameters(
    system,
    config,
    equilibria=equilibria,
)
```

### Command-line Usage

From command-line, an executable can be used to start an optimization from a set of files. `eqopt-calphad` optimizes a CALPHAD model from a parameterized TDB file and phase-equilibrium
data stored in CSV format:

```text
usage: eqopt-calphad [-h] [--config CONFIG] [--output-dir OUTPUT_DIR] tdb equilibria
```

#### Parameterized TDB file

Trainable parameters are written as symbols in a TDB `PARAMETER` expression. `T` is reserved for temperature, while every other symbol is treated as a trainable parameter. For example:

```text
PARAMETER G(FCC_A1,AL,ZN:VA;0) 298.15 -10000+2*T+a+b*T/1000; 6000 N !
```

Here, `a` and `b` are optimized while the remaining expression is fixed. Numerical factors should be chosen so that the trainable symbols have similar useful scales. Reusing a symbol within one expression reuses the same parameter in that expression. Recursive evaluation (`FUNCTIONS`) are not yet supported.

#### Phase-equilibrium CSV file

Each row represents one phase participating in an observed equilibrium. Rows with the same integer `eq_index` belong to the same equilibrium:

```csv
eq_index,temperature,phase_name,x_AL,x_ZN
0,500,FCC_A1,0.92,0.08
0,500,HCP_A3,0.01,0.99
1,700,FCC_A1,0.70,0.30
1,700,LIQUID,0.42,0.58
```

Composition columns use the form `x_<ELEMENT>` and must cover the elements represented by the TDB models. A phase with unknown equilibrium composition may be included by leaving all of its composition fields empty. Partially known composition is not supported currently. Phase names must match those in the TDB file.

#### Configurations

Optimization options are supplied through an optional YAML file. A complete commented template
is available at [`config/calphad-default.yaml`](config/calphad-default.yaml). It contains three
sections:

- `general`: output, recording, printing, and parameter-initialization settings.
- `optimization`: optimizer, loss, scheduler, and latent chemical-potential settings.
- `cef`: sampling and internal CEF minimization settings.

Copy the template and edit only the required values:

```bash
cp config/calphad-default.yaml calphad.yaml
eqopt-calphad model.tdb equilibria.csv --config calphad.yaml
```

### Repository Layout

- `examples` contain several runnable examples of the optimizations

- `src/eqopt` is the main package that defines loss function, abstract classes and optimization procedures

- `src/potentials` contains torch-compatible potentials and calculators

## Status and Limitations

This package is under development, and currently:

- Optimization utilizes phase equilibria only, optimization with respect to other forms of experimental data such as thermochemical data is not yet supported.

- Compound energy formalism (CEF) models support compound phases, solution phases, intermetallics,
  and the Inden-Hillert-Jarl magnetic Gibbs-energy contribution.

- Loading initial models from a TDB file is possible but is currently done in a very simple way and can be sensitive to the syntax. (see examples)

- Many parameters are set to reasonable default values. However, it is often important to adopt the value of parameters, such as learning rate, to ensure successful optimization.

## Citation

Zhang, W., Crivello, J.-C., Matsuoka, Y., Koyama, T. & Abe, T. Machine Learning Compatible CALPHAD-type Optimization from Phase Equilibria by Auto-differentiation. Preprint at https://doi.org/10.48550/ARXIV.2608.00516 (2026).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE.txt) for details.
