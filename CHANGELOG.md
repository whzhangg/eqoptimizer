# Change log

## 2026-10-24

- added order-disorder model (split-CEF) documented by '/docs/calphad.md'.

## 2026-10-21

### Code changes

- refactored CEF energy evaluation into additive contributions. `CEFSurface` now represents a
  reusable collection of CEF terms, while `CEF` can combine multiple contributions without
  embedding their implementation details.

- added the Inden-Hillert-Jarl magnetic Gibbs-energy contribution. Curie temperature and magnetic
  moment are represented as CEF surfaces, allowing their parameters to vary with the internal
  coordinates and participate in gradient-based optimization.

- extended CEF TDB parsing to read ordinary Gibbs-energy, Curie-temperature, and magnetic-moment
  parameters and assemble the corresponding nonmagnetic and magnetic contributions.

- introduced the `TemperatureExpression` interface and the differentiable
  `ParameterizedExpression`. Symbolic expressions use `T` as temperature and treat their other
  symbols as trainable PyTorch parameters; compiled expressions are rebuilt automatically after
  serialization.

- added `SymbolicCEFParser` as an alternative path for constructing complete CEF models directly
  from parameterized TDB expressions while retaining the existing polynomial parser.

- added DataFrame and CSV conversion functions for phase-equilibrium training data. Rows are
  grouped by integer `eq_index`, phases are resolved through a supplied `ThermodynamicSystem`, and
  blank composition rows represent unknown phase compositions.

### command-line workflow

- added the `eqopt-calphad` command-line workflow, which constructs an `EnsembleSystem` from a
  parameterized TDB, loads phase equilibria from CSV, applies optional YAML configuration, and
  writes optimization checkpoints and an optimized TDB.

- added a fully commented reference configuration at `config/calphad-default.yaml` containing all
  current CEF solver, optimization, and runtime defaults.

- renamed the installed distribution and command-line entry point to `eqopt` and
  `eqopt-calphad`, respectively, and updated the package version to 0.2.

## 2026-07-29

- added `prepare_for_loss()` method to thermodynamic models, which can trigger some actions before loss functions are calculated, such as doing some internal minimization.

- added example of `AuCu` in which a semi-empirical EAM potential is optimized based on a simplified binary phase diagram of the system.
