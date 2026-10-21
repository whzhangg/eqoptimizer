# Change log

## 2026-10-21

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

## 2026-07-29

- added `prepare_for_loss()` method to thermodynamic models, which can trigger some actions before loss functions are calculated, such as doing some internal minimization.

- added example of `AuCu` in which a semi-empirical EAM potential is optimized based on a simplified binary phase diagram of the system.
