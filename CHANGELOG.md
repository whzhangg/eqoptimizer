# Change log

## 2026-10-28

- improved grand-potential EGD to reject uphill proposals jointly across sublattices for each
  sample, restore its accepted state and gradient, and halve its step sizes. Acceptance checks
  reuse existing evaluations rather than introducing a full backtracking loop.

- cached grand-potential step sizes alongside live coordinates so warm starts retain progress
  instead of repeating overshooting updates.

- added `grand_potential_initial_steps_factor`, defaulting to 4, to increase the descent budget
  on the first grand-potential evaluation with a fresh runtime cache.

## 2026-10-27

- added minimized internal coordinate point in the Gibbs energy to the grand potential samples to faciliate accurate calculation.

- changed the parser to allow "*" for sublattices.

- disordered reference phases are excluded from the considered phases that can participate in equilibrium. This is done by adding "excluded_phase_ids" to the system.

- added interior simplex `q`-grid sampling for grand-potential minimization, combined with
  optional Dirichlet samples and cached live points.

- separated Gibbs-energy and grand-potential sampling options. Random sample counts now scale
  with the number of internal degrees of freedom.

- separated composition-specific Gibbs runtime caches from phase-specific grand-potential
  caches, preventing repeated occurrences of the same phase from sharing Gibbs states.

- updated TDB export to write optimized `FUNCTION` definitions with their original temperature
  intervals instead of copying the initial definitions unchanged.

## 2026-10-24

- added order-disorder model (split-CEF) documented in '/docs/calphad.md'.

- improved parser to support different ways to specify the thermodynamic model parameters. This is now also documented in '/docs/calphad.md'.

## 2026-10-25

- recursive parsing of the TDB file supporting FUNCTIONs as individual Module with parameter and reuse of FUNCTION in other expressions. Temperature segment is also supported

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
