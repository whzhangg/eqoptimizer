# Change log

## 2026-07-29

- added `prepare_for_loss()` method to thermodynamic models, which can trigger some actions before loss functions are calculated, such as doing some internal minimization.

- added example of `AuCu` in which a semi-empirical EAM potential is optimized based on a simplified binary phase diagram of the system.