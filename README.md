# Torch Compatible Thermodynamic Optimization

## Installation

To install the code, simply install dependencies through `pip` and install directly. The necessary packages are `torch`, `pycalphad`, `numpy`, `scipy`, `rich`

```bash
conda create -n eqopt python=3.12
pip install -e .
```

## Usage

A few optimization examples including binary and ternary have been included in the example folder.

## Changes

### 2026-07-29

- added `prepare_for_loss()` method to thermodynamic models, which can trigger some actions before loss functions are calculated, such as doing some internal minimization.
- added example of `AuCu` in which an semi-empirical EAM potential is optimized based on a simplified binary phase diagram of the system.