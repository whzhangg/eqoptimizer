# eqopt: differentiable thermodynamic optimization in PyTorch

This package supports optimization of CALPHAD-type and non-CALPHAD-type thermodynamic models by PyTorch. CALPHAD-type CEF models are implemented, and new model can be defined according to need of user as a PyTorch `nn.Module` and `ThermodynamicModel`.

## Quick Start

### Installation

This project depends on the following packages: `torch`, `numpy`, `scipy`, `rich`. `ase` is needed for the use of potential. Optionally, `pycalphad` is used for reference equilibrium calculation.

To install the code, simply install dependencies through `pip` and install directly. `conda-forge` can be used to manage the environment:

```bash
conda create -n eqopt python=3.12
conda activate eqopt
conda install torch numpy scipy rich ase pycalphad
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

### Repository Layout

- `examples` contain several runnable examples of the optimizations

- `src/eqopt` is the main package that defines loss function, abstract classes and optimization procedures

- `src/potentials` contains torch-compatible potentials and calculators

## Status and Limitations

This package is under development, and currently:

- Optimization utilizes phase equilibria only, optimization with respect to other forms of experimental data such as thermochemical data is not yet supported.

- Compound energy formalism (CEF) model is supported which includes compound phases, solid solution phases and intermetallics. However, excess physical terms (for example, magnetic contribution) in the Gibbs energy is not yet implemented.

- Loading initial models from a TDB file is possible but is currently done in a very simple way and can be sensitive to the syntax. (see examples)

- Many parameters are set to reasonable default values. However, it is often important to adopt the value of parameters, such as learning rate, to ensure successful optimization.

## Citation

Zhang, W., Crivello, J.-C., Matsuoka, Y., Koyama, T. & Abe, T. Machine Learning Compatible CALPHAD-type Optimization from Phase Equilibria by Auto-differentiation. Preprint at https://doi.org/10.48550/ARXIV.2608.00516 (2026).
