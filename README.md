# Torch Compatible Thermodynamic Optimization

## Installation

To install the code, simply install dependencies through `pip` and install directly.

```bash
conda create -n eqopt python=3.12
pip install -e .
```

## Formulation

### Gibbs energy model

In this project, a `pytorch` compatible scheme is implemented that can optimize Gibbs energy of phases with respect to global phase equilibrium. In particular, for a system described by internal coordinates $\mathbf{y}^{\alpha}$ and its thermodynamic parameters $\mathbb{W}^{\alpha}$, we define Gibbs energy:
$$
G_m^{\alpha}(\mathbf{y}^{\alpha},\mathbb{T},\mathbb{P},\mathbb{W}^{\alpha})
$$
as a reference Gibbs energy, and seek correction term $\mathcal{G}$ that is parameterized by $\mathbb{D}^{\alpha}$ with its own internal degrees of freedom $\mathbf{z}^{\alpha}$:
$$
\mathcal{G}_m^{\alpha} (\mathbf{z}^{\alpha},\mathbb{T},\mathbb{P},\mathbb{D}^{\alpha})
$$
so that the total Gibbs energy of the phase $\alpha$ can be written as:
$$
\mathbf{G}_m^{\alpha} (\mathbf{y}^{\alpha},\mathbf{z}^{\alpha},\mathbb{T},\mathbb{P},\mathbb{W}^{\alpha},\mathbb{D}^{\alpha}) = G_m^{\alpha} + \mathcal{G}_m^{\alpha}
$$

From this definition, we optimize $\mathbf{G}_m^{\alpha}$ to reproduce experimental phase boundary. The benefit of doing this is that we are free to design model for $\mathcal{G}_m^{\alpha}$ without concerned with the reference $G_m^{\alpha}$, which can be obtained by free energy calculation from DFT, machine learning potentials (MLIPs) or CALPHAD assessments. If $G_m^{\alpha}$ already provide a rather accurate free energy, we can choose a much simpler model for $\mathcal{G}_m^{\alpha}$ for example a Redlich–Kister polynomial, ignoring $\mathbb{P}$:
$$
\mathcal{G}_m^{\alpha} (\mathbf{x}^{\alpha},\mathbb{T},\mathbb{D}^{\alpha})
= \sum_{i=A}^{N} x_i g_i(T) + \sum_{i=A}^{N}\sum_{j>i} x_i x_j \left[\sum_{n=0}^{v}L_{i,j}^{(n)}(T)\cdot (x_i-x_j)^n\right]
$$
The internal degree of freedome is thus just the composition vector of the phase. where both $g_i(T)$ and $L_{i,j}^{(n)}(T)$ can be expressed as polynomial of $T$:
$$
g(T) = \sum_{n=0}^{n_{\mathrm{max}}} a_n T^n;\quad 
L(T) = \sum_{n=0}^{n_{\mathrm{max}}'} b_n T^n 
$$ 
typically only $a+bT$ suffice and we can keep relatively low order of interaction parameters. This is just a simple example, but it's also possible to use deep machine learning models that take a crystal structure as input as long as it returns scalar value.

### Loss function