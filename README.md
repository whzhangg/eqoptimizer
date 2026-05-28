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
In these expressions, we use bold letter as external controbl variables that specified thermodynamic constraints such as $\mathbb{T}$ and $\mathbb{P}$, in constrast with internal variables such as $\mathbf{y}$ and $\mathbf{z}$. However, we require that $\mathbf{y}$ and $\mathbf{z}$ should be consistent with each other so that the state that they define are possible. For example, if $\mathbf{z}$ is the chemical composition $x_{A,B,\cdots}$, then the value of $y$ should be constrained given $\mathbf{z}$:
$$
x_A = X_A(\mathbf{y})
$$ 
where $X_A$ is a function that calculate composition at a coordinate of $\mathbf{y}$. The easiest way to make $\mathbf{y}$ and $\mathbf{z}$ competible with each other is to make $\mathbf{z}$ as a function of $\mathbf{y}$, like above. Nonetheless, we can denote internal parameters collectively as $(\mathbf{y},\mathbf{z})$ generally.

From this definition, we optimize $\mathbf{G}_m^{\alpha}$ to reproduce experimental phase boundary. The benefit of separating $\mathbf{G}_m^{\alpha}$ into two contribution is that we are free to design model for $\mathcal{G}_m^{\alpha}$ without concerned with the reference $G_m^{\alpha}$, which can be obtained by free energy calculation from DFT, machine learning potentials (MLIPs) or CALPHAD assessments. For example, if $G_m^{\alpha}$ already provide a rather accurate free energy, we can choose a simple model for $\mathcal{G}_m^{\alpha}$, for example, a **Redlich–Kister** polynomial. Ignoring $\mathbb{P}$, it can be written as, for a phase:
$$
\mathcal{G}_m^{\mathrm{RF}} (\mathbf{x},\mathbb{T},\mathbb{D})
= \sum_{i=A}^{N} x_i g_i(T) + \sum_{i=A}^{N}\sum_{j>i} x_i x_j \left[\sum_{n=0}^{v}L_{i,j}^{(n)}(T)\cdot (x_i-x_j)^n\right]
$$
The internal degree of freedome is thus just the composition vector of the phase. where both $g_i(T)$ and $L_{i,j}^{(n)}(T)$ can be expressed as polynomial of $T$:
$$
g(T) = \sum_{n=0}^{n_{\mathrm{max}}} a_n \left(\frac{T}{T_{\mathrm{ref}}}\right)^n;\quad 
L(T) = \sum_{n=0}^{n_{\mathrm{max}}'} b_n \left(\frac{T}{T_{\mathrm{ref}}}\right)^n
$$ 
where $n_{\mathrm{max}}$ is the order of temperature dependence, and we can keep relatively low order of interaction parameters. $T_{\mathrm{ref}}$ is introduced to keep coefficients in similar order of magnitude. For compounds, the simplest correction model is just a polynomial without internal degrees of freedom:
$$
\mathcal{G}_m^{\mathrm{comp}}(\mathbb{T},\mathbb{D}) = \sum_{n=0}^{n_{\mathrm{max}}} d_n \left(\frac{T}{T_{\mathrm{ref}}}\right)^n
$$
These are just a simple examples of possible free energy correction model, but it's also possible to use deep machine learning models such as neural networks, that take structure of the phase and composition as input as long as it faciliates the calculation of combined Gibbs energy $\mathbf{G}_m^{\alpha}$.


### Loss function

In the following, we omit the dependence of functions on external parameters for simplicity. At a phase equilibrium, all stable phases share the same chemical potential $\mu_A, \mu_B,\cdots$ where $A$, $B$ and so on index elements. For any phase, we can check if it can be in equilibrium with the system by calculating its grand potential $\Phi_m^{\alpha}$ at these chemical potentials by:
$$
\Phi_m^{\alpha} = \min_{(\mathbf{y},\mathbf{z})} \left[ \mathbf{G}_m^{\alpha}(\mathbf{y},\mathbf{z}) - \sum_A \mu_A M_A^{\alpha}(\mathbf{y},\mathbf{z}) \right]
$$
and:
$$
\begin{align*}
\Phi_m^{\alpha} = \Phi_m^{\beta} = \cdots = 0\quad&\text{for stable phases} \\
\Phi_m^{\gamma} > 0\quad&\text{for unstable phases} \\
\end{align*}
$$

On the other hand, at equilibrium, the chemical potentials themselves can be calculated by the phase equilibrium:
$$
\begin{gather*}
\mathbf{G}_m^{\alpha} (\mathbf{M}^{\alpha}) - \sum_{A}\mu_A M_A^{\alpha} = 0 \\
\mathbf{G}_m^{\beta} (\mathbf{M}^{\beta})- \sum_{A}\mu_A M_A^{\beta} = 0 \\
\cdots
\end{gather*}
$$
where $\mathbf{G}_m^{\alpha} (\mathbf{M}^{\alpha})$ is the Gibbs energy of a single phase $\alpha$.

The above formulation gives a definition for loss function with respect to equilibrium. At observed chemical composition for each phases, given by $\mathbf{M}^{\alpha,\beta,\cdots}$, we first determine a chemical potential $\boldsymbol{\mu}$ from the above equation using composition constrained (single phase) Gibbs energy $\mathbf{G}_m^{\alpha,\beta,\cdots} (\mathbf{M}^{\alpha,\beta,\cdots})$. Then, we calculate the grand potential $\Phi_m^{\alpha,\beta,\cdots}$ for all phases present. For the observed stable phases, we require that their grand potential is zero:
$$
l_1 = \sum_{\alpha\in\mathrm{observed}} \left|\frac{\Phi_m^{\alpha}}{RT}\right|^2
$$
For unobserved phases, we penalize when their grand potential is larger than zero:
$$
l_2 = \sum_{\beta\in\mathrm{unobserved}} \mathrm{ReLU}\left(-\frac{\Phi_m^{\beta}}{RT}\right)
$$
a scaling factor by $RT$ is used in "rough search" in PANDAT and is also introduced here. The final loss function can be written as:
$$
l = \lambda_1 l_1 + \lambda_2 l_2 + \lambda_3 l_{\mathrm{reg}}
$$
where we have added a regularization term $l_{\mathrm{reg}}$, and $\lambda_{1/2/3}$ are weights. To calculate grand potential term $\Phi_m$, we replace the function $\min$ with $\mathrm{Softmin}$, defined by "LogSumExp":
$$
\Phi_m^{\alpha} = -\tau \log \sum_{j} \exp\left[-\frac{\mathbf{G}_m^{\alpha}(\mathbf{y}_j,\mathbf{z}_j) - \sum_A \mu_A M_A^{\alpha}(\mathbf{y}_j,\mathbf{z}_j)}{\tau}\right]
$$

The $\mathrm{Softmin}$ function has the following property, for a sequence of $n$ points $\mathbf{x} = (x_1,x_2,\cdots)$
$$
\mathrm{Softmin}(\mathbf{x},\tau) = -\tau \log\sum_j \exp\left[-\frac{x_j}{\tau}\right] 
$$
is always smaller than the true minimal and is bounded by:
$$
\min(\mathbf{x}) - \tau \log(n) \leq \mathrm{Softmin}(\mathbf{x},\tau) < \min(\mathbf{x})
$$
therefore, the smaller $\tau$, the more accurate to find the minimal within the sequence $\mathbf{x}$. 