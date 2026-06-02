# Torch Compatible Thermodynamic Optimization

## Installation

To install the code, simply install dependencies through `pip` and install directly.

```bash
conda create -n eqopt python=3.12
pip install -e .
```

## Formulation

### Loss Functions

In this project, we consider the optimization of parameters of thermodynamic models given by:
$$
\mathbf{G}_M^{\alpha}(\mathbf{x}^{\alpha}, \mathbb{T},\mathbb{P},\mathbb{W}^{\alpha})
$$
where $\mathbf{x}^{\alpha}$ are internal coordinates of the phase $\alpha$. We use bold letter as external controbl variables that specified thermodynamic constraints such as $\mathbb{T}$ and $\mathbb{P}$, in constrast with internal variables $\mathbf{x}$. Thermodynamic parameters are denoted as $\mathbb{W}^{\alpha}$.

At a multi-phase equilibrium, all stable phases share the same chemical potential $\mu_A, \mu_B,\cdots$ where $A$, $B$ and so on index elements. For any phase, we can check if it can be in equilibrium with the system by calculating its grand potential $\Phi_m^{\alpha}$ at these chemical potentials by:
$$
\Phi_M^{\alpha}(\mathbb{U},\mathbb{T},\mathbb{P},\mathbb{W}^{\alpha}) = \min_{\mathbf{x}} \left[ \mathbf{G}_m^{\alpha}(\mathbf{x},\mathbb{T},\mathbb{P},\mathbb{W}^{\alpha}) - \sum_A \mathbb{U}_A M_A^{\alpha}(\mathbf{x}) \right]
$$
where $M_A^{\alpha}$ is the number of element $A$ in molar formula unit of phase $\alpha$, and:
$$
\begin{align*}
\Phi_M^{\alpha} = \Phi_M^{\beta} = \cdots = 0\quad&\text{for stable phases} \\
\Phi_M^{\gamma} > 0\quad&\text{for unstable phases} \\
\end{align*}
$$

On the other hand, at equilibrium, the chemical potentials themselves can be calculated by the phase equilibrium:
$$
\begin{gather*}
\mathbf{G}_M^{\alpha} (\mathbb{M}^{\alpha},\mathbb{T},\mathbb{P},\mathbb{W}^{\alpha}) - \sum_{A}\mu_A \mathbb{M}_A^{\alpha} = 0 \\
\mathbf{G}_M^{\beta} (\mathbb{M}^{\beta},\mathbb{T},\mathbb{P},\mathbb{W}^{\beta})- \sum_{A}\mu_A \mathbb{M}_A^{\beta} = 0 \\
\cdots
\end{gather*}
$$
where $\mathbf{G}_M^{\alpha} (\mathbb{M}^{\alpha},\mathbb{T},\mathbb{P},\mathbb{W}^{\alpha})$ is the single phase Gibbs energy subject to external condition $\mathbb{T}$, $\mathbb{P}$ and single phase composition $\mathbb{M}^{\alpha}$.

The above formulation gives a definition for loss function with respect to equilibrium. At observed chemical composition for each phases, given by $\mathbb{M}^{\alpha,\beta,\cdots}$, we first determine a chemical potential $(\mu_A,\mu_B,\cdots)$ from the above equation using composition constrained (single phase) Gibbs energy $\mathbf{G}_m^{\alpha} (\mathbb{M}^{\alpha},\mathbb{T},\mathbb{P},\mathbb{W}^{\alpha})$ calculated at the observed equilibrium composition. Then, we calculate the grand potential $\Phi_m^{\alpha,\beta,\cdots}$ for all phases present using the determined $\mathbb{U}=\mu$. For the observed stable phases, we require that their grand potential is zero:
$$
l_1 = \sum_{\alpha\in\mathrm{observed}} \left|\frac{\Phi_M^{\alpha}}{R\mathbb{T}}\right|^2
$$
For unobserved phases, we penalize when their grand potential is larger than zero:
$$
l_2 = \sum_{\beta\in\mathrm{unobserved}} \mathrm{ReLU}\left(-\frac{\Phi_m^{\beta}}{R\mathbb{T}}\right)
$$
a scaling with temperature is used in "rough search" in PANDAT and is also introduced here. The final loss function can be written as:
$$
l = \lambda_1 l_1 + \lambda_2 l_2 + \lambda_3 l_{\mathrm{reg}}
$$
where we have added a regularization loss $l_{\mathrm{reg}}$, and $\lambda_{1/2/3}$ are loss weights. The regularization loss is: $l_{\mathrm{reg}} = \sum_i |d_i|^2$. To calculate grand potential term $\Phi_m$, we replace the function $\min$ with $\mathrm{Softmin}$, defined by "LogSumExp":
$$
\Phi_M^{\alpha} = -\tau \log \sum_{j} \exp\left[-\frac{\mathbf{G}_M^{\alpha}(\mathbf{x}_j) - \sum_A \mathbb{U}_A M_A^{\alpha}(\mathbf{x}_j)}{\tau}\right]
$$

### Thermodynamic Models

To calculate the above loss function, thermodynamic model need to be able to calculate Gibbs energy at a given chemical composition $\mathbb{M}$ and to calculate grand-potential $\Phi$ at a given chemical potential $\mathbb{U}$. As long as a model satisfy such requirement, it can be used to calculate the above loss function. By using the $\mathrm{Softmin}$, grand-potential term $\Phi$ can be calculated as long as the model provide a dense sampling of its internal coordinate $\mathbf{x}$. Some examples are given below. For simplicity, we ignore the model dependence on $\mathbb{P}$:

#### 1. Direct implemented models

**Compounds:** For line compounds without internal degree of freedom, its thermodynamic model can be simply its single phase Gibbs energy:
$$
\mathbf{G}_M(\mathbb{T},\mathbb{W}) = \sum_{n=0}^{n_{\mathrm{max}}} w_n \left(\frac{T}{T_{\mathrm{ref}}}\right)^n
$$
where $n_{\mathrm{max}}$ is the order of temperature dependence, and we introduce $T_{\mathrm{ref}}$ to keep coefficients in similar order of magnitude. 
 
**Solid solutions:** Redlich-Kister polynomial combined with ideal entropy of mixing can be used to describe the single phase Gibbs energy of solid solution as a function of composition:
$$
\mathbf{G}_m (\mathbf{y},\mathbb{T},\mathbb{W})
= N\left(\sum_{i=A}^{N} y_i g_i(T) + RT\sum_{i=A}^N y_i \ln y_i + \sum_{i=A}^{N}\sum_{j>i} y_i y_j \left[\sum_{n=0}^{v}L_{i,j}^{(n)}(T)\cdot (y_i-y_j)^n\right]\right)
$$
if we have $N$ atoms in a formula unit. The internal degree of freedome $0\leq \mathbf{y} \leq 1$ and $\sum_i y_i = 1$. Sometimes, ternary terms can be added:
$$
\sum_i\sum_{j>i}\sum_{k>j} y_iy_jy_k L_{ijk} + \sum_i\sum_{j>i}\sum_{k>j}\sum_{f>k} y_iy_jy_ky_f L_{ijkf} + \cdots 
$$
as outlined by Lukas et al 2007. 
The number of atoms and the composition (atomic fraction) are given by:
$$
M_i = Ny_i;\quad x_i = \frac{M_i}{\sum_{j\neq\mathrm{va}} M_j}
$$
which coincide with $y_i$ if there is no vacancy. In the case of two or three components, the third term (excess term) can be written as:
$$
\begin{align*}
G^{\mathrm{ex}}_{(2)} &= y_A y_B \sum_n L_{A,B}^{(n)} (y_A-y_B)^n  \\
G^{\mathrm{ex}}_{(3)} &= y_A \left[ y_B \sum_n L_{A,B}^{(n)} (y_A-y_B)^n + y_C \sum_n L_{A,C}^{(n)} (y_A-y_C)^n\right] + y_B \left[y_C \sum_n L_{B,C}^{(n)} (y_B-y_C)^n\right]
\end{align*}
$$

**CEF:** In the compound energy formalism description, we define the site fraction for sublattice $s$ of element $A$ whose value is between 0 and 1:
$$
y_A^{(s)} = \frac{n_A^{(s)}}{n_A^{(s)}+n_B^{(s)}+\cdots}
$$
where $n_A^{(s)}$ is the number of element $A$ per molar formula at the sublattice $s$. The total number of component $A$ in a molar formula unit is obtained by counting the number of $A$ is each sublattices:
$$
M_i = \sum_s N_s y_i^{(s)}; \quad x_i = \frac{M_i}{\sum_j M_j} 
$$
where $N_s$ is the number of sublattice in a molar formula unit. For example, in the case of $\sigma$-phase, one formula unit maybe defined to be a unit cell with 30 atoms, and then: $N_{\mathrm{2a}}=2,N_{\mathrm{4f}}=4, \cdots$. The later expression gives the chemical composition of $A$ in molar fraction. The thermodynamic model is given by:
$$
\mathbf{G}_m(\mathbf{y},\mathbb{T}) = \sum_{I} P_{I}(\mathbf{y}) g_I + RT \sum_s N_s \sum_{i=A}^{N} y_i^{(s)}\ln y_i^{(s)} + G_m^{\mathrm{ex}}(\mathbf{y},\mathbb{T})
$$
where the first sum over $I$ is over possible component array specifying the occupancy of sites in the end-members. For example: $I=(AB\cdots)$ with $A$ occupy the first sublattice, etc, and $g_I$ can be interpreted as its end-member energy $g_{AB\cdots}$. The value of $P_I$ is $y_A^{(1)}y_B^{(1)}\cdots$. For the excess Gibbs energy, we first define the pairwise mixing on the same sublattice. 
$$
G_m^{\mathrm{ex, pair}} = \sum_s N_s \left[\sum_{I_{\mathbf{t}}}\prod_i y_{k_i}^{(s_i)} \left( \sum_{i=A}^{N}\sum_{j>i} y_i^{(s)} y_j^{(s)} \mathcal{L}_{ij:I_{\mathbf{t}}}^{(s):(\mathbf{t})} \right) \right];\quad
\mathcal{L}_{ij:I_{\mathbf{t}}}^{(s):(\mathbf{t})} = \sum_{n=0}^{v}L_{ij:I_{\mathbf{t}}}^{(s):(\mathbf{t})} (y_i^{(s)}-x_j^{(s)})^n
$$
where we first sum over all sublattices. For mixing on sublattice $s$, it is weighted with respect to the possible occupancy of all other sublattices (denoted as $\mathbf{t}$) with weights given by the product $\prod_i y_{k_i}^{(s_i)}$. $I_{\mathbf{t}}=(k_1,k_2,\cdots)$ index the elements occupying the other sublattices. 

Additional two sublattice mixing can be defined as follows:
$$
G_m^{\mathrm{ex, 2pair}} = \sum_s \sum_{t> s} \left[\sum_{I_{\mathbf{r}}}\prod_i y_{k_i}^{(s_i)} \left(\sum_{i}\sum_{j>i}\sum_m\sum_{n>m} y_i^{(s)} y_i^{(s)}y_m^{(t)} y_n^{(t)} L^{(s):(t):{\mathbf{r}}}_{ij:mn:I_{\mathbf{r}}}\right) \right] 
$$
where we sum over two-sublattice pairs and weighted on the occupancy of other not selected sublattices. The mixing is over product of two pairs of elements occupying different sites. 

#### 2. Mixture model

It is often the case for thermodynamic optimization to provide corrections to a set of initial thermodynamic models, from experimental results. In this sense, we can write:
$$
\mathbf{G}_m^{\alpha}
= \underbrace{G_m^{\alpha}(\mathbf{y}^{\alpha},\mathbb{T},\mathbb{D}^{\alpha})}_{\mathrm{reference}}
+ \underbrace{\mathcal{G}_m^{\alpha}(\mathbf{z}^{\alpha},\mathbb{T},\mathbb{Q}^{\alpha})}_{\mathrm{correction}}
$$
in which only parameters $\mathbb{Q}$ corresponding to corrections will be optimized while the parameters $\mathbb{W}^{\alpha}$ in the reference model remain unchanged. $G_{m}^{\alpha}$ and $\mathcal{G}_m^{\alpha}$ maybe described by different set of internal coordinates that are consistent with each other so that they define the same state. For example, if $\mathbf{y}$ is the site fraction of sublattices in a compound energy formalism (CEF) model and $\mathbf{z}$ is the atomic fraction, then, clearly they need to be related (The easiest way to make $\mathbf{y}$ and $\mathbf{z}$ consistent with each other is to make $\mathbf{z}$ as a function of $\mathbf{y}$):
$$
\mathbf{x}_A^{\alpha} = \frac{M_A^{\alpha}(\mathbf{y}^{\alpha})}{M_A^{\alpha}(\mathbf{y}^{\alpha})+M_B^{\alpha}(\mathbf{y}^{\alpha})+\cdots}
$$

From this definition, we optimize $\mathbf{G}_m^{\alpha}$ to reproduce experimental phase boundary with respect to $\mathbb{Q}$. The benefit of separating $\mathbf{G}_m^{\alpha}$ into two contribution is that we are free to design model for $\mathcal{G}_m^{\alpha}$ without concerned with the reference $G_m^{\alpha}$, which can be obtained by free energy calculation from DFT, machine learning potentials (MLIPs) or CALPHAD assessments, and formulated in different ways. If $G_m^{\alpha}$ already provide a rather accurate free energy, we can use simple model for $\mathcal{G}_m^{\alpha}$, for example, a Redlich–Kister polynomial for any phases with a wide homogeneity range:
$$
\mathcal{G}_m^{\mathrm{RF}} (\mathbf{x},\mathbb{T},\mathbb{Q})
= \sum_{i=A}^{N} x_i g_i(T) + \sum_{i=A}^{N}\sum_{j>i} x_i x_j \left[\sum_{n=0}^{v}L_{i,j}^{(n)}(T)\cdot (x_i-x_j)^n\right] \\
g(T) = \sum_{n=0}^{n_{\mathrm{max}}} a_n \left(\frac{T}{T_{\mathrm{ref}}}\right)^n;\quad 
L(T) = \sum_{n=0}^{n_{\mathrm{max}}} b_n \left(\frac{T}{T_{\mathrm{ref}}}\right)^n
$$
where we can generally keep low order in terms of $v$ in interaction parameter, and only linear in temperature. For compounds, the simplest correction model is just a polynomial without internal degrees of freedom:
$$
\mathcal{G}_m^{\mathrm{comp}}(\mathbb{T},\mathbb{D}) = \sum_{n=0}^{n_{\mathrm{max}}} d_n \left(\frac{T}{T_{\mathrm{ref}}}\right)^n
$$
Since the term $G_m^{\alpha}(\mathbf{y}^{\alpha},\mathbb{T},\mathbb{D}^{\alpha})$ is treated as a constant reference Gibbs energy surface, optimization can be kept simple.

---

### Additionals

#### Weights in the loss function

Maximum a posteriori estimate of model parameter gives the form of loss function:
$$
E(\mathbf{w}) = \frac{1}{2\sigma^2} \sum_{n=1}^{N}[y(x_n,\mathbf{w})-t_n]^2 + \frac{1}{2s^2}\sum_i w_i^2
$$
where $[y(x_n,\mathbf{w})-t_n]$ is the difference between model prediction and target, and $\sigma^2$ is its variance. $s^2$ is the variance of the prior zero-centered distribution of parameter.
In our case of the loss function, Suppose that we have certain amount of noise in the calculation of $\Phi$ (for example, $50$ J/mol), denoted by $\sigma_{\Phi}$, then the weights $\lambda_1$ can be:
$$
\lambda_1 = \frac{1}{2}\left(\frac{RT}{\sigma_{\Phi}}\right)^2
$$
$\lambda_2$ can set accordingly with a ratio $\eta$: $\lambda_2 = \eta \lambda_1$. For the regularization, if model parameters $\mathbb{D}$ can be interpreted as correction in free energy, then its prior distribution can be a zero-centered Gaussian with standard deviation $s$, thus:
$$
\lambda_3 = \frac{1}{2s^2};\quad\text{or}\quad \frac{\lambda_3}{\lambda_1} = \left(\frac{\sigma}{s RT}\right)^2
$$
In such case, roughly, we can set $s$ to be about $2000$ J/mol and we have $\lambda_3/\lambda_1$ at the order of $10^{-11}$.

#### Softmin

The $\mathrm{Softmin}$ function has the following property, for a sequence of $n$ points $\mathbf{x} = (x_1,x_2,\cdots)$
$$
\mathrm{Softmin}(\mathbf{x},\tau) = -\tau \log\sum_j \exp\left[-\frac{x_j}{\tau}\right] 
$$
is always smaller than the true minimal and is bounded by:
$$
\min(\mathbf{x}) - \tau \log(n) \leq \mathrm{Softmin}(\mathbf{x},\tau) < \min(\mathbf{x})
$$
therefore, the smaller $\tau$, the more accurate to find the minimal within the sequence $\mathbf{x}$. I found that using $\tau\approx 0.1$ lead to reasonable results.
