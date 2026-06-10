# Torch Compatible Thermodynamic Optimization

## Installation

To install the code, simply install dependencies through `pip` and install directly.

```bash
conda create -n eqopt python=3.12
pip install -e .
```

## Introduction

## Formulation

### Equilibrium condition

### Loss Functions

In this project, we consider the optimization of parameters of thermodynamic *models* given by:
$$
G_M^{\alpha}(\mathbf{y}^{\alpha}, \mathbb{T},\mathbb{P},\mathbb{W}^{\alpha})
$$
where $\mathbf{y}^{\alpha}$ are internal coordinates of the phase $\alpha$. We use blackboard letter as external controbl variables that specified thermodynamic constraints such as $\mathbb{T}$ and $\mathbb{P}$, in constrast with internal variables $\mathbf{y}$. Thermodynamic parameters are denoted as $\mathbb{W}^{\alpha}$. The subscript $M$ indicate that this value $G_M$ is defined for one molar of the cell, in which vacancy could occupy some sites. The amount of chemical species in one molar of the same cell is given by:
$$
M_{A}^{\alpha} = M_{A}^{\alpha}(\mathbf{y}^{\alpha});\quad
M_{B}^{\alpha} = M_{B}^{\alpha}(\mathbf{y}^{\alpha});\quad\cdots
$$
The total amount of chemical species is: $M^{\alpha} = \sum_{i\neq \mathrm{Vac}} M_i^{\alpha}$ and from this, we can calculate thermodynamic properties per atom:
$$
G_m^{\alpha} = \frac{G_M^{\alpha}}{M^{\alpha}}
$$

At a multi-phase equilibrium, all stable phases share the same chemical potential $\mu_A, \mu_B,\cdots$ where $A$, $B$ and so on index elements. For any phase, we can check if it can be in equilibrium with the system by calculating its grand potential $\Phi_m^{\alpha}$ at these chemical potentials by:
$$
\Phi_m^{\alpha}(\mathbb{U},\mathbb{T},\mathbb{P},\mathbb{W}^{\alpha}) = \min_{\mathbf{y}} \left[ \frac{G_M^{\alpha}(\mathbf{y},\mathbb{T},\mathbb{P},\mathbb{W}^{\alpha}) - \sum_A \mathbb{U}_A M_A^{\alpha}(\mathbf{y})}{M^{\alpha}(\mathbf{y}_j)} \right]
$$
and:
$$
\begin{align*}
\Phi_m^{\alpha} = \Phi_m^{\beta} = \cdots = 0\quad&\text{for stable phases} \\
\Phi_m^{\gamma} > 0\quad&\text{for unstable phases} \\
\end{align*}
$$

On the other hand, at equilibrium, the equilibrium composition are related to the chemical potentials themselves by the equilibrium condition:
$$
\begin{gather*}
\mathbf{G}_M^{\alpha} (\mathbb{M}^{\alpha},\mathbb{T},\mathbb{P},\mathbb{W}^{\alpha}) - \sum_{A}\mu_A \mathbb{M}_A^{\alpha} = 0 \\
\mathbf{G}_M^{\beta} (\mathbb{M}^{\beta},\mathbb{T},\mathbb{P},\mathbb{W}^{\beta})- \sum_{A}\mu_A \mathbb{M}_A^{\beta} = 0 \\
\cdots
\end{gather*}
$$
where $\mathbf{G}_M^{\alpha} (\mathbb{M}^{\alpha},\mathbb{T},\mathbb{P},\mathbb{W}^{\alpha})$ is the single phase Gibbs energy subject to external condition $\mathbb{T}$, $\mathbb{P}$ and the given amount of species $\mathbb{M}^{\alpha}$.

The above formulation gives a definition for loss function which should reach zero when the equilibrium is reached. Given a set of chemical potential $\boldsymbol{\mu}$, we calculate the grand potential $\Phi_m^{\alpha,\beta,\cdots}$ for all phases present. For the observed stable phases, we require that their grand potential is zero:
$$
l_1(\mathbb{W},\boldsymbol{\mu}) = \sum_{\alpha\in\mathrm{observed}} \left|\frac{\Phi_m^{\alpha}}{R\mathbb{T}}\right|^2 \to 0
$$
At the same time, since no phase should have grand potential less than zero, for both observed and unobserved phases, we define the penalty, which is zero if $\Phi_m>0$:
$$
l_2(\mathbb{W},\boldsymbol{\mu}) = \sum_{\beta} \mathrm{ReLU}\left(-\frac{\Phi_m^{\beta}}{R\mathbb{T}}\right) \to 0
$$
A scaling with temperature is used in "rough search" in PANDAT and is also introduced here. The chemical potential $\boldsymbol{\mu}$ can be obtained using the equilibrium condition. If $N$ phase are in equilibrium in a $N$ component system (eg. two-phase equilibrium in a binary system, three-phase equilibrium in a ternary system), the chemical potential are uniquely determined from the Gibbs energy $\mathbf{G}_M^{\alpha},\cdots$. In other cases (eg. two-phase equilibrium in a ternary system), we define auxiliary chemical potential vectors $(\mu_A, \mu_B, \cdots)$ for each phase equilibrium, which is minimized during the optimization so that:
$$
l_0(\mathbb{W},\boldsymbol{\mu}) = \sum_{\alpha\in\mathrm{observed}} \left[\frac{\mathbf{G}_M^{\alpha} (\mathbb{M}^{\alpha},\mathbb{T},\mathbb{P},\mathbb{W}^{\alpha}) - \sum_{A}\mu_A \mathbb{M}_A^{\alpha}}{(\sum_A \mathbb{M}_A^{\alpha}) R\mathbb{T}}\right]^2 \to 0
$$
The final loss function, for a single phase equilibrium $\varepsilon$, can thus be written as:
$$
l_{\varepsilon}(\mathbb{W},\boldsymbol{\mu}_{\varepsilon}) = \begin{cases}
\lambda_1 l_1 + \lambda_2 l_2 & \text{if $\boldsymbol{\mu}$ can be determined from $\mathbf{G}_M$} \\
\lambda_0 l_0 + \lambda_1 l_1 + \lambda_2 l_2 & \text{otherwise}
\end{cases}
$$
where we have added a regularization loss $l_{\mathrm{reg}}$, and $\lambda$ are loss weights. The total loss with respect to all considered phase equilibrium, plus a regularization term, can be written as:
$$
l_{\mathrm{tot}} = \sum_{\varepsilon} l_{\varepsilon}(\mathbb{W},\mu_{\varepsilon}) + \lambda_4 l_{\mathrm{reg}}(\mathbb{W})
$$
This loss function should be minimized with respect to the thermodynamic parameter $\mathbb{W}$ and possible auxiliary chemical potential $\mu$ defined for each phase equilibria. A possible regularization loss is: $l_{\mathrm{reg}} = \sum_i |\Delta w_i|^2$, where $\Delta w_i$ is the change of parameters, which follows if expected changes are zero centered.

Before the optimization of thermodynamic parameter $\mathbb{W}$, it maybe benefitial to obtain a good estimate of chemical potential for each phase equilibrium if they are not uniquely determined from the Gibbs energy. It can be done by:
$$
\boldsymbol{\mu}_0 = \arg\min_{\boldsymbol{\mu}} [\lambda_0 l_0 + \lambda_1 l_1 + \lambda_2 l_2]
$$

### Efficient determination of grand potential

In minimization, grand potential for all possible phases need to be evaluated. Thus, we want to be able to determine the grand potential efficient. One approach is to use a $\mathrm{Softmin}$ function (defined as "LogSumExp") instead of $\min$.
$$
\Phi_m^{\alpha} = -\tau \log \sum_{j} \exp\left[-\frac{\mathbf{G}_M^{\alpha}(\mathbf{y}_j,\mathbb{T},\mathbb{P},\mathbb{W}^{\alpha}) - \sum_A \mathbb{U}_A M_A^{\alpha}(\mathbf{y}_j)}{\tau M^{\alpha}(\mathbf{y}_j)}\right]
$$
The $\mathrm{Softmin}$ function has the following property, for a sequence of $n$ points $\mathbf{x} = (x_1,x_2,\cdots)$
$$
\mathrm{Softmin}(\mathbf{x},\tau) = -\tau \log\sum_j \exp\left[-\frac{x_j}{\tau}\right] 
$$
is always smaller than the true minimal and is bounded by:
$$
\min(\mathbf{x}) - \tau \log(n) \leq \mathrm{Softmin}(\mathbf{x},\tau) < \min(\mathbf{x})
$$
The more points we sample, the larger estimation error we will have. If we set an upper bound for error for example $1\, \mathrm{J/mol}$, we can set $\tau\approx 1/\log(n)$. However, if we do not sample densely, we could miss the energy minimal. To sample vector in simplex sample, Dirichlet distribution can be used.

On the other hand, sampling is very important to ensure the accuracy of softmin. A dense uniform sample can still lead to bad estimation of minimum. For example, if an intermetallic phase have narrow composition region but quite a few internal degrees of freedom, it is will be difficult to find samples that in that possible region. To have a better approximation of minimum. We can take the following approach:
$$
\text{Dirichlet sampling of $\mathrm{y}$} \to \text{Local gradient descent} \to \text{Softmin}
$$
We consider the gradient descent step. In CEF, the grand potential term is a function on direct sum of simplex. Consider a function on a simplex $w$, minimization can be done using the exponential gradient descent:
$$
w_i^{(t+1)} = \frac{w_i^{(t)}\exp(-\eta g_i^{t})}{\sum_j w_j^{(t)}\exp(-\eta g_j^{t})}
$$
where $\eta$ is a step size parameter. $w^{t+1}$ remain in the simplex space. When $\eta\to 0$, we have:
$$
\exp(-\eta g) \approx 1-\eta g
$$
and therefore
$$
\begin{align*}
w_i' \approx w_i \frac{1-\eta g_i}{\sum_j w_j - \eta \sum_j w_j g_j} 
&= w_i \frac{1-\eta g_i}{1 - \eta \bar{g}} \\
&\approx w_i (1-ng_i)(1+n\bar{g}) \approx w_i[1-\eta( g_i-\bar{g})]
\end{align*}
$$
to the first order in $\eta$. Therefore, $\Delta w_i\approx -\eta w_i (g_i-\bar{g})$ If we want the maximal step size to be approximately $\delta$, then we can choose $\eta$:
$$
\eta \approx \frac{\delta}{\max_i w_i |g_i -\bar{g}| + \epsilon}
$$
A suitable step size would perhaps be 0.4 and total number of step $6$. The longer the update step, minimal can the calculated more accurately.

### Thermodynamic Models

To calculate the above loss function, thermodynamic model need to be able to calculate Gibbs energy at a given chemical composition $\mathbb{M}$ and to calculate grand-potential $\Phi$ at a given chemical potential $\mathbb{U}$. As long as a model satisfy such requirement, it can be used to calculate the above loss function. By using the $\mathrm{Softmin}$, grand-potential term $\Phi$ can be calculated as long as the model provide a dense sampling of its internal coordinate $\mathbf{x}$. Some examples are given below. For simplicity, we ignore the model dependence on $\mathbb{P}$:

#### Compound Energy Formalism (CEF)

In the compound energy formalism description, we define the site fraction for sublattice $s$ of element $A$ whose value is between 0 and 1:
$$
y_A^{(s)} = \frac{n_A^{(s)}}{n_A^{(s)}+n_B^{(s)}+\cdots}
$$
where $n_A^{(s)}$ is the number of element $A$ per molar formula at the sublattice $s$. The total number of component $A$ in a molar formula unit is obtained by counting the number of $A$ is each sublattices:
$$
M_i = \sum_s N_s y_i^{(s)}; \quad x_i = \frac{M_i}{\sum_j M_j} 
$$
where $N_s$ is the number of sublattice in a molar formula unit. For example, in the case of $\sigma$-phase, one formula unit maybe defined to be a unit cell with 30 atoms, and then: $N_{\mathrm{2a}}=2,N_{\mathrm{4f}}=4, \cdots$. The later expression gives the chemical composition of $A$ in molar fraction. The thermodynamic model is given by:
$$
G_m(\mathbf{y},\mathbb{T}) = \sum_{I} P_{I}(\mathbf{y}) g_I + RT \sum_s N_s \sum_{i=A}^{N} y_i^{(s)}\ln y_i^{(s)} + G_m^{\mathrm{ex}}(\mathbf{y},\mathbb{T})
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

---

### Supplementary

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

#### Exponential gradient descent

For ordinary gradient descent, $\mathbf{x}^{t+1} = \mathbf{x}^{t} - \eta \mathbf{g}$, where $\mathbf{g}$ is the gradient of the target function $f$, it can be equivalent be expressed as a minimization problem [[cmu](https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15850-f20/www/notes/lec19.pdf),[Zeyuan](https://arxiv.org/abs/1407.1537),[Bubeck](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=http://sbubeck.com/Bubeck15.pdf&ved=2ahUKEwjF-NL6gO-UAxVwjK8BHYSMDtsQFnoECAsQAQ&usg=AOvVaw0ZKZnTafCWEWxqWr1Bxjox),[Sham Kakade](https://homes.cs.washington.edu/~sham/courses/stat928/lectures/lecture22.pdf)]:
$$
\begin{align*}
\mathbf{x}^{t+1} &\leftarrow \arg\min_{\mathbf{x}} \left\{ \underbrace{f(\mathbf{x}^{(t)}) + \langle \mathbf{g}, \mathbf{x}-\mathbf{x}^{(t)}\rangle}_{\text{linearized estimate at $\mathbf{x}$}} + \underbrace{\frac{\lambda}{2}||\mathbf{x}-\mathbf{x}^{(t)}||^2}_{\text{penalty}}\right\} \\
&\leftarrow \arg\min_{\mathbf{x}} \left\{ \eta \langle \mathbf{g}, \mathbf{x}\rangle + \frac{1}{2}||\mathbf{x}-\mathbf{x}^{(t)}||^2\right\}
\end{align*}
$$
The penalty term is necessary so that the linearization will be valid. The second line is obtained by removing the part that does not depend on $\mathbf{x}$ and a scaling by $\eta=1/\lambda$. Minimizing the term in the bracket leads to gradient descent:
$$
L = \eta \sum_i g_i x_i + \frac{1}{2} \sum_i (x_i-x_i^{(t)})^2\quad\Rightarrow \quad
\frac{\partial L}{\partial x_i} = \eta g_i + x_i - x_i^{(t)}= 0 \\
x_i = x_i^{(t)} - \eta g_i
$$

Similarly, we can change the penalty term from eculidean norm to divergence measure to reflect the property of the underlying geometry. In simplex, a vector can be interpreted as probability distributions, and based on this, KL divergence measure could be introduced:
$$
\mathbf{x}^{t+1} \leftarrow \arg\min_{\mathbf{x}\in\mathcal{X}} \left\{ \eta \langle \mathbf{g}, \mathbf{x}\rangle + D_{\mathrm{KL}}(\mathbf{x}|\mathbf{x}^{(t)})\right\}
$$
and we have restricted $\mathbf{x}$ to be in the simplex: $\sum_ix_i = 1$. The constrained minimization can be done using the Lagrange:
$$
L = \eta \sum_i g_i x_i + \sum_i x_i \log\left(\frac{x_i}{x_i^{(t)}}\right) + \lambda \left(\sum_ix_i - 1\right)
$$
and we have:
$$
\frac{\partial L}{\partial x_i} = \eta g_i + \log\left(\frac{x_i}{x_i^{(t)}}\right) + 1 + \lambda = 0 \quad \Rightarrow\quad x_i = x_i^{(t)} e^{-\eta g_i - 1 - \lambda} \\
\frac{\partial L}{\partial \lambda} = \sum_i x_i - 1 = 0 \quad\Rightarrow\quad
\sum_i x_i^{(t)} e^{-\eta g_i - 1 - \lambda} = e^{-1-\lambda} \sum_i x_i^{(t)} e^{-\eta g_i} = 1
$$
so that we see $e^{-1-\lambda}$ is just a normalization factor, leading to:
$$
\boxed{
e^{-1-\lambda} = x_i^{(t)} e^{-\eta g_i - 1 - \lambda} = \frac{x_i^{(t)} e^{-\eta g_i}}{\sum_i x_i^{(t)} e^{-\eta g_i}}
}
$$
