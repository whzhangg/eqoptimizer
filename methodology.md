# Introduction

Phase diagrams are of foundamental importance in materials development, design and optimization. However, as more and more research focus has been placed in multicomponent systems for both functional and structural applications, revelant phases diagrams become less available. 

If some experimental observation are available, CALculation of PHAse Diagram (CALPHAD) assessment can be used to efficiently interpolate between experimental data to determine phase boundaries and predict phase diagrams. In the CALPHAD approach, phenomenological free energy models are parameterized as a function of temperature, pressure and site-fractions. The parameters are determined directly or indirectly by optimization with respect to experimental thermodynamic quantitatives such as heat capacity or observed compositions at phase equilibria. Since smooth polynomials are often used in the thermodynamic models, some interpolation into unknown regions of the phase diagram is made possible.

Without experimental data, phase diagrams can be predicted from first-principle. In a typical approach, free energy of a phase can be modelled using energy and its derivatives from density functional theory (DFT) calculations. Harmonic or quasi-harmonic phonon calculations account for vibrational entropy. Configurational entropy can be approximated by in Bragg-Williams approximation or more accurately based on cluster expansion models. While these methods are readily available, high computational cost of density functional theory (DFT) evaluation hinders their applications to many practical problems. Recently advancement in universal machine learning interatomic potentials (UMLIPs) promises to significantly accelerate the prediction of phase diagram by providing a surrogate energy models that are a few magnitude faster to evaluate. The state-of-the-art UMLIPs achieve near DFT accuracy and thus can be used instead of DFT for free energy calculations. Indeed, recent works have shown its success in different cases. 

However, practical usage of phase diagram depends crucially on the accuracy in describing phase boundaries and phase transitions temperatures. As UMLIPs are typically trained on DFT data calculated at PBE level, machine learning error ($\approx 1\,\mathrm{kJ/mol}$) will be compounded with DFT error ($\approx 1\,\mathrm{kJ/mol}$). As a result, UMLIP predicted phase diagrams often quantitatively deviate from experimental results and cannot be easily amended by further increasing training DFT data size. As a result, unreliable UMLIP phase diagrams may not be attractive to guide experimental effort. 

It is thus clear that DFT or UMLIP predictions of phase diagrams can only be a first approximations to the true phase diagrams determined from experiments. To improve the accuracy of the resulting phase diagrams, further optimization is necessary. A common approach is to combine DFT assessment of free energy with CALPHAD assessment, in which a set of thermodynamic parameters is first derived by fitting with respect to the calculated free energy, and then subsequently optimized by fitting to experimental observations. The same approach can be applied readily without modification to free energy from UMLIPs. However, since UMLIPs can be considered to be thermodynamic models themselves, there is no theoretical obstacles to optimize UMLIP model parameters directly from experimental data especially phase equilibria data.

Practical optimization however, is not straightforward. Typical approach such numerical gradient or blackbox optimization as used in many CALPHAD optimizations can not be reliably used for machine learning model optimization. Recently, analytic gradient based optimization have been developed and implemented by Kunselman et al. However, their formulation is rigid and cannot be extended beyond the CALPHAD framework. Auto-differentiation maybe the most promising route to enable gradient optimization of any thermodynamic models regardless of complexity. In this work, we implemented an auto-differentitation based optimization for optimizing thermodynamic models based on experimental phase equilibria. To use auto-differentiation, a loss function have to be defined and expressed as a differentiable computation graph that are efficient to compute. In this work, we derive a formula of the loss function rigorously from equilibrium condition. Using this loss function and CALPHAD model as example, we show that our implementation allows accurate reproduction of experimental phase equilibrium. The optimization workflow is implemented in pytorch and can thus support any thermodynamic model with different complexity. 

This article is organized as follows: first, we will derived a differentiable loss function from the equilibrium condition. Then, we outline the techniques that enable the calculation of the loss function. Finally, we will show examples of the optimization. A review of related work on CALPHAD optimization is supplied in Supplementary Material.

# Methodology

## 1. Equilibrium condition

In this work, we use $\alpha$, $\beta$ to indicate a phase. A thermodynamic model $G_M^{\alpha}(\mathbf{y}^{\alpha}, T, P,\mathbb{W}^{\alpha})$ for a phase $\alpha$ is a function with which energy can be calculated from input variables $T$, $P$ and a set of internal coordinate $\mathbf{y}^{\alpha}$, which may be required to satisfy a set of constraints $C_k^{\alpha}(\mathbf{y}^{\alpha}) = 0$. The parameters of the model is denoted as $\mathbb{W}^{\alpha}$.
The subscript $M$ in $G_M$ indicate that this value $G_M$ is defined for one molar of the cell, in which vacancy could occupy some sites. The amount of chemical species $(A,B,\cdots)$ in one molar of the same cell is given as functions of internal coordinates:
$$
N_{A}^{\alpha} = N_{A}^{\alpha}(\mathbf{y}^{\alpha});\quad
N_{B}^{\alpha} = N_{B}^{\alpha}(\mathbf{y}^{\alpha});\quad\cdots
$$
The total amount of chemical species is: $N^{\alpha} = \sum_{i\neq \mathrm{Vac}} N_i^{\alpha}$ and from this, we can calculate thermodynamic properties per atom: $G_m^{\alpha} = G_M^{\alpha}/ M^{\alpha}$.

We use black-bold to denote independent control variables. We consider a set of phases $\alpha, \beta, \gamma, \delta\cdots$ with phase fraction $\mathcal{N}^{\alpha}, \mathcal{N}^{\beta}\cdots$.
The global equilibrium at given temperature $\mathbb{T}$, pressure $\mathbb{P}$ and total number of atoms $\mathbb{N}_i$ for each element $i$ can be obtained by minimizing thermodynamic model with respect to the constraints:
$$
\begin{align*}
\mathbf{G}(\mathbb{N}, \mathbb{T}, \mathbb{P}) &= \min_{(\mathcal{N}\ge 0,\mathbf{y},T,P)} \left[\sum_{\alpha} \mathcal{N}^{\alpha} G_M^{\alpha}(\mathbf{y}^{\alpha}, T, P,\mathbb{W}^{\alpha})\right] \\ &\text{subject to}
\begin{cases}
T = \mathbb{T}\\
P = \mathbb{P}\\
\sum_{\alpha}\mathcal{N}^{\alpha} N_{A}^{\alpha}(\mathbf{y}^{\alpha}) = \mathbb{N}_A, \cdots\\
C_1^{\alpha}(\mathbf{y}^{\alpha}) = 0, \cdots
\end{cases}
\end{align*}
$$
It is important to note that we also have inequality constraint $\mathcal{N}\ge 0$, so that for a phase $\gamma$ that is not represent, $\mathcal{N}^{\gamma}=0$. The minimization problem can be solved by found by finding the stationary point of the Largrangian:
$$
\begin{align*}
L &=\sum_{\alpha} \mathcal{N}^{\alpha} G_M^{\alpha}(\mathbf{y}^{\alpha}, \mathbb{T}, \mathbb{P},\mathbb{W}^{\alpha})
+ \sum_A \mu_A \left[ \mathbb{N}_A-\sum_{\alpha}\mathcal{N}^{\alpha} N_{A}^{\alpha}(\mathbf{y}^{\alpha}) \right] + \sum_{\alpha} \sum_k \zeta_k^{\alpha} C_k^{\alpha}(\mathbf{y}^{\alpha}) \\
& = \sum_{\alpha} \mathcal{N}^{\alpha} \Phi_M^{\alpha}(\mathbf{y}^{\alpha}, \boldsymbol{\mu}, \mathbb{T}, \mathbb{P},\mathbb{W}^{\alpha})
+ \sum_A \mu_A \mathbb{N}_A + \sum_{\alpha} \sum_k \zeta_k^{\alpha} C_k^{\alpha}(\mathbf{y}^{\alpha})
\end{align*}
$$
where $T=\mathbb{T}$ and $P=\mathbb{P}$ is solved trivally and can be inserted into the Lagrangian. They are dropped in the following. $\boldsymbol{\mu}$ and $\boldsymbol{\zeta}$ are lagrange multiplier for the mass balance constraints and constraints on $\mathbf{y}$. $\boldsymbol{\mu}$ can be shown to be the chemical potential. We have defined a quantity $\Phi$, which has the form of a grand potential:
$$
\Phi_M^{\alpha}(\mathbf{y}^{\alpha}, \boldsymbol{\mu}, \mathbb{W}^{\alpha})
= G_M^{\alpha}(\mathbf{y}^{\alpha}, \mathbb{W}^{\alpha}) - \sum_A \mu_A N_{A}^{\alpha}(\mathbf{y}^{\alpha})
$$
The stationary point is given by the following set of equations:
$$
\begin{gather}
\frac{\partial L}{\partial \mathcal{N}^{\alpha}} = \Phi_M^{\alpha}(\mathbf{y}^{\alpha}, \boldsymbol{\mu}, \mathbb{W}^{\alpha}) \begin{cases}=0\quad \text{if $\mathcal{N}^{\alpha}>0$}\\
> 0 \quad \text{if $\mathcal{N}^{\alpha}=0$} \end{cases} \\
\frac{\partial L}{\partial y_i^{\alpha}} = \mathcal{N}^{\alpha} \frac{\partial \Phi_M^{\alpha}}{\partial y_i^{\alpha}} + \sum_k \zeta_k \frac{C_k^{\alpha}(\mathbf{y}^{\alpha})}{\partial y_i} = 0 \\
\frac{\partial L}{\partial \mu_A}=\sum_{\alpha}\mathcal{N}^{\alpha} N_{A}^{\alpha}(\mathbf{y}^{\alpha}) - \mathbb{N}_A = 0 \\
\frac{\partial L}{\partial \zeta_k^{\alpha}}=C_k^{\alpha}(\mathbf{y}^{\alpha}) = 0
\end{gather}
$$
where the above four equations are defined for each phase, each internal coordinates, each chemical species and each constraints, respectively.
We note that for stable phase with $\mathcal{N}^{\alpha}>0$, we require $\Phi_M^{\alpha}(\mathbf{y}^{\alpha}, \boldsymbol{\mu}, \mathbb{W}^{\alpha})=0$, but for unstable phase $\gamma$, this condition does not need to be satisfied and thus the internal coordinates for phases with $\mathcal{N}=0$ cannot be uniquely determined. Nonetheless, with determined chemical potential $\boldsymbol{\mu}$, the condition for a phase $\gamma$ to be unstable is that:
$$
\min_{\mathbf{y}^{\gamma},C_k^{\gamma}} \Phi_M^{\gamma}(\mathbf{y}^{\gamma},\mu,\mathbb{W}^{\gamma}) > 0
$$
where $C_k^{\gamma}$ means that constraints on $\mathbf{y}^{\gamma}$ need to be satisfied.
On the other hand, for a stable phase $\alpha$, if we determine the following quantity with $\boldsymbol{\mu}$, we find solving the minimization problem $\min_{\mathbf{y}^{\alpha},C_k^{\alpha}(\mathbf{y}^{\alpha}) = 0} \Phi_M^{\alpha}(\mathbf{y}^{\alpha},\mu,\mathbb{W}^{\alpha})$
lead to the following set the equations which is identical to equation condition:
$$
\frac{\partial \Phi_M^{\alpha}}{\partial y_i^{\alpha}} + \sum_k \zeta_k \frac{\partial C_k^{\alpha} (\mathbf{y}^{\alpha})}{\partial y_i^{\alpha}} = 0 \quad\text{and}\quad C_k^{\alpha} (\mathbf{y}^{\alpha})=0
$$
So we see that the internal coordinate $\mathbf{y}^{\alpha}$ solved from the the global phase equilibrium also minimizes $\Phi_M^{\alpha}$. Therefore, we can combine the equilibrium conditions (1) and (2) to give the equilibrium condition for a stable phase:
$$
\min_{\mathbf{y}^{\alpha},C_k^{\alpha}} \Phi_M^{\alpha}(\mathbf{y}^{\alpha},\mu,\mathbb{W}^{\alpha})=0
$$
The two equations of $\Phi$ are equivalent condition to equilibrium condition (1) and (2), that also allow the determination of internal coordinate $\mathbf{y}$ and $\boldsymbol{\mu}$ when solved together with the mass balance constraints. 

## 2. Loss Functions

### Definition

Typically, experimental phase equilibrium data are given by the measured compositions of phases in equililibrum: 
$$
(\mathbb{N}_A^{\alpha},\mathbb{N}_b^{\alpha},\cdots),
(\mathbb{N}_A^{\beta},\mathbb{N}_b^{\beta},\cdots),
\cdots
$$ 
If an experimental equilibrium is indeed reproduced by the thermodynamic model, then, the internal coordinates that satisfy the equilibrium conditions should also satisfy:
$$
N_A^{\alpha}(\mathbf{y}^{\alpha}) = \mathbb{N}_A^{\alpha} \cdots
$$
Reversely, deviation from equilibrium can also be measured using the equilibrium condition, but at constrained composition of each observed phase. Since the chemical potential is also unknown, we set auxiliary chemical potential vectors $\boldsymbol{\mu}'$ for this phase equilibrium and we find deviation to the observed phase equilibrium:
$$
\min_{\mathbf{y}^{\alpha},C_k^{\alpha}} \Phi_M^{\alpha}(\mathbf{y}^{\alpha},\boldsymbol{\mu}',\mathbb{W}^{\alpha})\neq0
$$
Furthermore, solving $\mathbf{y}$ in above term may not reproduce the observed equilibrium composition. If the above term is computed at the constrained composition $N_A^{\alpha}(\mathbf{y}^{\alpha}) = \mathbb{N}_A^{\alpha}$ and so on, we find that locally:
$$
\begin{align*}
\min_{\mathbf{y}^{\alpha},C_k^{\alpha},\mathbb{N}^{\alpha}} \Phi_M^{\alpha}(\mathbf{y}^{\alpha},\boldsymbol{\mu}',\mathbb{W}^{\alpha}) &= 
\min_{\mathbf{y}^{\alpha},C_k^{\alpha},\mathbb{N}^{\alpha}} G_M^{\alpha}(\mathbf{y}^{\alpha}, \mathbb{W}^{\alpha}) - \sum_A \mu'_A \mathbb{N}_{A}^{\alpha}
\\ &= \mathbf{G}_M^{\alpha}(\mathbb{N}^{\alpha}, \mathbb{W}^{\alpha}) - \sum_A \mu'_A \mathbb{N}_{A}^{\alpha}
\end{align*}
$$
where the first term is composition constrained Gibbs energy. For unobserved phase $\gamma$, we still require that:
$$
\min_{\mathbf{y}^{\gamma},C_k^{\gamma}} \Phi_M^{\gamma}(\mathbf{y}^{\gamma},\boldsymbol{\mu}',\mathbb{W}^{\gamma}) > 0
$$
with the auxiliary chemical potential. Considering the above, the total deviation can be measured as follows:
$$
\begin{align*}
\mathcal{L}(\boldsymbol{\mu}',\mathbb{W}) &= \sum_{\alpha\in\text{stable}} \left[\mathbf{G}_m^{\alpha}(\mathbb{X}^{\alpha}, \mathbb{W}^{\alpha}) - \sum_A \mu'_A \mathbb{X}_{A}^{\alpha}\right]^2 \\ &+ 
\sum_{\alpha\in\text{stable}} \left[\min_{\mathbf{y}^{\alpha},C_k^{\alpha}} \Phi_m^{\alpha}(\mathbf{y}^{\alpha},\boldsymbol{\mu}',\mathbb{W}^{\alpha})\right]^2 \\ &+ \sum_{\gamma\notin\text{stable}} \mathrm{ReLU}\left[-\min_{\mathbf{y}^{\gamma},C_k^{\gamma}} \Phi_m^{\gamma}(\mathbf{y}^{\gamma},\boldsymbol{\mu}',\mathbb{W}^{\gamma})\right]
\end{align*}
$$
at a given auxiliary chemical potential and thermodynamic parameters $\mathbb{W}$. We have normalized the energies to per-molar-atoms and $\mathbb{X}$ is the atomic fraction of the observed phase. The auxiliary chemical potential need to be optimized. Thus, optimal parameters can be found as:
$$
\mathbb{W}_{\mathrm{opt}}= \arg\min_{\mathbb{W}} \left[ \min_{\boldsymbol{\mu}'} \mathcal{L}(\boldsymbol{\mu}',\mathbb{W}) + \lambda \|W\|_2^2 \right]
$$
where we have introduced the regularization with weight $\lambda$. The first term should be zero when the experimentally observed phase equilibrium is reproduced by optimized model parameter. In practice, $\mathbb{W}_{\mathrm{opt}}$ can be found by minimizing $\mathcal{L}(\boldsymbol{\mu}',\mathbb{W})+ \lambda \|W\|_2^2$ with respect to $\mathbb{W}$ and $\boldsymbol{\mu}'$ at the same time.

### Constraining auxiliary chemical potential

When the number of phases in equilibrium is equal to the number of chemical components. The loss $\sum_{\alpha\in\text{stable}} \left[\mathbf{G}_m^{\alpha}(\mathbb{X}^{\alpha}, \mathbb{W}^{\alpha}) - \sum_A \mu'_A \mathbb{X}_{A}^{\alpha}\right]^2 = 0$ leads to a full-rank linear system from which auxiliary chemical potential can be solved. For example, in a binary system, knowing the composition and composition constrained Gibbs energy of two phases allow us to define a chemical potential tangent plane that pass throughs both points in the composition-energy space. In such case, it is possible to define $\boldsymbol{\mu}'$ from the above linear equation so that they are a function of model parameter $\mathbb{W}$, thus eliminating them from optimization, as well as the corresponding loss terms.

However, this is not in general possible. Consider a two phase equilibrium in a ternary system. Two points in the energy composition space cannot uniquely define a chemical potential tangent plane. However, it is possible to force the tangent plane to cross known points from the two phases in equilibrium. Thus, only one degree of freedom need to be introduced to define the auxiliary potential and the first term in the loss function is again eliminated. This allow us to greatly reduce the number of auxiliary potential term that need to be minimized in addition.

### Envelope theorem

Additional minimization with respect to internal coordinates can be found in the defined loss function for both the grand potential term $\Phi$ and Gibbs energy $\mathbf{G}$. Internal coordinates $\mathbf{y}^*$ that minimizes these quantities are themselves a function of $\mathbb{W}$. Fortunately, using the envelope theorem, it is not necessary to evaluate the derivative of $\mathbf{y}^*$ with respect to $\mathbb{W}$. A review of envelop theorem is provided in the Supplementary Materials. To calculate the derivative of loss $\mathcal{L}$ with respect to parameters $\mathbb{W}$, it is only necessary to consider $\mathbf{y}^*$ as input without needing to track the computational graphs that produce $\mathbf{y}^*$.

The internal coordinates of the phases depends on the details of the model and should allow the description of all possible states in the phase space. The simplest coordinates are the composition vector themselves. In phenomenological CALPHAD models with fixed lattice, internal coordinates are usually the occupancy of components on the defined sublattices. The global minimization of internal coordinates for both Gibbs energy and grand potential terms in the framework of CALPHAD can be solved efficient using exponential gradient descend method.

### Final form

By constraining the auxiliary chemical potential, the first term in the loss function disappears. Furthermore, we can normalize the energy terms by $R\mathbb{T}$ and the total number of components ($\Phi_M \to \Phi_m$), to obtain the following loss function:
$$
\mathcal{L}(\boldsymbol{\mu}',\mathbb{W}) =
\lambda_1 \sum_{\alpha\in\text{stable}} \left[\frac{\min_{\mathbf{y}^{\alpha}} \Phi_m^{\alpha}(\mathbf{y}^{\alpha},\boldsymbol{\mu}',\mathbb{W}^{\alpha})}{R\mathbb{T}}\right]^2 + \lambda_2 \sum_{\gamma\in\text{all}} \mathrm{ReLU}\left[-\frac{\min_{\mathbf{y}^{\gamma}} \Phi_m^{\gamma}(\mathbf{y}^{\gamma},\boldsymbol{\mu}',\mathbb{W}^{\gamma})}{R\mathbb{T}}\right] + \lambda_3 l_{\mathrm{reg}}(\mathbb{W})
$$
where the last term is a regularization term. A possible regularization loss is: $l_{\mathrm{reg}} = \sum_i |w_i-\bar{w}_i|^2$, which follows if the prior of the parameter are centered at $\bar{w}_i$. Furthermore, we have also included observed phases in the second term, which does not have any impact for the final results but is found to help optimization.

## Thermodynamic Models

To describe a thermodynamic systems, it is necessary for the model to provide free energy for all possible phases. In the CALPHAD approach, each phase has their own thermodynamic model with different parameters and the thermodynamic system is just an ensemble of defined models. However, it is also possible that a global model provide thermodynamic description of all phases, which will be the case of machine learning interatomic potential.

### Compound energy formalism

In the compound energy formalism description, we define the site fraction for sublattice $s$ of element $A$ denoted as $y_A^{(s)} = n_A^{(s)}/(n_A^{(s)}+n_B^{(s)}+\cdots)$
where $n_A^{(s)}$ is the number of element $A$ per molar formula at the sublattice $s$. The total number of component $A$ in a molar formula unit is obtained by counting the number of $A$ is each sublattices $N_i = \sum_s N_s y_i^{(s)}$ where $N_s$ is the number of sublattice in a molar formula unit. Omitting the pressure, the thermodynamic model is given by:
$$
G_M(\mathbf{y},T) = \sum_{I} P_{I}(\mathbf{y}) g_I + RT \sum_s N_s \sum_{i=A}^{N} y_i^{(s)}\ln y_i^{(s)} + G_M^{\mathrm{ex}}(\mathbf{y},T)
$$
where the first sum over $I$ is over possible component array specifying the occupancy of sites in the end-members. For example: $I=(AB\cdots)$ with $A$ occupy the first sublattice, etc, and $g_I$ can be interpreted as its end-member energy $g_{AB\cdots}$. The value of $P_I$ is $y_A^{(1)}y_B^{(1)}\cdots$. 

The excess energy term can be consist of different terms. Contribution from pairwise excess term can be defined as follows, where $L^{(n)}=L(T)$ is a temperature polynomial:
$$
G^{\mathrm{ex,pair}}_{M,ab\cdots\underbrace{(ij)}_{(s)}\cdots c}
= y_a^{(1)}y_b^{(2)}\cdots (y_i^{(s)}y_j^{(s)})\cdots y_c^{(N)} \left[
\sum_{n=0}^{v}L^{(n)} (y_i^{(s)}-y_j^{(s)})^n
\right]
$$
where the index $ab\cdots(ij)_{(s)}\cdots c$ means that the specific term is related to the mixing on the $(s)$-th sublattice with component $i$ and $j$, while all other sublattices are occupied by $a,b,\cdots, c$, respectively. $v$ index the order of the excess term. 
Ternary mixing on the same site can be given as follows, with order on interaction parameter limited to (0,1,2):
$$
G^{\mathrm{ex,ternary}}_{M,ab\cdots\underbrace{(ijk)}_{(s)}\cdots c}
= y_a^{(1)}y_b^{(2)}\cdots (y_i^{(s)}y_j^{(s)}y_k^{(s)})\cdots y_c^{(N)} \left[
v_i L^{(0)} + v_j L^{(1)} + v_k L^{(2)} 
\right] \\
\begin{cases}
v_i = y^{(s)}_i + (1-y^{(s)}_i-y^{(s)}_j-y^{(s)}_k)/3\\
v_j = y^{(s)}_j + (1-y^{(s)}_i-y^{(s)}_j-y^{(s)}_k)/3\\
v_k = y^{(s)}_k + (1-y^{(s)}_i-y^{(s)}_j-y^{(s)}_k)/3
\end{cases}
$$
and for the two sublattice binary mixing, one possible definition is given as follows:
$$
\begin{align*}
G^{\mathrm{ex,2pair}}_{M,ab\cdots\underbrace{(ij)}_{(s)}\cdots\underbrace{(mn)}_{(r)}\cdots c}
= y_a^{(1)}y_b^{(2)}\cdots & (y_i^{(s)}y_j^{(s)})\cdots(y_m^{(r)}y_n^{(r)})\cdots y_c^{(N)} \\
&\times \left[
L^{(0)} + (y_m^{(r)}-y_n^{(r)})L^{(1)} + (y_i^{(s)}-y_j^{(s)})L^{(2)} 
\right]
\end{align*}
$$

Implementation notes:
- To make the energy competible to `pycalphad` definition, if only the $L^{(0)}$ term is specified, it will be set that $L^{(0)} = L^{(1)} = L^{(2)}$.
- In pycalphad, element order are sorted alphabetically within each sublattice, and this convention is also used.
- For two sublattice mixing, convention of the $L^{(1)}$ and $L^{(2)}$ follows implementation of `pycalphad`, which seems to further refers to result from `ThermoCal`.

#### Constrained minimization of gibbs energy

To find the internal coordinate $\mathbf{y}_i^{\alpha}$ that minimizes the Gibbs energy at the observed composition, we have used used constraint minimization routine "SLSQP" implemented in `SciPy.optimize.minimize`. Gradient calculated from auto-differentiation is used in the minimization. The resulting $\mathbf{y}$ is then inserted to compute the loss function. In terms of CEF, the starting point of the minimization is when the site fraction on each sublattice is set to correspond to the composition of the phase.

#### Determination of grand potential

The grand potential is only constrained in terms of the internal coordinates but not by external variables, utilizing the fact that site fraction $\mathbf{y}$ are all positive and sums up to 1.0, the minimization can be performed using exponential gradient descent. To find the coordinates that minimizes the grand potential. We consider the following approach:
$$
\text{Dirichlet sampling of $\mathrm{y}$} \to \text{Local gradient descent} \to \text{Softmin}
$$
Dirichlet sampling enable a uniform sampling on the simplex space. Next, gradient descent steps are performed for each of the sampled points. In CEF, the grand potential term is a function on direct sum of simplex. Consider a function on $w$ in a simplex, minimization can be done using the exponential gradient descent:
$$
w_i^{(t+1)} = \frac{w_i^{(t)}\exp(-\eta g_i^{t})}{\sum_j w_j^{(t)}\exp(-\eta g_j^{t})}
$$
where $g_i$ is the gradient and $\eta$ is a step size parameter. It can be verified that $w^{t+1}$ remain in the simplex space. Finally, the minimal value can be found by either using
$$
\Phi = \Phi(\mathbf{y} = \arg\min_{\mathbf{y}}\Phi)
$$
or by the $\mathrm{Softmin}$ function (defined as "LogSumExp").
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
The more points we sample, the larger estimation error we will have. If we set an upper bound for error for example $1\, \mathrm{J/mol}$, we can set $\tau\approx 1/\log(n)$.

#### Step size of exponential gradient descent

In the exponential gradient descent, when $\eta\to 0$, we have:
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

## Supplementary

### Weights in the loss function

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

### Exponential gradient descent

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

### Exponential gradient descent with linear constraints.

Gibbs energy minimization require

### Envelope theorem

Envelope theorem ([Kevin Wainwright](https://www.sfu.ca/~wainwrig/Econ331/env-theorem2.pdf)) gives the derivative of a function $V$ that has the following form: 
$$
V(\boldsymbol{\omega})=\min_{\mathbf{x}}f(\mathbf{x},\boldsymbol{\omega})
$$
To see envelope theorem, we first see that $\mathbf{x}^*(\boldsymbol{\omega})=\arg\min_{\mathbf{x}} f$ can be found by the following set of equations:
$$
\frac{\partial f(\mathbf{x},\boldsymbol{\omega})}{\partial x} = 0 \quad\Rightarrow\quad
\mathbf{x}^*(\boldsymbol{\omega})
$$ 
and thus: $V(\boldsymbol{\omega}) = f(\mathbf{x}^*(\boldsymbol{\omega}), \boldsymbol{\omega})$. The derivative of $V$ with respect to parameter $\omega$ is then:
$$
\frac{\partial V(\boldsymbol{\omega})}{\partial \omega} = \frac{\partial f}{\partial \omega} + \sum_i \frac{\partial f}{\partial x_i^*}\frac{\partial x_i^*}{\partial \omega} = \frac{\partial f}{\partial \omega}\quad\text{since}\quad  \frac{\partial f}{\partial x_i^*} = 0
$$
evaluated at the optimal $\mathbf{x}^*(\boldsymbol{\omega})$. If we have constraints on $\mathbf{x}$ given by $g(\mathbf{x},\boldsymbol{\omega})=0$, $\mathbf{x}^*(\boldsymbol{\omega})$ is determined by the stationary point of the lagrangian:
$$
\frac{\partial f}{\partial x} + \lambda \frac{\partial g}{\partial x} = 0; \quad g=0
$$
since at $\mathbf{x}^*(\boldsymbol{\omega})$, the constraints are always satisfied, thus:
$$
\frac{\partial g(\mathbf{x}^*(\boldsymbol{\omega}),\boldsymbol{\omega})}{\partial \omega} = \frac{\partial g}{\partial \omega} + \sum_i \frac{\partial g}{\partial x_i^*}\frac{\partial x_i^*}{\partial \omega} \equiv 0 
$$
The derivative is:
$$
\begin{align*}
\frac{\partial V(\boldsymbol{\omega})}{\partial \omega} &= \frac{\partial f}{\partial \omega} + \sum_i \frac{\partial f}{\partial x_i^*}\frac{\partial x_i^*}{\partial \omega} \\
&= \frac{\partial f}{\partial \omega} + \sum_i \frac{\partial f}{\partial x_i^*}\frac{\partial x_i^*}{\partial \omega} + \lambda \left[ \frac{\partial g}{\partial \omega} + \sum_i \frac{\partial g}{\partial x_i^*}\frac{\partial x_i^*}{\partial \omega} \right] \\
&= \frac{\partial L}{\partial \omega}
\quad \left(=\frac{\partial f}{\partial \omega}\ \text{if $g$ does not explicitly depend on $\omega$}\right)
\end{align*}
$$
again evaluated at $\mathbf{x}^*(\boldsymbol{\omega})$. Thus, the envelope theorem allow us to calculate the derivative of the loss function without requiring full differentiation through the entire minimization.

#### Sampling internal coordinates with constraints

Here, we consider compound energy formalism models in which internal coordinate denoted as $y_{i,A}$ is the site occupancy of component $A$ at the $i$-th sublattice. Given a chemical composition $(x_A,x_B,\cdots)$ summing up to one, we consider random uniform sampling of internal coordinates that follows this given composition. Hit-and-run sampling method is used (see [Chen 2018](https://jmlr.csail.mit.edu/papers/volume19/18-158/18-158.pdf)). We also denote the sublattice multiplicities as $m_i$. First, we can write the following constraints:
$$
\mathbf{y}\ge 0; \quad \quad \sum_A y_{i,A} = 1\ \text{for each sublattice $i$}\\
\sum_i m_i y_{i,A} = x_A \left(\sum_{B\neq \mathrm{Va}}\sum_i m_i y_{i,B}\right) = x_A\left(\sum_i m_i-\sum_i m_i y_{i,\mathrm{Va}})\right) \quad\quad\text{for $(N-1)$ elements $A$}
$$
Since all composition sum to 1, only $N-1$ number of composition constraints are necessary. The linear constraints can be collected so that we can write:
$$
\mathbf{A}\mathbf{y} = \mathbf{B}
$$
Disregarding the inequality constraints. All solutions to the equation can be written in the following form:
$$
\mathbf{y} = \mathbf{y}_0 + \mathbf{Z} \cdot \mathbf{j}
$$
where $\mathbf{Z}$ is the matrix with each column corresponding to basis in the null space of $\mathbf{A}$, and $\mathbf{j}$ is coefficient vector that indicate vectors in the null space. $\mathbf{y}_0$ can be any feasible solution. The sampling is thus performed in a polytope: $\mathbf{y}_0 + \mathbf{Z} \cdot \mathbf{j}\ge 0$.In the hit-and-run sampling, we start from a feasible point, and select a random direction in the null space given by $\mathbf{v} = \mathbf{Z}\cdot \mathbf{j}'$. From $\mathbf{y}_0$, a line is draw in this direction:
$$
\mathbf{y} = \mathbf{y}_0 + t \mathbf{v} \quad\quad t_{\mathrm{min}} \le t \le t_{\mathrm{max}}
$$
where the lower and upper limits given by the requirement that any of the value of $\mathbf{y}\ge 0$, and the next point can draw from a uniform distribution along this line, and this can be continued to draw $N$ samples. After than, a subset of samples can be selected randomly or by fartherest point sampling.

#### Minimization of Gibbs energy

Exponential gradient descent enforces the constraints on the site fraction of $\mathbf{y}$, however, to determine Gibbs energy, $\mathbf{y}$ that need to follow the external composition constraints $(x_A,x_B,\cdots)$. Writing composition constraints as:
$$
\mathbf{A} \mathbf{y} - \mathbf{B} = 0
$$
exactly as above, generalized gradient descent can be performed similar to above but with additional constraints, we explicitly consider the case where we have sublattices denoted by index $s$:
$$
L = \eta \sum_{is} g_{is} y_{is}^{(t+1)} + \sum_{is} y_{is}^{(t+1)} \log\left(\frac{y_{is}^{(t+1)}}{y_{is}^{(t)}}\right) + \sum_s \lambda_s \left(\sum_{i} y_{is}^{(t+1)} - 1\right) + \boldsymbol{\mu} (\mathbf{A} \mathbf{y}^{(t+1)} - \mathbf{B})
$$
where $\boldsymbol{\mu}$ is vector of $N-1$ multiplier. The stationary point can be given by:
$$
\eta g_{is} + \log\left(\frac{y_{is}^{(t+1)}}{y_{is}^{(t)}}\right) + 1 + \lambda_s + \sum_{a} \mu_a A_{a,is} = 0;\quad \sum_iy_{is}^{(t+1)} - 1;\quad \mathbf{A} \mathbf{y}^{(t+1)} = \mathbf{B}
$$
leading to:
$$
\log y_{is}^{(t+1)} = \log y_{is}^{(t)}  -\eta g_{is} - (1 + \lambda_s + \sum_a \mu_a A_{a,is});\quad y_{is}^{(t+1)} = \frac{y_{is}^{(t)} e^{-\eta g_i - \sum_a\mu_a A_{a,is}}}{e^{1+\lambda_s}}
$$
we see that $e^{1+\lambda_s}$ term is the normalizer for each sublattice. So $\lambda_s$ can be elimitated:
$$
y_{is}^{(t+1)} = \frac{y_{is}^{(t)} e^{-\eta g_{is} - \sum_a\mu_a A_{a,{is}}}}{\sum_{j} y_{js}^{(t)} e^{-\eta g_{js} - \sum_a\mu_a A_{a,js}}}
$$
Next, we still need to determine the set of $\mu$. They are determined by the nonlinear set of equations. For constraints indexed by $b$:
$$
F_b(\boldsymbol{\mu}) = \sum_{is} A_{b,is} y_{is}^{(t+1)} - B_b = 0
$$
Newton's method can be used to iteratively solve the non-linear equations. First: we find:
$$
F_b(\boldsymbol{\mu}') + \sum_{a} \frac{\partial F_b(\boldsymbol{\mu}')}{\partial \mu_{a}} \Delta \mu_{a} = 0;\ \cdots \quad \Rightarrow \quad \mathbf{F}(\boldsymbol{\mu}') + \mathbf{J} (\boldsymbol{\mu}') \Delta \boldsymbol{\mu} = 0
$$
where $\mathbf{J}$ is the Jacobian with matrix elements $\mathbf{J}_{ab} = \partial F_a / \partial \mu_b$. The Newton update can be found by $\Delta \boldsymbol{\mu} = -\mathbf{J}^{-1} (\boldsymbol{\mu}') \mathbf{F}(\boldsymbol{\mu}')$.
The Jacobian matrix elements is given by:
$$
\frac{\partial F_b}{\partial \mu_a} = \sum_{is} A_{b,is} \frac{\partial y_{is}^{(t+1)}}{\partial \mu_a} \\
\frac{\partial y_{is}^{(t+1)}}{\partial \mu_a} = y_{is}^{(t+1)} \left( \sum_j A_{a,js} y^{(t+1)}_{js} - A_{a,is}\right)
$$
where we note that $\mathbf{y}^{(t+1)}$ is a function of $\boldsymbol{\mu}$.  

Avoiding repeated initializing of $y$:
- for grand potential, $y$ is independent of any constraints, so it can be cached. However, to be above to remain global, we can perhaps keep all the initial grid plus the evloving grids. In this case, we can guarantee that minimal can be found quite easily because of the warm start while still keep a global prespective.
- The same can be done for $y$ sampled in gibbs potential. we just make a single sampling and let $y$ evolve. However, we need to be careful if the composition constraints start to deviate. perhaps it is fine if we add penalty. We can keep a additional pool of $y$. During the optimization, if some y deviate from composition, we replace it with new ones.
- All the sames are stored detached. 