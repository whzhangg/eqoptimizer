# Methodology

## Equilibrium condition

In this project, we consider thermodynamic *models* for a phase $\alpha$ with which energy can be calculated from input variables $T$, $P$ and internal coordinate $\mathbf{y}^{\alpha}$. The model is parameterized by $\mathbb{W}^{\alpha}$:
$$
G_M^{\alpha}(\mathbf{y}^{\alpha}, T, P,\mathbb{W}^{\alpha})
$$
The internal coordinates of the phase $\alpha$ may be required to satisfy a series of constraints: 
$$
C_1^{\alpha}(\mathbf{y}^{\alpha}) = 0,\quad C_2^{\alpha}(\mathbf{y}^{\alpha}) = 0,\quad \cdots
$$
The subscript $M$ in $G_M$ indicate that this value $G_M$ is defined for one molar of the cell, in which vacancy could occupy some sites. The amount of chemical species in one molar of the same cell is given as functions of internal coordinates:
$$
M_{A}^{\alpha} = M_{A}^{\alpha}(\mathbf{y}^{\alpha});\quad
M_{B}^{\alpha} = M_{B}^{\alpha}(\mathbf{y}^{\alpha});\quad\cdots
$$
The total amount of chemical species is: $M^{\alpha} = \sum_{i\neq \mathrm{Vac}} M_i^{\alpha}$ and from this, we can calculate thermodynamic properties per atom:
$$
G_m^{\alpha} = \frac{G_M^{\alpha}}{M^{\alpha}}
$$

We use black-bold to denote independent control variables. We consider a set of phases $\alpha, \beta, \gamma, \delta\cdots$ with phase fraction $\mathcal{N}^{\alpha}, \mathcal{N}^{\beta}\cdots$.
The global equilibrium at given temperature $\mathbb{T}$, pressure $\mathbb{P}$ and total number of atoms for each element $A$: $\mathbb{N}_A$ can be obtained by minimizing thermodynamic model with respect to the constraints:
$$
\mathbf{G}(\mathbb{N}, \mathbb{T}, \mathbb{P}) = \min_{(\mathcal{N}\ge 0,\mathbf{y},T,P)} \left[\sum_{\alpha} \mathcal{N}^{\alpha} G_M^{\alpha}(\mathbf{y}^{\alpha}, T, P,\mathbb{W}^{\alpha})\right] \quad\text{subject to}
\begin{cases}
T = \mathbb{T}\\
P = \mathbb{P}\\
\sum_{\alpha}\mathcal{N}^{\alpha} M_{A}^{\alpha}(\mathbf{y}^{\alpha}) = \mathbb{N}_A, \cdots\\
C_1^{\alpha}(\mathbf{y}^{\alpha}) = 0, \cdots
\end{cases}
$$
It is important to note that we also have inequality constraint $\mathcal{N}\ge 0$. The minimization problem can be solved by found by finding the stationary point of the Largrangian:
$$
\begin{align*}
L &=\sum_{\alpha} \mathcal{N}^{\alpha} G_M^{\alpha}(\mathbf{y}^{\alpha}, \mathbb{T}, \mathbb{P},\mathbb{W}^{\alpha})
+ \sum_A \mu_A \left[ \mathbb{N}_A-\sum_{\alpha}\mathcal{N}^{\alpha} M_{A}^{\alpha}(\mathbf{y}^{\alpha}) \right] + \sum_{\alpha} \sum_k \zeta_k^{\alpha} C_k^{\alpha}(\mathbf{y}^{\alpha}) \\
& = \sum_{\alpha} \mathcal{N}^{\alpha} \Phi_M^{\alpha}(\mathbf{y}^{\alpha}, \boldsymbol{\mu}, \mathbb{T}, \mathbb{P},\mathbb{W}^{\alpha})
+ \sum_A \mu_A \mathbb{N}_A + \sum_{\alpha} \sum_k \zeta_k^{\alpha} C_k^{\alpha}(\mathbf{y}^{\alpha})
\end{align*}
$$
where $T=\mathbb{T}$ and $P=\mathbb{P}$ is solved trivally and is inserted into the Lagrangian. $\boldsymbol{\mu}$ and $\boldsymbol{\zeta}$ are lagrange multiplier for the mass balance constraints and constraints on $\mathbf{y}$. Furthermore, $\boldsymbol{\mu}$ can be shown to be the chemical potential. We have defined a quantity $\Phi$:
$$
\Phi_M^{\alpha}(\mathbf{y}^{\alpha}, \boldsymbol{\mu}, \mathbb{T}, \mathbb{P},\mathbb{W}^{\alpha})
= G_M^{\alpha}(\mathbf{y}^{\alpha}, \mathbb{T}, \mathbb{P},\mathbb{W}^{\alpha}) - \sum_A \mu_A M_{A}^{\alpha}(\mathbf{y}^{\alpha})
$$
The stationary point is given by:
$$
\begin{gather}
\frac{\partial L}{\partial \mathcal{N}^{\alpha}} = \Phi_M^{\alpha}(\mathbf{y}^{\alpha}, \boldsymbol{\mu}, \mathbb{T}, \mathbb{P},\mathbb{W}^{\alpha}) \begin{cases}=0\quad \text{if $\mathcal{N}^{\alpha}>0$}\\
> 0 \quad \text{if $\mathcal{N}^{\alpha}=0$} \end{cases} \\
\frac{\partial L}{\partial y_i^{\alpha}} = \mathcal{N}^{\alpha} \frac{\partial \Phi_M^{\alpha}}{\partial y_i^{\alpha}} + \sum_k \zeta_k \frac{C_k^{\alpha}(\mathbf{y}^{\alpha})}{\partial y_i} = 0 \\
\frac{\partial L}{\partial \mu_A}=\sum_{\alpha}\mathcal{N}^{\alpha} M_{A}^{\alpha}(\mathbf{y}^{\alpha}) - \mathbb{N}_A = 0\quad\quad 
\frac{\partial L}{\partial \zeta_k^{\alpha}}=C_k^{\alpha}(\mathbf{y}^{\alpha}) = 0
\end{gather}
$$
where we note that for stable phase with $\mathcal{N}>0$, we require $\Phi_M^{\alpha}(\mathbf{y}^{\alpha}, \boldsymbol{\mu}, \mathbb{T}, \mathbb{P},\mathbb{W}^{\alpha})=0$, but for unstable phase, this condition does not need to be satisfied. However, since the internal coordinates for phases with $\mathcal{N}=0$ cannot be uniquely determined, the condition for a phase $\gamma$ to be unstable is that:
$$
\boxed{
\min_{\mathbf{y}^{\gamma}} \Phi_M^{\gamma}(\mathbf{y}^{\gamma},\mu,\mathbb{T}, \mathbb{P},\mathbb{W}^{\gamma}) > 0 \quad\quad \text{subject to constraints}
}
$$
On the other hand, for a stable phase $\alpha$, if we determine the following quantity at the determined chemical potential $\mu$:
$$
\min_{\mathbf{y}^{\alpha}} \Phi_M^{\alpha}(\mathbf{y}^{\alpha},\mu,\mathbb{T}, \mathbb{P},\mathbb{W}^{\alpha}) \quad\quad \text{subject to constraints}
$$
we find the solution is given the stationary point:
$$
\frac{\partial \Phi_M^{\alpha}}{\partial y_i^{\alpha}} + \sum_k \zeta_k \frac{\partial C_k^{\alpha} (\mathbf{y}^{\alpha})}{\partial y_i^{\alpha}} = 0 \quad\text{and}\quad C_k^{\alpha} (\mathbf{y}^{\alpha})=0
$$
which is just equation (2) in the equilibrium condition with a factor. So we see that the internal coordinate $\mathbf{y}^{\alpha}$ solved from the the global phase equilibrium also minimizes $\Phi_M^{\alpha}$. Therefore, we can combine the equilibrium conditions (1) and (2) to give the equilibrium condition for a stable phase:
$$
\boxed{
\min_{\mathbf{y}^{\alpha}} \Phi_M^{\alpha}(\mathbf{y}^{\alpha},\mu,\mathbb{T}, \mathbb{P},\mathbb{W}^{\alpha})=0 \quad\quad \text{subject to constraints}
}
$$
The two boxed equation are equivalent condition to equilibrium condition (1) and (2). Together with the mass balance constraints, they allow us to determine the internal coordinate $\mathbf{y}$ and $\boldsymbol{\mu}$. 

## Loss Functions

### Definition

Typically, experimental phase equilibrium data are given by the measured compositions of phases in equililibrum: 
$$
(\mathbb{M}_A^{\alpha},\mathbb{M}_b^{\alpha},\cdots),
(\mathbb{M}_A^{\beta},\mathbb{M}_b^{\beta},\cdots),
\cdots
$$ 
If the experimental equilibrium is indeed reproduced by the thermodynamic model, then, the internal coordinates that satisfy the equilibrium conditions should also satisfy:
$$
M_A^{\alpha}(\mathbf{y}^{\alpha}) = \mathbb{M}_A^{\alpha} \cdots
$$
On the other hand, deviation from equilibrium can be measured also using the equilibrium condition, but at constrained composition of each observed phase. Since the chemical potential is also unknown, we set auxiliary chemical potential vectors $\boldsymbol{\mu}'$ for this phase equilibrium. For the set of chemical potential, we find deviation to the observed phase equilibrium:
$$
\min_{\mathbf{y}^{\alpha}} \Phi_M^{\alpha}(\mathbf{y}^{\alpha},\boldsymbol{\mu}',\mathbb{T}, \mathbb{P},\mathbb{W}^{\alpha})\neq0 \quad\quad \text{subject to $C_k^{\alpha}$}
$$
Furthermore, solving $\mathbf{y}$ in above term should reproduce the observed equilibrium composition. If the above term is computed at the constrained composition $M_A^{\alpha}(\mathbf{y}^{\alpha}) = \mathbb{M}_A^{\alpha}$ and so on, we find that locally:
$$
\min_{\mathbf{y}^{\alpha}} \Phi_M^{\alpha}(\mathbf{y}^{\alpha},\boldsymbol{\mu}',\mathbb{T}, \mathbb{P},\mathbb{W}^{\alpha}) = 
\min_{\mathbf{y}^{\alpha}} G_M^{\alpha}(\mathbf{y}^{\alpha}, \mathbb{T}, \mathbb{P},\mathbb{W}^{\alpha}) - \sum_A \mu'_A \mathbb{M}_{A}^{\alpha}
= \mathbf{G}_M^{\alpha}(\mathbb{M}^{\alpha}, \mathbb{T}, \mathbb{P},\mathbb{W}^{\alpha}) - \sum_A \mu'_A \mathbb{M}_{A}^{\alpha}
$$
where the first term is composition constrained Gibbs energy. For unobserved phase $\gamma$, we still require that:
$$
\min_{\mathbf{y}^{\gamma}} \Phi_M^{\gamma}(\mathbf{y}^{\gamma},\boldsymbol{\mu}',\mathbb{T}, \mathbb{P},\mathbb{W}^{\gamma}) > 0
$$
with the auxiliary chemical potential. The total deviation can be measured as follows:
$$
\begin{align*}
\mathcal{L}(\boldsymbol{\mu}',\mathbb{W}) &= \sum_{\alpha\in\text{stable}} \left[\mathbf{G}_M^{\alpha}(\mathbb{M}^{\alpha}, \mathbb{W}^{\alpha}) - \sum_A \mu'_A \mathbb{M}_{A}^{\alpha}\right]^2 \\ &+ 
\sum_{\alpha\in\text{stable}} \left[\min_{\mathbf{y}^{\alpha}} \Phi_M^{\alpha}(\mathbf{y}^{\alpha},\boldsymbol{\mu}',\mathbb{W}^{\alpha})\right]^2 + \sum_{\gamma\notin\text{stable}} \mathrm{ReLU}\left[-\min_{\mathbf{y}^{\gamma}} \Phi_M^{\gamma}(\mathbf{y}^{\gamma},\boldsymbol{\mu}',\mathbb{W}^{\gamma})\right]
\end{align*}
$$
at a given auxiliary chemical potential and thermodynamic parameters $\mathbb{W}$. The auxiliary chemical potential need to be optimized so that:
$$
\mathcal{L}(\mathbb{W}) = \min_{\boldsymbol{\mu}'} \mathcal{L}(\boldsymbol{\mu}',\mathbb{W}) \to 0
$$
which should be zero when the experimentally observed phase equilibrium is reproduced by model parameter $\mathbb{W}$. Optimized thermodynamic parameter can be found by minimizing $\mathcal{L}(\boldsymbol{\mu}',\mathbb{W})$ with respect to $\mathbb{W}$ and $\boldsymbol{\mu}'$ at the same time.

### Constraining auxiliary chemical potential

When the number of phases in equilibrium is equal to the number of chemical components. The requirement that $\sum_{\alpha\in\text{stable}} \left[\mathbf{G}_M^{\alpha}(\mathbb{M}^{\alpha}, \mathbb{W}^{\alpha}) - \sum_A \mu'_A \mathbb{M}_{A}^{\alpha}\right]^2 = 0$ leads to a set of linear equation from which auxiliary chemical potential can be solved. For example, in a binary system, knowing the composition and composition constrained Gibbs energy of two phases allow us to define a chemical potential hyper-plane (a line in the case of binary system) that pass throughs both points. In such case, chemical potential are a function of model parameter $\mathbb{W}$ through its dependence on the Gibbs energy. 

However, this is not in general possible. Consider a two phase equilibrium in a ternary system. Two points in the energy composition space cannot uniquely define the chemical potential hyperplane. However, the hyperplane can be constrained to cross these two points, so that only one degree of freedom need to be introduced to define the auxiliary potential. This allow us to reduce the number of auxiliary potential term in the minimization.

### Envelope theorem

It can be noted that to calculate the loss function, it is necessary to solve the $\mathbf{y}$ that minimizes the constrained Gibbs energy as well as unconstrained grand potential. Since the optimized $\mathbf{y}$ are a function of $\mathbb{W}$, it may seems that the derivative of the loss with respect to $\mathbb{W}$ require the derivative of $\mathbf{y}$ with respect to $\mathbb{W}$, which require auto-differentiation through the optimization of $\mathbf{y}$. However, this can be avoided by using the envelope theorem. To calculate the derivative of loss $\mathcal{L}$ with respect to parameters $\mathbb{W}$, it is only necessary to use the solved $\mathbf{y}$ as input. The internal coordinates of the phases depends on the details of the model and should allow the description of all possible states in the phase space. In phenomenological CALPHAD models with fixed lattice, internal coordinates are usually the occupancy of components on the defined sublattices. 

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
G_M(\mathbf{y},\mathbb{T}) = \sum_{I} P_{I}(\mathbf{y}) g_I + RT \sum_s N_s \sum_{i=A}^{N} y_i^{(s)}\ln y_i^{(s)} + G_M^{\mathrm{ex}}(\mathbf{y},\mathbb{T})
$$
where the first sum over $I$ is over possible component array specifying the occupancy of sites in the end-members. For example: $I=(AB\cdots)$ with $A$ occupy the first sublattice, etc, and $g_I$ can be interpreted as its end-member energy $g_{AB\cdots}$. The value of $P_I$ is $y_A^{(1)}y_B^{(1)}\cdots$.

Contribution from pairwise excess term can be defined as follows, where $L^{(n)}=L(T)$ is a temperature polynomial:
$$
G^{\mathrm{ex,pair}}_{M,ab\cdots\underbrace{(ij)}_{(s)}\cdots c}
= y_a^{(1)}y_b^{(2)}\cdots (y_i^{(s)}y_j^{(s)})\cdots y_c^{(N)} \left[
\sum_{n=0}^{v}L^{(n)} (y_i^{(s)}-y_j^{(s)})^n
\right]
$$
where the index $[ab\cdots(ij)_{(s)}\cdots c]$ means that the specific term is related to the mixing on the $(s)$-th sublattice with component $i$ and $j$, while all other sublattices are occupied by $a,b,\cdots, c$, respectively. $v$ index the order of the excess term.

For the two sublattice binary mixing, one possible definition is given as follows:
$$
\begin{align*}
G^{\mathrm{ex,2pair}}_{M,ab\cdots\underbrace{(ij)}_{(s)}\cdots\underbrace{(mn)}_{(r)}\cdots c}
= y_a^{(1)}y_b^{(2)}\cdots & (y_i^{(s)}y_j^{(s)})\cdots(y_m^{(r)}y_n^{(r)})\cdots y_c^{(N)} \\
&\times \left[
L^{(0)} + (y_m^{(r)}-y_n^{(r)})L^{(1)} + (y_i^{(s)}-y_j^{(s)})L^{(2)} 
\right]
\end{align*}
$$

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
From this definition, extension to quaternary mixing on the same lattice will be possible. 

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

