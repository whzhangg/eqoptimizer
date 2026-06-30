# Methodology

## Loss function

We consider a set of phases $\alpha, \beta, \cdots, \gamma$ in which some of them ($\alpha,\beta,\cdots$) are found to be in equilibrium experimentally at temperature $\mathbb{T}$ and pressure $\mathbb{P}$, while others ($\gamma, \cdots$) are not observed. The observed composition of the phases are denoted as $\mathbb{X}^{\alpha},\mathbb{X}^{\alpha},\cdots$. The loss function associated with this phase equilibrium is given by:
$$
\begin{align*}
\mathcal{L}(\boldsymbol{\mu}',\mathbb{W}) &= \lambda_0 \sum_{\alpha\in\text{stable}} \left|\mathbf{G}_m^{\alpha}(\mathbb{X}^{\alpha}, \mathbb{W}^{\alpha}) - \sum_A \mu'_A \mathbb{X}_{A}^{\alpha}\right|^p \\ &+ 
\lambda_1 \sum_{\alpha\in\text{stable}} \left|\min_{\mathbf{y}^{\alpha},C_k^{\alpha}} \Phi_m^{\alpha}(\mathbf{y}^{\alpha},\boldsymbol{\mu}',\mathbb{W}^{\alpha})\right|^p \\ &+ \lambda_2 \sum_{\gamma\notin\text{stable}} \mathrm{ReLU}\left[-\min_{\mathbf{y}^{\gamma},C_k^{\gamma}} \Phi_m^{\gamma}(\mathbf{y}^{\gamma},\boldsymbol{\mu}',\mathbb{W}^{\gamma})\right]
\end{align*}
$$
where $p$ is the order of the loss, $p=1$ gives linear loss whereas $p=2$ gives squared loss. $\boldsymbol{\mu}'$ are a set of values for each elements that defines auxiliary chemical potentials. $\lambda$ are relative weights of the terms. Utilizing the first term in the loss function, we can reduce the number of axuiliary chemical potentials and elimiate this term directly. In the end, we obtain optimal thermodynamic parameter by:
$$
\begin{align*}
\mathbb{W}_{\mathrm{opt}} 
= \arg\min_{\mathbb{W}} & \left\{\min_{\{\boldsymbol{\mu}'\}} \left(
   \lambda_1 \sum_t \sum_{\alpha\in\text{stable}^t} \left|\min_{\mathbf{y}^{\alpha},C_k^{\alpha}} \Phi_m^{\alpha}(\mathbf{y}^{\alpha},\boldsymbol{\mu}_t',\mathbb{W}^{\alpha})\right|^p \right.\right.
   \\ & \left.\left. + \lambda_2 \sum_t \sum_{\gamma\notin\text{stable}^t} \mathrm{ReLU}\left[-\min_{\mathbf{y}^{\gamma},C_k^{\gamma}} \Phi_m^{\gamma}(\mathbf{y}^{\gamma},\boldsymbol{\mu}_t',\mathbb{W}^{\gamma}) \right]
\right) + \lambda_{\mathrm{reg}} \|W\|_2^2 \right\}
\end{align*}
$$
where the last term is regularization term and it should be noted that $\boldsymbol{\mu}_t'$ now depends on the Gibbs energy at observed composition. This optimization is solved by gradient descent optimization with Adam optimizer. The Gibbs energy and the grand potential term is calculated using Softmin over a set of sampled coordinates, which are locally optimized to ensure that true minimum is found. This is summarized by the following formula.
$$
\text{Sampling of $\mathbf{y}$} \to \text{Local gradient descent} \to \text{Softmin}
$$
In practice, for each phase in a given phase equilibrium, we store a set of initial samples of $\mathbf{y}_0$ to avoid repeated generation of samples. At the end of local gradient descent, the best $N$ samples are cached is included in the calculation of potential at the next epoch. This ensures some degree of continuity of minimal between the epoches, and also help to better locate minimal in case limited gradient descent step in each epoch does not reach the true minimal. 

## Regularization weight

Maximum a posteriori estimate of model parameter gives the form of loss function:
$$
E(\mathbf{w}) = \frac{1}{2\sigma^2} \sum_{n=1}^{N}[y(x_n,\mathbf{w})-t_n]^2 + \frac{1}{2s^2}\sum_i w_i^2
$$
where $[y(x_n,\mathbf{w})-t_n]$ is the difference between model prediction and target, and $\sigma^2$ is its variance. $s^2$ is the variance of the prior zero-centered distribution of parameter. 

In our implementation of CEF, the model is parameterized in the form of temperature polynomials and parameters $\mathbb{W}$ are the coefficients. We can rewrite the polynomial in the form of:
$$
\begin{align*}
L(T) &= a + bT + cT^2 + dT^3 + \cdots \\
&= a + b\left(\frac{T}{T_{\mathrm{ref}}}\right) + c\left(\frac{T}{T_{\mathrm{ref}}}\right)^2 + d\left(\frac{T}{T_{\mathrm{ref}}}\right)^3 + \cdots 
\end{align*}
$$
where $T_{\mathrm{ref}}$ is a fixed reference temperature and all parameters will have the unit of energy. In the case of parameterized corrections as described in the article, the distribution of parameter is likely to be zero centered Gaussian. If we assume that the variance $s$ of these parameters are about $5000$ J/mol, and further that the variance $\sigma$ is small $\approx 2$ J/mol, we have rougly:
$$
\lambda_{\mathrm{reg}} \approx 1.6\times 10^{-7} \lambda_1
$$

## Initial sampling of the degrees of freedom

We consider the case of compound energy formalism when the internal coordinates are site occupancy that should satisfy the constraints on $\mathbf{y}$: 
$$0 \le \mathbf{y} \quad \text{and}\quad \sum_i y_i^{(s)} = 1$$
Without further constraints on $\mathbf{y}$, a random sampling can be done using the [Dirichlet sampling](https://en.wikipedia.org/wiki/Dirichlet_distribution). The number of points to be sampled should cover the multi-dimensional space with some density. Here, given a parameter $N$, the total number of points to be sampled is implemented as:
$$
N_{\mathrm{total}} = \frac{N^{h-1}}{(h-1)!}
$$
where $h$ is the number of degree of freedom $\mathbf{y}$. For $h=1$, $N_{\mathrm{total}}=1$ as there is no degree of freedom ($y=1$). For $h=2$, $N_{\mathrm{total}}=N$, and for $h=3$, $N_{\mathrm{total}}=N^2/2$.

If we have composition constraints, as required to calculate the Gibbs energy, we need to sample under constraints using Monte-Carlo methods. One simple approach is the hit and run method (see [Chen 2018](https://jmlr.csail.mit.edu/papers/volume19/18-158/18-158.pdf)). The internal coordinate can be denoted as $y_{i,A}$, the site occupancy of component $A$ at the $i$-th sublattice. We also denote the sublattice multiplicities as $m_i$. Given a chemical composition $(\mathbb{X}_A,\mathbb{X}_B,\cdots)$ summing up to one, we have $N-1$ constrains. For on the the element $A$, the constraint is written as:
$$
\sum_i m_i y_{i,A} = \mathbb{X}_A \left(\sum_{B\neq \mathrm{Va}}\sum_i m_i y_{i,B}\right) = \mathbb{X}_A\left(\sum_i m_i-\sum_i m_i y_{i,\mathrm{Va}})\right)
$$
These constraints are linear and they can be collected, together with the constraints that $\sum_i y_i^{(s)} = 1$, as:
$$
\mathbf{A}\mathbf{y} = \mathbf{B}
$$
Since the degrees of freedom is in general larger than the number of composition constraints, all possible solutions to the above equation can be written in the following form:
$$
\mathbf{y} = \mathbf{y}_0 + \mathbf{Z} \cdot \mathbf{j}
$$
where $\mathbf{Z}$ is the matrix with each column corresponding to basis in the null space of $\mathbf{A}$, and $\mathbf{j}$ is coefficient vector that indicate vectors in the null space. $\mathbf{y}_0$ can be any feasible solution. Adding inequality constraints, The sampling is thus performed in a polytope: $\mathbf{y}_0 + \mathbf{Z} \cdot \mathbf{j}\ge 0$.

In the hit-and-run sampling, we start from a feasible point, and select a random direction in the null space given by $\mathbf{v} = \mathbf{Z}\cdot \mathbf{j}'$. From $\mathbf{y}_0$, a line within the polytope is than draw in this direction:
$$
\mathbf{y} = \mathbf{y}_0 + t \mathbf{v} \quad\quad t_{\mathrm{min}} \le t \le t_{\mathrm{max}}
$$
where the lower and upper limits given by the requirement that any of the value of $\mathbf{y}\ge 0$. A new next point can then draw from a uniform distribution along this line, which becomes the next $\mathbf{y}_0$. This can be continued until $N$ samples were draw. After that, a subset of samples can be selected randomly or by fartherest point sampling.

## Minimization of Grand Potential

The detail of exponential gradient descend (EGD) with/without constraints are described in the manuscript. Some additonal reference can be found at [[cmu](https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15850-f20/www/notes/lec19.pdf),[Zeyuan](https://arxiv.org/abs/1407.1537),[Bubeck](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=http://sbubeck.com/Bubeck15.pdf&ved=2ahUKEwjF-NL6gO-UAxVwjK8BHYSMDtsQFnoECAsQAQ&usg=AOvVaw0ZKZnTafCWEWxqWr1Bxjox),[Sham Kakade](https://homes.cs.washington.edu/~sham/courses/stat928/lectures/lecture22.pdf)]. 

For the minimization in grand potential, no constraints are necessary when using the EGD formal since all steps are normalized and are positive. The gradient descent is given by the formula:
$$
y_i^{(t+1)} = \frac{y_i^{(t)}\exp(-\eta g_i^{(t)})}{\sum_j y_j^{(t)}\exp(-\eta g_j^{(t)})}
$$
where $\mathbf{g}^{(t)}$ is the gradient at step $t$. The step size parameter $\eta$ can be estimated as follows. When $\eta\to 0$, we have:
$$
\exp(-\eta g) \approx 1-\eta g
$$
and for simplicity, writing $y_i' = y_i^{(t+1)}$ and $y_i=y_i^{(t)}$, we have
$$
\begin{align*}
y_i' \approx y_i \frac{1-\eta g_i}{\sum_j y_j - \eta \sum_j y_j g_j} 
&= y_i \frac{1-\eta g_i}{1 - \eta \bar{g}} \\
&\approx y_i (1-ng_i)(1+n\bar{g}) \approx y_i[1-\eta( g_i-\bar{g})]
\end{align*}
$$
to the first order in $\eta$. Therefore, the change is $\Delta y_i\approx -\eta y_i (g_i-\bar{g})$. If we want the maximal step size to be approximately $\delta$, then we can choose $\eta$:
$$
\eta \approx \frac{\delta}{\max_i (y_i |g_i -\bar{g}|) + \epsilon}
$$
In the implementation, a suitable step size $\delta$ is chosen to be 0.4 and total number of step is chosen to be $6$. The longer the update step, minimal can the calculated more accurately.

## Minimization of Gibbs energy

In this work, EGD is also used to minimize Gibbs energy with respect to $\mathbf{y}$. We denote composition (linear) constraint by $\mathbf{A} \mathbf{y} - \mathbf{B} = 0$. As described in the article, the next step in the minimization is given by:
$$
y_{is}^{(t+1)} = \frac{y_{is}^{(t)} e^{-\eta g_{is} - \sum_a\mu_a A_{a,{is}}}}{\sum_{j} y_{js}^{(t)} e^{-\eta g_{js} - \sum_a\mu_a A_{a,js}}}
$$
where $y_{is}$ is the site fraction of component $i$ on sublattice $s$. In this equation, $\mu$ is determined so that the composition constraints are satisfied. They are determined by the nonlinear set of equations. For constraints indexed by $b$:
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

Determine $\mu$ accurate is very important to ensure that composition constraints are followed. In the implementation, we have used Newton method with back-tracking to ensure that a solution is found robustly. If the composition constraints are still violated after a given number of Newton steps, Newton step size is decreased and $\mu$ is determined again. 

In the case of Gibbs energy, the feasibility of input equilibrium composition is also important. If the input composition is not feasible due to numerical inaccuracy for example, the above minimization with constraints will fail since no point will satisfy the imposed composition. Thus, before minimization, a projection is done to make compositions feasible. Error will be reported if the projected composition deviated from the input composition by a certain threshold.

## $\mathrm{Softmin}$ function

As stated in the article, $\mathrm{Softmin}$ function is used instead of taking the true $\min$. While it may increase computational cost slightly, it's more robust when there are multiple minimia that are very close in energy. The $\mathrm{Softmin}$ function is given by the following, for a sequence of $n$ points $\mathbf{x} = (x_1,x_2,\cdots)$:
$$
\mathrm{Softmin}(\mathbf{x},\tau) = -\tau \log\sum_j \exp\left[-\frac{x_j}{\tau}\right] 
$$
It has the following property that it is always smaller than the true minimal and the function value is bounded by:
$$
\min(\mathbf{x}) - \tau \log(n) \leq \mathrm{Softmin}(\mathbf{x},\tau) < \min(\mathbf{x})
$$
The lower bound can be reached when all values of $\mathbf{x}$ are equal. The $\tau$ parameter controls the contribution of points above the minimium. From the formula, we see that the more points we sample, the larger estimation error we will have. If we set an upper bound for error for example $1\, \mathrm{J/mol}$, we can set $\tau\approx 1/\log(n)$. 

## Thermodynamic Models

To describe a thermodynamic systems, it is necessary for the model to provide free energy for all possible phases. In this work, we consider that phases are associated with a unique identifier (`PhaseID` in the implementation). A model of the thermodynamic system (`ThermodynamicSystem` in the implementation) defines free energy of possible phases and should be able to provide Gibbs energy of any of the defined phases given an input composition and grand potential given an input chemical potential. 

In the CALPHAD approach, each phase has their own thermodynamic model with different parameters and the thermodynamic system is just an ensemble (`EnsembleSystem(ThermodynamicSystem)`) of defined single phase models (`ThermodynamicModel`). Thus, the thermodynamic potential of each phase by calculated by the associated model. In particular, we have defined the class `CEF(ThermodynamicModel)` for compound energy formalism models, the detail of which is given in the manuscript. However, it is also possible that a global model provide thermodynamic description of all phases, which may be the case of machine learning interatomic potential.

Regarding the implementation of compound energy formalism, a few details are considered to make the resulting energy identifical to pycalphad:

- In pycalphad, element order are sorted alphabetically within each sublattice, and this convention is also used.
- To make the energy competible to pycalphad definition, if only the $L^{(0)}$ term is specified for the ternary interaction, it will be set that $L^{(0)} = L^{(1)} = L^{(2)}$.
- For two sublattice mixing, convention of the $L^{(1)}$ and $L^{(2)}$ ($L^{(1)}$ for the second sublattice of mixing and $L^{(2)}$ for the first) follows implementation of pycalphad, which seems to further refers to result from ThermoCal.
