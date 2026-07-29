# Optimizing CuAu binary EAM

## Introduction

In this example, we finetune an empirical EAM potential from a simplified CuAu binary phase diagram. The EAM potential is given by Gola and Pastewka. Fine-tuning is done by adding correction to the density functions and pair-wise interaction functions, which is linear combination of chebyshev basis functions in an interval multiplied by an envelope function which is zero at the two ends of the interval.

## Fitting CuAu potential

We consider the CuAu binary system and we have set up a very simple phase diagram model where we define three intermetallic phases: $L_{10}$ CuAu, $L_{12}$ CuAu3 and AuCu3. They are modelled only by their formation enthalpy $H$.

We also have the FCC solid solution phase. Their Gibbs energy is described by the following formula:
$$
\begin{align*}
G_{\mathrm{FCC}}(x_{\mathrm{Cu}}) = & x_{\mathrm{Cu}} G_{\mathrm{Cu}} + x_{\mathrm{Au}} G_{\mathrm{Au}} + RT\sum_{i=\mathrm{Cu,Au}}x_i \ln x_i \\ &+ x_{\mathrm{Au}}x_{\mathrm{Cu}} \left( L_{\mathrm{Au,Cu}}^{(0)} + L_{\mathrm{Au,Cu}}^{(1)}(x_{\mathrm{Au}}-x_{\mathrm{Cu}}) + L_{\mathrm{Au,Cu}}^{(2)}(x_{\mathrm{Au}}-x_{\mathrm{Cu}})^2 \right)
\end{align*}
$$
and the parameters $G_{\mathrm{Cu}}$,  $G_{\mathrm{Au}}$, $L_{\mathrm{Au,Cu}}^{(0)}$, $L_{\mathrm{Au,Cu}}^{(1)}$ and $L_{\mathrm{Au,Cu}}^{(2)}$ are all scalar parameters determined by from the energy of the SQS structures. In the end, the simple phase diagram is completely determined by **8 parameters**, which are the energies of calculated structures. 

## EAM potential

We consider the EAM potential to describe the atomistic interaction in the Cu-Au binary. The atomic energy in EAM potential is given by the formula:
$$
E_i = F_{\sigma_i}(\rho_i) + \frac{1}{2} \sum_{j\neq i} \phi_{\sigma_i\sigma_j}(r_{ij})
$$
where $\sigma_i\in(\mathrm{Au},\mathrm{Cu})$ is the species of atom $i$. $F_{\sigma}$ is the embedding function of the species $\sigma$ and $\rho_i$ is the effective electron density at the position of atom $i$. The density is computed by:
$$
\rho_i = \sum_{j\neq i} f_{\sigma_i|\sigma_j}(r_{ij})
$$
where the $f_{\sigma_i|\sigma_j}$ is the contribution of the density from species $\sigma_j$ to $\sigma_i$ as a function of atomic distance and $f_{\sigma_i|\sigma_j}\neq f_{\sigma_j|\sigma_i}$. $\phi_{\sigma_i,\sigma_j}$ is a symmetric function of pairwise potential. Therefore, the atomistic potential is specified by a total of 9 functions:
$$
\begin{gather*}
F_{\mathrm{Au}}(\rho),
F_{\mathrm{Cu}}(\rho), \\
\phi_{\mathrm{AuAu}}(r),
\phi_{\mathrm{AuCu}}(r) = \phi_{\mathrm{CuAu}}(r),
\phi_{\mathrm{CuCu}}(r), \\
f_{\mathrm{Au|Au}}(r),
f_{\mathrm{Au|Cu}}(r),
f_{\mathrm{Cu|Au}}(r),
f_{\mathrm{Cu|Cu}}(r),
\end{gather*}
$$

In the work by Gola and Pastewka, the unary description $f_{\sigma_i|\sigma_i}$, $F_{\sigma_i}$ and $\phi_{\sigma_i\sigma_i}$ is taken from previous fitting. The pairwise interaction is obtained by:
$$
\begin{gather*}
\phi_{\mathrm{AuCu}} = 
\phi_{\mathrm{CuAu}} = \alpha_{\mathrm{Cu}} \phi_{\mathrm{CuCu}} + \alpha_{\mathrm{Au}} \phi_{\mathrm{AuAu}} \\
f_{\mathrm{Au|Cu}} = \beta_{\mathrm{Cu}} f_{\mathrm{Cu|Cu}} + \gamma_{\mathrm{Cu}} f_{\mathrm{Au|Au}}  \\
f_{\mathrm{Cu|Au}} = \beta_{\mathrm{Au}} f_{\mathrm{Au|Au}} + \gamma_{\mathrm{Au}} f_{\mathrm{Cu|Cu}}
\end{gather*}
$$
so that the parameters of the binary interactions are $\alpha, \beta, \gamma$ parameters (6 of them), which are fitted to reproduce experimental value of lattice parameters and formation enthalpy of the compound. The values are given in the article.

## eam.fs

In the EAM/FS file format, we have:
- lines 1, 2, 3: the comments
- line 4: Nelements, element1, element2 ...
- line 5: $N_{\rho}$, $\mathrm{d}_{\rho}$, $N_r$, $\mathrm{d}_r$, $r_{\mathrm{cutoff}}$

Following the header, we have Nelement sections. Each section contain, for element $\beta$:
- line 1: atomic number, mass, lattice constant, type
- $N_{\rho}$ lines of the embedding function $F_{\beta}$
- density function $f_{i \beta}$ for all Nele $i$

Finally, we have the pairpotential. The value of the pairpotential is given by $r\cdot \phi$ for a total $N_{\mathrm{ele}}^2$ blocks. Since pairpotential are symmetric, only $\phi_{ij}$ with $i\le j$ are listed. The blocks are ordered as first by $i$ and then by $j$: $(0,0), (1,0), (1,1), (2,0), (2,1), (2,2), \cdots$.
