# Implemented CALPHAD models

## Compound energy formalism model

In CALPHAD, the free energy of the phase is commonly described by the compound energy formalism (CEF), in which all sites in the unit cell are partitioned into sublattices. We define the site fraction for sublattice $s$ of element $A$ as $y_A^{(s)} = n_A^{(s)}/(n_A^{(s)}+n_B^{(s)}+\cdots)$
where $n_A^{(s)}$ is the number of atoms of element $A$ per molar formula on sublattice $s$. The total number of atoms of component $A$ in a molar formula unit is obtained by counting the number of $A$ on each sublattice: 
$$
N_i = \sum_s N_s y_i^{(s)}$$
where $N_s$ is the number of sites in sublattice $s$ in a molar formula unit. Omitting pressure, the thermodynamic model is given by:
$$
G_M(\mathbf{y},T) = \sum_{I} P_{I}(\mathbf{y}) g_I + RT \sum_s N_s \sum_{i=A}^{N} y_i^{(s)}\ln y_i^{(s)} + G_M^{\mathrm{ex}}(\mathbf{y},T)
$$
where the first sum over $I$ is over possible component arrays specifying the occupancy of sites in the end-members. For example, $I=(AB\cdots)$ with $A$ occupying the first sublattice, etc., and $g_I$ can be interpreted as the free energy of the end-member configuration $g_{AB\cdots}$, which can be written as temperature polynomials. The value of the coefficient $P_I$ is $y_A^{(1)}y_B^{(1)}\cdots$.

### excess energy

The excess energy can consist of different terms. Contributions from pairwise and ternary excess terms on the same sublattice, as well as binary mixing on two sublattices, are given by Lukas et al., where $L^{(n)}=L(T)$ are temperature polynomials indexed by integer $n$:
$$
\begin{align*}
G^{\mathrm{ex,pair}}_{M,ab\cdots\underbrace{(ij)}_{(s)}\cdots c}
=& y_a^{(1)}y_b^{(2)}\cdots (y_i^{(s)}y_j^{(s)})\cdots y_c^{(N)} \left[
\sum_{n=0}^{v}L^{(n)} (y_i^{(s)}-y_j^{(s)})^n\right] \\
G^{\mathrm{ex,ternary}}_{M,ab\cdots\underbrace{(ijk)}_{(s)}\cdots c}
=& y_a^{(1)}y_b^{(2)}\cdots (y_i^{(s)}y_j^{(s)}y_k^{(s)})\cdots y_c^{(N)} \left[
v_i L^{(0)} + v_j L^{(1)} + v_k L^{(2)} \right]  \\
G^{\mathrm{ex,2pair}}_{M,ab\cdots\underbrace{(ij)}_{(s)}\cdots\underbrace{(mn)}_{(r)}\cdots c}
=& y_a^{(1)}y_b^{(2)}\cdots (y_i^{(s)}y_j^{(s)})\cdots(y_m^{(r)}y_n^{(r)})\cdots y_c^{(N)} \times \\ & \quad \left[
L^{(0)} + (y_m^{(r)}-y_n^{(r)})L^{(1)} + (y_i^{(s)}-y_j^{(s)})L^{(2)} 
\right]
\end{align*}
$$
where the index $ab\cdots(ij)_{(s)}\cdots c$ means that the specific term is related to mixing on the $(s)$-th sublattice with components $i$ and $j$, while all other sublattices are occupied by $a,b,\cdots, c$, respectively. The same notation can be extended to ternary mixing and mixing on different sublattices. In ternary mixing, the normalized term $v_i = y^{(s)}_i + (1-\sum_j y^{(s)}_j)/3$ is used. 

Regarding the implementation of compound energy formalism, a few details are considered to make the resulting energy identifical to pycalphad:

- In pycalphad, element order are sorted alphabetically within each sublattice, and this convention is also used.
- To make the energy competible to pycalphad definition, if only the $L^{(0)}$ term is specified for the ternary interaction, it will be set that $L^{(0)} = L^{(1)} = L^{(2)}$.
- For two sublattice mixing, convention of the $L^{(1)}$ and $L^{(2)}$ ($L^{(1)}$ for the second sublattice of mixing and $L^{(2)}$ for the first) follows implementation of pycalphad, which seems to further refers to result from ThermoCal software.

### Magnetism

For a phase that  that carries magnet moments, magnetism contribute to the Gibbs energy. In the **Inden-Hillert-Jarl** formula, the magnetic contribution is modeled by modeling the heat capacity $C_p$ before (Ferro) and after (Para) the magnetic transition such as Curie temperature $T_c$. Note that paramagnetic case is different from the non-magnetic case because of possible spin excitation. In the case of ferromagnetism with Curie temperature $T_c$, The Gibbs energy contribution due to magnetism can be modeled by:
$$
G^{\mathrm{mag}}_m = n RT \ln(1 + \beta) g(\tau)
$$
where $\beta$ is the magnetic moment, $n$ is the number of atoms per formula unit that has the average magnetic moment $\beta$ and $\tau = T/T_c$. The function $g(\tau)$ is typically pre-determined with the following form. For $\tau < 1$ below the transition temperature:
$$
g(\tau) = 1 - \frac{1}{A}\left[\frac{79}{140f}\tau^{-1} + \frac{474}{497}\left(\frac{1}{f}-1\right)\left(\frac{\tau^3}{6}+\frac{\tau^9}{135}+\frac{\tau^{15}}{600}\right)\right]
$$
where
$$
A = \frac{518}{1125} + \frac{11692}{15975}\left(\frac{1}{f}-1\right)
$$
and above the transition temperature:
$$
g(\tau) = - \frac{1}{A}\left[\frac{\tau^{-5}}{10}+\frac{\tau^{-15}}{315}+\frac{\tau^{-25}}{1500}\right]
$$
Typically, the parameter $f=0.4$ is used for the BCC structure and $0.28$ is used in other crystal structures. 

Typically, $T_c$ and $\beta$ varies with composition. It should be noted however that the $T_c$ and $\beta$ used in the TDB file should be modified before they are used in the equations. This is because the sign of $T_c$ and $\beta$ is used to as an indicator of ferromagnetic or anti-ferromagnetic state. For either $T_c$ or $\beta$, a negative value indicate anti-ferromagnetic state and in such case, a modifier $\alpha$ is defined which has negative value $-1$ for BCC phase and $-3$ for all other structures, so that:
$$
T_c = \begin{cases} 
T_c^{\mathrm{tdb}} & T_c^{\mathrm{tdb}} \geq 0 \\
T_c^{\mathrm{tdb}}/\alpha & T_c^{\mathrm{tdb}} < 0 \\
\end{cases}
\
;\quad
\beta = \begin{cases} 
\beta^{\mathrm{tdb}} & \beta^{\mathrm{tdb}} \geq 0 \\
\beta^{\mathrm{tdb}}/\alpha & \beta^{\mathrm{tdb}} < 0 \\
\end{cases}
$$

A note related to the implementation: most software treat the calculated $RT \ln(1 + \beta) g(\tau)$ as the magnetic free energy for the formula unit. For per mole atom, the free energy need to divide $N_{\mathrm{atom}}$ the total atom in the formula unit.

## Order disorder phase

Split-CEF is a way to consistently consider disorder phase and ordered phase using the same sublattice model. The main point is that a disordered phase is used for reference and the Gibbs energy of the ordered phase will be enforced so that when the site fraction correspond to disordered case, its gibbs energy will equal to the disordered phase. However, in such case the end-member energy is "ordering energy" and does not correspond to the formation energy (or Gibbs energy) of the end-members.

We first define a reference disordered model $G^{\mathrm{disord}}$, which is usually a disordered solid-solution phase but may also be a CEF model containing $N_d$ sublattices:
$$
(s_d,s_2,\ldots,s_{N_d}).
$$
The first sublattice, $s_d$, represents the disordered substitutional sites, while the remaining sublattices $(s_2,\ldots,s_{N_d})$ are treated as non-ordering sublattices, such as interstitial sites.

The corresponding ordered phase contains $q$ ordering sublattices and $N_d-1$ non-ordering sublattices:
$$
(s_{o1},\ldots,s_{oq},s_2,\ldots,s_{N_d}),
$$
where $q\geq 2$. Therefore, the total number of ordered-phase sublattices is
$$
N_o=q+N_d-1\geq N_d+1.
$$
The sublattice correspondence is defined as
$$
\begin{aligned}
(s_{o1},\ldots,s_{oq}) &\longrightarrow s_d,\\
s_k^{\mathrm{ord}} &\longrightarrow s_k^{\mathrm{disord}},
\qquad k=2,\ldots,N_d.
\end{aligned}
$$
The final $N_d-1$ sublattices of the ordered phase must correspond to those of the disordered phase, in the same order, with compatible (same) constituent sets. In practice, we first match the trailing sublattices $s_2,\ldots,s_{N_d}$ in the ordered model and treat the remaining sublattices as $s_{o1},\ldots,s_{oq}$.
All ordering sublattices must have the same active constituent set as $s_d$. If an ordering sublattice contains only a subset of these constituents, the missing constituents are added to its allowed constituent set before constructing the ordered CEF model. 

If the disordered phase contain non-ordering sublattices, We require also that the ratio of multiplicity between the combined ordering sublattices $\sum_{j=1}^{q}m_{oj}$ and any of the remaining sublattices $m_{ok}$ to be the same as the between $m_d$ and the remaining $m_{dk}$ in the disordered phase, i.e.,
$$
\frac{\sum_{j=1}^{q}m_{oj}}{m_{d}}
=
\frac{m_{ok}}{m_{dk}},
\qquad k=2,\ldots,N_d,
$$

In the split CEF model, we have the following formula for the Gibbs energy per molar atom:
$$
\begin{align*}
G_m^{\mathrm{split}}(\mathbf{y}_{o1},\cdots,\mathbf{y}_{oq},\mathbf{y}_2,\cdots,\mathbf{y}_n) 
= & c \cdot G_m^{\mathrm{disord}}(\mathbf{y}_d,\mathbf{y}_2,\cdots,\mathbf{y}_n) \\ + 
& G_m^{\mathrm{ord}}(\mathbf{y}_{o1},\cdots,\mathbf{y}_{oq},\mathbf{y}_2,\cdots,\mathbf{y}_n) \\ - &G_m^{\mathrm{ord}}(\mathbf{y}_{d},\cdots,\mathbf{y}_{d},\mathbf{y}_2,\cdots,\mathbf{y}_n)
\end{align*}
$$
where $\mathbf{y}_d$ is the site fraction of the disordered sublattices, and is used for all the disorder-order sublattices in the last term. For a component $i$:
$$
y_{d,i}
=
\frac{\sum_{j=1}^{q}m_{oj}y_{oj,i}}
{\sum_{j=1}^{q}m_{oj}}
$$
for each constituent $i$, including vacancies where applicable. The composition of the ordered phase is exactly as the composition of the disordered phase.