## Magnetism

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

>It should be noted that most software treat the calculated $RT \ln(1 + \beta) g(\tau)$ as the magnetic free energy for the formula unit. For per mole atom, the free energy need to divide $N_{\mathrm{atom}}$ the total atom in the formula unit.