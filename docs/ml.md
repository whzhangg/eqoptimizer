# Method

## Model outline

In this work, we aim to develop a coarse-grained ML thermodynamic model that are easily compatible with Calphad methodology and at the same time descriptive enough to include various phases into a single model. Such a model aims to overcome the main limitation of the CALPHAD models in the sense that each phase with different structure has its own Calphad description and parameters cannot be naturally interpolated between different phases. On the other hand, we aim to perserve some of the advantages of Calphad-type models: 1) coarse-grained nature of the model which ignores the details of the atomistic properties, which is convenient for thermodynamic calculation, and 2) consistency in terms of chemical systems: lower order systems (binary) are the limiting case of higher order system (ternary, for example). To include structures into the free energy description, we utilized message passing framework to take into account the local coordination environment and utilized structure fingerprints generated from structure to take into account coarse-grained structure features. Coarse-graining in terms of the structure is done by selecting an appropriate prototype structure to represent a phase. In general, the free energy model is written:
$$
G_{\mathrm{out}}(T) = G(\mathbf{y},\mathcal{S},T;\boldsymbol{\theta}) + G_{\mathrm{id}}(T)
$$
where the input to the is the temperature, structural prototype $\mathcal{S}$ and internal coordinates such as site occupancy, the same input that is usually used for thermodynamic models. We also explicit included ideal Boltzmann entropy.

The model may therefore be viewed as injecting explicit and transferable structure dependence into a CALPHAD-like chemical representation, replacing independent phase-specific parameterizations by a shared thermodynamic functional conditioned on the structural prototype.

## Structure prototypes

For each phase, we choose a structure prototype that encode essential structure features such as coordination number and bonding environments. The prototypes are available from AFLOW database or can be determined from the experimental crystal structure. Since the prototype will represent a coarse-grained structure without shared by a single phase across composition, it does not need providing exact atomic positions and lattice vectors but rather representative values. As described in the following, prototype structures are also scaled before being used to create structural features to remove the dependence of these features on the cell size. This should improve transferability in the structural features. The prototype structure is also used to create the graph representation of the system for coordination based message passing.

## Message passing

The message passing neural network based on graphs (GNN) is considered to incorporate coordination environment dependence of free energy and local chemical interactions which can be important for the stability of phases. This is important because neighboring interaction play an significant role in determine stability of phases. In constract to GNN interatomic potentials, the model here based on prototype ignores any dependence on bond length and bond angles, thus coordinate-free after graph construction and only coordination aware. For a site $i$ in the graph, it is associated with a feature vector $\mathbf{h}_i$ and a set of neighboring sites $\sigma_i$. The feature vector is a combination of chemical features and further structural features:
$$
\mathbf{h}_i = \mathbf{h}_i^{\mathrm{chem}} \oplus \mathbf{h}^{\mathrm{struct}}_i
$$
The additional structure feature is included to describe longer range structure feature and many-body contribution that can be missed by coordination only message passing and help to differentiate sites. (e.g., sites with the same coordination but different arrangement).

The neighboring sites are determined using voronoi method to consider natural first-neighbor coordination and avoid the introduction of a cutoff distance. Through message passing, information from further neighbors can also be included. The message passing at layer $l$ is given by:
$$
\mathbf{h}^{l+1}_i = \mathbf{h}_i^l + \mathcal{F}_{\boldsymbol{\theta}}^l(\mathbf{h}_i^l, \{\mathbf{h}_j^l;j\in\sigma_i\})
$$
where $\mathcal{F}_{\boldsymbol{\theta}}^l$ is the update function considering the center atom $i$ and all its neighborings. We consider CGCNN-type gated updates:
$$
\mathcal{F}_{\boldsymbol{\theta}}^l(\mathbf{h}_i, \{\mathbf{h}_k;k\in\sigma_i\}) =
\sum_{j} \sigma(\mathbf{W}_f^{l} \mathbf{z}_{ij}^{l} + \mathbf{b}_f^{l}) \odot g (\mathbf{W}_s^{l} \mathbf{z}_{ij}^{l}  + \mathbf{b}_s^{l})
$$
where $\mathbf{z}_{ij}^l = \mathbf{h}_i^l\oplus \mathbf{h}_j^l$ and $\odot$ is the element-wise multiplication. The sigmoid function produces an element-wise gate, while $g$ is the conventionally softplus in CGCNN which produces the candidate message.

After $R$ layer of message passing, we read-out node output by linear transformation and average them to obtain the intensive output (per atom) by:
$$
\mathbf{e} = \frac{1}{N}\sum_{i} \left(\mathbf{W}_o \mathbf{h}^{(R)}_i + \boldsymbol{\beta}_o\right)
$$
The linear form of the output is motivated so that in the limiting case without message-passing updates, a linear readout preserves the CALPHAD-like chemical representation; for a fixed prototype, the structural descriptor contributes only a composition-independent offset. By performing message passing, the output become structural dependent not only due to coordination but also due to the additional chemical description $\mathbf{h}^{\mathrm{struct}}_i$. Note that the output $\mathbf{e}$ can be multi-dimensional and non-linear due to message passing.

## Temperature dependence

To output temperature $T$ dependent free energy, it is important to ensure that the temperature dependence is in general low order, as often observed in thermodynamic modeling. In fact, given a reference phase, the relative free energy in Calphad-type modeling can often reduced to linear $a+bT$. To enforce the low order temperature dependence of free energy, we determine free energy using the following formula:
$$
G(\mathbf{y},\mathcal{S},T) = \sum_{n=0}^N e_n(\mathbf{y},\mathcal{S}) T_n
$$
where $\mathbf{e}$ are the output of the message passing model $\mathbf{e}$, and $T_n$ is a set of temperature basis functions. Typical set of basis are polynomials or physical inspired:
$$
\{1, T, T^2, T^3,\cdots\}\; \quad\text{or}\quad \{1, T, T\ln T, T^2, \cdots\}
$$

## Chemical features

The main characteristics of the Calphad model is the smooth interpolation of free energy with respect to the internal coordinate of the system. A common form is the Redlich polynoimal which can include increasing order of composition dependence. The smooth functional form with controllable order of interaction with respect to chemistry is benefitial to the coarse-grained thermodynamic modeling. Considering this, we design the chemical feature for each site as:
$$
\begin{align*}
\mathbf{h}_i^{\mathrm{chem}} = \sum_\alpha y_{\alpha}^{(i)} x_\alpha & + 
\sum_{\alpha,\beta>\alpha} y_{\alpha}^{(i)}y_{\beta}^{(i)} f_2 (\{ x_\alpha\oplus y_{\alpha}^{(i)},x_\beta\oplus y_{\beta}^{(i)}\}) \\
& + \sum_{\alpha,\beta>\alpha, \gamma>\beta} y_{\alpha}^{(i)}y_{\beta}^{(i)}y_{\gamma}^{(i)} f_3 (\{ x_\alpha\oplus y_{\alpha}^{(i)},x_\beta\oplus y_{\beta}^{(i)},x_\gamma \oplus y_{\gamma}^{(i)}\}) + \cdots
\end{align*}
$$
where $f_i$ are neural network functions of order $i$ that are permutational invariant with the input for interaction ($i\geq 2$). $x_\alpha\oplus y_{\alpha}^{(i)}$ is the concatenated input vector for element $\alpha$. These networks are formulated following the deepset model:
$$
f_i (\{\mathbf{v}\}_I) = \rho_i \left( \sum_{\alpha \in I} \phi_i(\mathbf{v}_\alpha )\right);\quad |I|=i
$$
where $\rho_2$ and $\phi_2$ are both neural networks, and the output of the network has the same dimenionality as $\mathbf{x}$.

The above construction is an generalization of the calphad-type polynomial with controllable order. This can be seem that for a binary system, $f_3$ term will not have any contribution to the free energy, thus establishing a chemical hierarchy for different number of components, enabling easy extension from binary models to ternary systems.

Note: an alternative is to separate all channels for 1-body, 2-body etc which can more strictly retain body order.

## Structure features

To incorporate geometric features that are beyond local coordination, we consider BOP features which are associated with a canonical hamiltonian. The benefits of using BOP features are that:

- they are low dimensional descriptors that are physically inspired through the density of states and considers high order interactions.
- they are permutational and rotational invariant
- they lead to natural structure scaling of the structure through the second moments

Pervious work has shown that the BOP features can be used to differentiate crystal structures. For our model, we create BOP features from a structure prototype and scale it appropriately so that the overall band width is the same. Then, BOPfox program is used compute the moments directly. For $s$, $p$ and $d$ orbital channel, we consider up to the tenth moment leading a 27-dimension local structure descriptors (The first momemnt for all channels are zero by definition):
$$
\mathbf{h}^{\mathrm{struct}}_i = \mathbf{h}_i^{\mathrm{bop,s}} \oplus \mathbf{h}_i^{\mathrm{bop,p}} \oplus \mathbf{h}_i^{\mathrm{bop,d}}
$$
Since the magnitude of higher order moments are much larger than lower order moments, normalization is used.

## Interaction of chemical and geometry information

In our model, chemical interaction happens first for each sites to provide site chemical features. Such interaction follow strict hierarchy. Then, neural network update by message passing combines these chemical feature with geometric features including structure fingerprints by BOP and coordination by message passing. Since higher-order chemical features vanish identically on lower-order composition subspaces before structural coupling, the model preserves subsystem consistency: e.g. ternary interaction features do not contribute to binary systems. Subsequent nonlinear message passing may mix the surviving interaction orders, so the final free energy is not required to remain additively decomposable by chemical interaction order.

## Liquid phase

The liquid phase cannot be modelled using CGCNN since it does not have an associated structural prototype. However, as the liquid phase can be considered in a single description for all component considered, a special composition-only model can be employed. For a given composition $\mathbf{x}$, we can determine the chemical feature using the same functional form as above to obtain a feature vector $\mathrm{h}^{\mathrm{liq}}$ for a chemical composition. Then, we obtain output directly:
$$
\mathbf{e}^{\mathrm{liq}} = \phi_{\mathrm{liq}}(\mathrm{h}^{\mathrm{liq}})
$$
where $\phi_{\mathrm{liq}}$ is a neural network. Then, to read out the Gibbs energy, we consider the same formula:
$$
G_{\mathrm{liq}}(\mathbf{x},T) = \sum_{n=0}^N e_n(\mathbf{x}) T_n
$$

## Training and validation

### Task 1

Train on CALPHAD data. We will consider a few system and train the model so that it can include all different phases

### Task 1

Training on DFT/MLIP data. This can be done with increasing number of systems from simple unary to binary. We can take training data from alexandria database or generate training data from MLIPs. The prototype can be relaxed structure. For unary, only structure feature is useful to differentiate different structures.

# Appendix

## Review of the CGCNN model

CGCNN is a simple message passing graph neural network for property predictions. In the original definition, each node is presented by a feature vector $\mathbf{v}_i$ and each edge between node $i$ and $j$ is represented by $\mathbf{u}_{ijk}$ where we can have multiple bonds between the same node atoms corresponding to periodicity, indicated by $k$. The convolution function is used to update atom feature:
$$
\mathbf{v}_i^{(t+1)} = \mathbf{v}_i^t + \sum_{jk} \sigma(\mathbf{z}_{ijk}^{(t)} \mathbf{W}_f^{t} + \mathbf{b}_f^{(t)}) \odot g (\mathbf{z}_{ijk}^{(t)} \mathbf{W}_s^{t} + \mathbf{b}_s^{(t)})
$$
where $\mathbf{z}_{ijk}^{(t)} = \mathbf{v}_i^{(t)} \oplus \mathbf{v}_j^{(t)} \oplus \mathbf{u}_{ij,k}$ is the concatenated pairwise feature. $\odot$ is the element-wise multiplication. The sigmoid function produces an element-wise gate, while $g$ is the conventionally softplus in CGCNN which produces the candidate message. Their element-wise product is the message passed along the edge. Finally, a normalized summation as the pooling function is used to obtain global features of the crystal. In general, all node features from all layers can be used:
$$
\mathbf{v}_c = \mathrm{Pool}(v_0^{(0)},\cdots,v_N^{(R)})
$$
and in the implementation, only the last layer features are pooled by an average:
$$
\mathbf{v}_c = \frac{1}{N} \sum_i \mathbf{v}_i^{(R)}
$$
This global feature is then passed to an NN with $N_t$ depth to prepare for the final feature vector. Finally, several output layers can be defined corresponding to required prediction task.

For initialization, the edge feature $\mathbf{u}_{ijk}$ is obtained by Gaussian radiual base function, and the node feature $\mathbf{v}_i$ is obtained from a linear combination of the atomic embedding vector $\mathbf{a}_i$. The crystal graph is constructed using a cutoff of $5\,\mathrm{\AA}$ using the neighbor list.

## BOP descriptor

Pairwise message passing framework without dependence of atomic distance and bond angles will have difficult differentiating structures that share the same crystal graphs. One example is the FCC and HCP structure which cannot be differentiated only based on first neighbor connectivity. Previous works has tried different approaches to overcome this limitations such as sorting neighbor atoms depending on their distances. To overcome such as limitations, here we consider argumenting the node description so that local structure information is encoded.

We consider BOP moment features. In BOP formulation, local electronic moments, defined for an orbital $i$ of order $n$, is given by:
$$
\mu_{i}^{(n)} = \int_{-\infty}^{\infty} n_i(\epsilon) \epsilon^n \mathrm{d}\epsilon 
$$
where $\epsilon$ is the energy and $n_i(\epsilon)$ is the density of state of orbital $i$. The moments are immediately connected to the crystal structure by the moment theory:
$$
\mu_{i}^{(n)} = \langle i | H^{n} | i \rangle 
= \sum_j \langle i | H | j \rangle \langle j | H | i \rangle 
+ \sum_{jk} \langle i | H | j \rangle \langle j | H | k \rangle \langle k | H | i \rangle + \cdots
$$
and therefore includes all self-returning pass in the crystal graph over all the orbitals. The hamiltonian is typically a tight-binding hamiltonian which allows the direct calculation of the contribution of each self-returning pass as product of the TB hamiltonian metric elements. 

**Determination of the Hamiltonian** The hamiltonian is formulated canonically and is as follows, we have the radial part of the hamiltonian given by two effective parameters:
$$
\beta(r) = a \left(\frac{r}{c}\right)^b = \frac{a}{c^b} r^b
$$
where we choose $b=-5$. $c$ and $a$ parameters are directly correlated, thus we choose $c = 1$. The $a$ parameter is set to have the following ratio
$$
a_{ss\sigma}:a_{pp\sigma}:a_{pp\pi}:a_{dd\sigma}:a_{dd\pi}:a_{dd\delta} \\
= -1 : 1.570 : -0.517 : -1.604 : 1.069 : -0.267
$$
These ratio are set as following:
- $s$, $p$ and $d$ interactions are set to zero so the three chanels can be set independently. 
- $dd\sigma:dd\pi:dd\delta = -6:4:-1$ according to Andersen et al.
- $pp\pi:pp\sigma = -0.76:2.31$ according to Pettifor.
- In a FCC reference structure scaled by the second moment of the $d$ orbital so that $\mu_d^{(2)}=1$, the relative ratio of the coefficients of $s$ interaction and $p$ interaction are determined so that their respective second moment also vanish: $\mu_s^{(2)}=1$ and $\mu_p^{(2)}=1$. This can be easily achieved by normalization.

Reference structure: we use an FCC reference structure with lattice parameter $a=3.95\,\mathrm{\AA}$ (nearest neighbor distance of $2.79\, \mathrm{\AA}$, which correspond roughly to twice the average covalent radius of the first row transition metal.

**Structure scaling** to obtain the local structure descripton of a structure type without explicit dependence on structure scale, we adopted structure scaling from Jenke which normalize the second moment. Amoung the BOP moments, the $\mu^{(2)}$ is the first moment that depend on neighboring structure (one-hop neighbors) and its strength is related to the interatomic distances. This gives an electronic structure-based metric to scale the input structure. We define global averaged (over $s$, $p$ and $d$ channel) second moments $\mu^{(2)}_{\mathrm{global}}$ is describe the overall average width of the density of states and it is related to moments of each site by averaging:
$$
\sum_i \mu_i^{(n)} = \int_{-\infty}^{\infty} \epsilon^n \left( \sum_i n_i(\epsilon)\right) \mathrm{d}\epsilon \\
\mu_{\mathrm{global}}^{(n)} = \int_{-\infty}^{\infty} \epsilon^n n(\epsilon) \mathrm{d}\epsilon
$$
For each structure, its size scaled so that $\mu^{(2)}_{\mathrm{global}} = 1$ is satisfied. This can be achieved by two calculations: 1) calculate $\mu^{(2)}_{\mathrm{global}}$ with the original input structure, and 2) determine the scale factor:
$$
s = \left[\mu^{(2)}_{\mathrm{global}}\right]^{1/10}
$$
The factor $1/10$ is due to the fact that the Hamiltonian elements depend on structure as $r^{-5}$ and the second moment therefore depend on structure as $r^{-10}$.

## Roost model (not used)

Roost (representation learning from stoichiometry) is a good performing composition only model that uses message passing for different chemical components. Given an initial chemical embedding, a linear operation transfers it into the initial internal representation of element $i$: $\mathbf{h}_i^{(0)}$ with dimension $d$, which contain $d-1$ elemental feature and an additional composition value of the component $w_i$.

The composition is represented as a fully connected directed graph whose nodes are the chemical species present in the composition. The Aviary implementation also includes self-edges. The main update to the nodes are given by the following steps:
$$
e_{ij}^{(t)} = f^{(t)}(\mathbf{h}_i^{(t)} \oplus \mathbf{h}_j^{(t)})
$$
where $\oplus$ is concatenation and $f^{(t)}$ is a neural network called gate. Then, attention coefficients for species $i$ from $j$ are computed:
$$
a_{ij}^{(t)} = \frac{w_j \exp(e_{ij}^{(t)})}{\sum_k w_k \exp(e_{ik}^{(t)})}
$$
where $\mathbf{w}$ is the composition. $a_{ij}^{(t)}$ is then used finally to update the node feature:
$$
\mathbf{h}_i^{(t+1)} = \mathbf{h}_i^{(t)} + \frac{1}{N_{\mathrm{head}}} \sum_{m,j} a_{ij}^{(t,m)} g^{(t,m)}(\mathbf{h}_i^{(t)} \oplus \mathbf{h}_j^{(t)})
$$
where multiple attention head is introduced and indexed by $m$, and $g^{(t,m)}$ is the message function which is a neural network. The authors called this method "weighted soft-attention".

After the final iteration, a set of final features are obtained. These features are then pooled again by similar weighted soft-attention into a single vector and the description of the composition, which follows the above structure but with:
$$
e_{i} = f(\mathbf{h}_i^{(R)}) \\
a_{i} = \frac{w_i \exp(e_{i})}{\sum_j w_j \exp(e_{j})} \\
\mathbf{h_c} = \frac{1}{N_{\mathrm{head}}}\sum_{m,i} a_{i}^{(m)} g^{(m)}(\mathbf{h}_i^{(R)})
$$

Finally, this overall representation is passed to a multilayer FF neural network to obtain the final predictions, similar to the implementation of the CGCNN.
