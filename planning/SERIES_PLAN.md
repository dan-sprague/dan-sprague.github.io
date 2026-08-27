# Bayes Through Physics: Series Plan

## Central premise

> A probability distribution specifies where probability should be. Physics and geometry offer different answers for how it can get there.

The project is built from scratch in Julia and follows one continuous argument:

\[
\text{probability}
\rightarrow
\text{energy}
\rightarrow
\text{equilibrium}
\rightarrow
\text{free energy}
\rightarrow
\text{dynamics}
\rightarrow
\text{geometry}
\rightarrow
\text{online information}
\rightarrow
\text{discrete flow}
\rightarrow
\text{GFlowNets}.
\]

The project does not claim that inference is secretly governed by one unique physical law. It investigates several physical and geometric dynamics that share a probabilistic target and asks what each reveals about learning.

## Part I — Probability as energy

### 1. Every Probability Distribution Is an Energy Landscape

For an unnormalized positive weight \(f(\theta)\), define

\[
U(\theta)=-\log f(\theta),
\qquad
\pi(\theta)=\frac{e^{-U(\theta)}}{Z}.
\]

Develop probability ratios as energy differences, additive energy constants, forbidden states, partition functions, reference measures, and changes of coordinates. Implement discrete, Gaussian, beta, and mixture examples.

### 2. Bayes' Theorem Adds Energies

Starting from

\[
p(\theta\mid D)=\frac{p(\theta)p(D\mid\theta)}{p(D)},
\]

derive

\[
U_{\mathrm{posterior}}
=U_{\mathrm{prior}}+U_{\mathrm{likelihood}},
\qquad Z=p(D).
\]

The central correspondence is that probabilities multiply while energies add.

### 3. Why Thermal Systems Produce Distributions

Derive the Gibbs distribution

\[
\rho(\theta)=\frac{e^{-E(\theta)/(k_BT)}}{Z}
\]

by maximizing entropy under an expected-energy constraint. Distinguish physical energy \(E\) from dimensionless energy \(U=E/(k_BT)\).

### 4. Bayesian Inference with Springs

For a Gaussian prior and Gaussian observations, show that the prior is a spring and every observation adds another spring. Interpret posterior mean as the balance point, posterior precision as total stiffness, and posterior uncertainty as thermal motion.

### 5. MLE, MAP, and Bayes Are Different Physical Questions

Contrast minimizing likelihood energy, minimizing posterior energy, and occupying the full landscape thermally. Show optimization emerging as a zero-temperature limit.

## Part II — Free energy and the physical meaning of KL

### 6. KL Divergence Is Excess Free Energy

For current distribution \(\rho\) and equilibrium distribution \(\pi\), derive

\[
\boxed{
\mathcal F[\rho]-\mathcal F[\pi]
=k_BT\,\mathrm{KL}(\rho\Vert\pi).
}
\]

Thus the same forward KL is statistical target discrepancy, physical excess free energy divided by \(k_BT\), and computational reorganization still required. State the assumptions: a shared energy landscape, heat bath, temperature, and reference measure.

### 7. Bayes as a Free-Energy Minimum

Define

\[
\mathcal G[\rho]=\mathbb E_\rho[U]-H[\rho]
\]

and derive

\[
\mathcal G[\rho]
=\mathrm{KL}(\rho\Vert\pi)-\log Z.
\]

MAP minimizes energy over points; Bayes minimizes free energy over distributions.

## Part III — Dynamics toward the posterior

### 8. Why the Heat Equation Does Not Perform Bayes

Implement \(\partial_t\rho=D\Delta\rho\). Show that pure diffusion smooths probability but contains no target information and ordinarily approaches a uniform equilibrium on a bounded reflecting domain.

### 9. Langevin Dynamics: Heat Plus a Force

Introduce

\[
d\theta_t=-\nabla U(\theta_t)dt+\sqrt{2T}\,dW_t
\]

and derive its Fokker–Planck equation. Compare particle and density simulations and verify the Gibbs equilibrium.

### 10. Free-Energy Dissipation

For a fixed target, derive under consistent units

\[
\frac{d}{dt}\mathrm{KL}(\rho_t\Vert\pi)
=-I(\rho_t\Vert\pi).
\]

Interpret relative Fisher information as the instantaneous rate of free-energy dissipation.

### 11. Closed-Form Posterior, Unsolved Dynamics

Compare Gaussian/Ornstein–Uhlenbeck, gamma/CIR, beta/Wright–Fisher, and a known posterior with a difficult transient. Emphasize that a closed-form equilibrium does not imply a closed-form route to equilibrium.

## Part IV — Least action and probability geometry

### 12. Probability as a Conserved Fluid

Introduce the continuity equation

\[
\partial_t\rho+\nabla\cdot(\rho v)=0
\]

and implement conservative transport before optimizing the velocity.

### 13. The Least-Action Route from Prior to Posterior

Among all flows joining the prior to posterior, minimize

\[
\mathcal A[\rho,v]
=\int_0^1\int\frac12\rho\lVert v\rVert^2\,d\theta\,dt
\]

subject to probability conservation. Introduce the Benamou–Brenier formulation, Wasserstein distance, and geodesics in distribution space.

### 14. Hamilton–Jacobi and Hamiltonian Probability Flow

Derive potential flow \(v=\nabla\phi\) and

\[
\partial_t\phi+\frac12\lVert\nabla\phi\rVert^2=0.
\]

Connect density and velocity potential as conjugate variables to classical mechanics, HMC, time-dependent Hamiltonians, and—carefully—the Madelung representation of quantum mechanics.

### 15. Two Geometries of Probability

Contrast Wasserstein transport,

\[
\partial_t\rho+\nabla\cdot(\rho v)=0,
\]

with Fisher–Rao reweighting,

\[
\partial_t\rho=\rho(g-\mathbb E_\rho[g]).
\]

Show that KL is globally an excess free energy and locally generates the Fisher metric:

\[
\mathrm{KL}(p_\theta\Vert p_{\theta+d\theta})
=\frac12d\theta^T I(\theta)d\theta+O(\lVert d\theta\rVert^3).
\]

## Part V — Online Bayesian learning

### 16. Discrete Information Arrives as Energy Pulses

For observations \(d_1,d_2,\ldots\), write

\[
U_n(\theta)
=U_{\mathrm{prior}}(\theta)
-\sum_{i=1}^n\log p(d_i\mid\theta).
\]

Animate data as pulses deforming the landscape and as new springs in the Gaussian system.

### 17. Continuous Bayes and Fisher–Rao Flow

Let \(\ell_t(\theta)\) be a log-likelihood rate and define

\[
\pi_t(\theta)=
\frac{\pi_0(\theta)\exp(\int_0^t\ell_s(\theta)ds)}{Z_t}.
\]

Differentiate to obtain the exact continuous update

\[
\partial_t\pi_t
=\pi_t\left(\ell_t-\mathbb E_{\pi_t}[\ell_t]\right).
\]

Connect this to replicator dynamics and information geometry.

### 18. Can the Inference System Keep Up?

Separate the exact moving posterior \(\pi_t\) from the actual computational distribution \(\rho_t\), and define

\[
K_t=\mathrm{KL}(\rho_t\Vert\pi_t).
\]

For instantaneous Langevin relaxation, derive

\[
\dot K_t
=-I(\rho_t\Vert\pi_t)
+\mathbb E_{\pi_t}[\ell_t]
-\mathbb E_{\rho_t}[\ell_t].
\]

Interpret this as

\[
\text{change in excess free energy}
=\text{information-driven work}-\text{dissipation}.
\]

### 19. The Posterior as an Inertial Fluid

Construct a compressible, isothermal probability fluid driven by the time-dependent Bayesian potential. Show that hydrostatic equilibrium is the posterior and the strong-friction limit recovers Fokker–Planck dynamics. Explore overshoot, waves, viscosity, and inference lag without claiming that Bayes uniquely implies Navier–Stokes.

## Part VI — Discrete probability flow

### 20. From PDEs to Master Equations on Graphs

Replace continuous states by graph nodes:

\[
\dot p_i=\sum_j(q_{ji}p_j-q_{ij}p_i).
\]

Develop transition rates, detailed balance, graph Laplacians, and the discrete continuity equation.

### 21. Deriving a GFlowNet Before Naming It

Construct a discrete object along a DAG and assign

\[
R(x)=p(x)p(D\mid x)=e^{-U(x)}.
\]

Require conservation at internal nodes and derive

\[
P_{\mathrm{terminal}}(x)
=\frac{R(x)}{\sum_yR(y)}
=p(x\mid D),
\qquad
F(s_0)=Z=p(D).
\]

Only after deriving the result introduce the name Generative Flow Network.

### 22. Building a GFlowNet from Scratch in Julia

Implement a finite DAG, tabular flows, forward and backward policies, flow matching, detailed balance, trajectory balance, exact enumeration, and sampling diagnostics. Plot both training loss and actual terminal KL because they are not generally identical.

### 23. A Hydraulic GFlowNet

Visualize states as junctions, edges as pipes, sampled trajectories as droplets, terminal states as reservoirs, conservation residuals as local accumulation, and root flow as the partition function. Explore the underdetermination of internal flow.

### 24. Online Bayesian GFlowNets

Let

\[
\frac{d}{dt}\log R_t(x)=\ell_t(x),
\qquad
\pi_t(x)=\frac{R_t(x)}{Z_t}.
\]

Measure the terminal tracking lag \(\mathrm{KL}(q_t\Vert\pi_t)\) as information changes terminal energies and the internal flow reorganizes. Compare discrete updates, continuous updates, warm adaptation, and retraining.

### 25. Neural GFlowNets

Replace tabular flow by

\[
P_F(a\mid s)=\operatorname{softmax}(f_\phi(s)).
\]

The neural network is presented as a scalable parameterization of a flow system already understood exactly.

## Epilogue — Could We Build the Physical Machine?

Return to springs, Brownian particles, optical traps, stochastic circuits, resistor networks, heated graph structures, and literal fluid networks. Carefully distinguish physical visualization, analog equation solving, and actual thermodynamic posterior sampling.

## Recurring visual and mathematical language

Every relevant simulation should expose:

- \(U_t\): energy landscape;
- \(\pi_t\): exact target posterior;
- \(\rho_t\) or \(q_t\): current approximation;
- \(\mathrm{KL}(\rho_t\Vert\pi_t)\): statistical lag;
- \(k_BT\,\mathrm{KL}(\rho_t\Vert\pi_t)\): excess free energy;
- \(I(\rho_t\Vert\pi_t)\): dissipation rate;
- incoming information rate;
- flow-conservation residuals where applicable.

The recurring gauge is

\[
\boxed{
\text{KL lag}
=\text{inference error}
=\frac{\text{excess free energy}}{k_BT}.
}
\]

## Planned Julia artifacts

1. `energy_landscapes.jl`
2. `bayes_adds_energies.jl`
3. `bayesian_springs.jl`
4. `free_energy_and_kl.jl`
5. `heat_equation.jl`
6. `langevin_particles.jl`
7. `fokker_planck.jl`
8. `least_action_transport.jl`
9. `fisher_rao_bayes.jl`
10. `online_posterior.jl`
11. `probability_fluid.jl`
12. `graph_master_equation.jl`
13. `tabular_gflownet.jl`
14. `streaming_gflownet.jl`

