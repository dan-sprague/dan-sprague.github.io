# Post 1 Plan: Every Probability Distribution Is an Energy Landscape

## Subtitle

> Before we can make Bayes flow through pipes, we need to turn probability into terrain.

## Reader promise

By the end, the reader will understand and implement

\[
\pi(\theta)=\frac{e^{-U(\theta)}}{Z}
\qquad\Longleftrightarrow\qquad
U(\theta)=-\log\pi(\theta)+C.
\]

The reader will also understand what this does and does not mean physically.

## Narrative outline

### 1. Open with a visual question

Show a Gaussian bell curve and ask:

> What terrain would make a thermally moving particle spend about 68% of its time within one standard deviation of the center?

Apply the negative logarithm:

\[
\pi(\theta)=\frac{1}{\sqrt{2\pi\sigma^2}}
e^{-(\theta-\mu)^2/(2\sigma^2)},
\]

so

\[
-\log\pi(\theta)
=\frac{(\theta-\mu)^2}{2\sigma^2}
+\frac12\log(2\pi\sigma^2).
\]

Ignoring the additive constant gives

\[
U(\theta)=\frac{(\theta-\mu)^2}{2\sigma^2},
\]

a quadratic potential with the mathematical form of a spring.

Do not yet derive Bayesian updating. Promise that observations will become additional springs later.

### 2. Begin with relative probability

For two states with unnormalized weights \(f(\theta_1)\) and \(f(\theta_2)\), define

\[
U(\theta)=-\log f(\theta).
\]

Then

\[
U(\theta_2)-U(\theta_1)
=-\log\frac{f(\theta_2)}{f(\theta_1)},
\]

or

\[
\frac{f(\theta_2)}{f(\theta_1)}
=e^{-[U(\theta_2)-U(\theta_1)]}.
\]

Core lessons:

- probability ratios become energy differences;
- multiplication becomes addition;
- only relative energies matter.

### 3. Normalize the landscape

Define

\[
Z=\int e^{-U(\theta)}d\theta,
\qquad
\pi(\theta)=\frac{e^{-U(\theta)}}{Z}.
\]

Explain that \(Z\) makes total probability one, is the partition function in statistical mechanics, and is often the hard integral in inference. Preview—but do not yet develop—the fact that it becomes Bayesian evidence.

### 4. Energy is a landscape, not a single number

Clarify that \(U\) assigns an energy to every possible state. Distinguish

\[
\arg\min_\theta U(\theta)
\]

from the full occupancy distribution

\[
\pi(\theta)=e^{-U(\theta)}/Z.
\]

The minimum identifies a preferred point; the distribution describes occupancy across the landscape.

### 5. Implement four examples

#### A. Discrete hypotheses

Use weights \([1,2,4,8]\), compute \(U_i=-\log f_i\), and verify that `softmax(-U)` recovers normalized weights. Draw the states as discrete energy wells.

#### B. Gaussian

Show that \(\mu\) moves the bowl and \(\sigma\) changes its curvature. A narrow Gaussian corresponds to a stiff well.

#### C. Beta

For

\[
\pi(\theta)\propto\theta^{\alpha-1}(1-\theta)^{\beta-1},
\qquad 0<\theta<1,
\]

derive

\[
U(\theta)
=-(\alpha-1)\log\theta
-(\beta-1)\log(1-\theta)+C.
\]

Use the boundary behavior to introduce constrained state spaces and infinite energy barriers.

#### D. Gaussian mixture

For a two-component mixture, compute \(U=-\log\pi\) numerically. Show the double-well landscape and preview why multimodal inference is hard: probability must cross an energy barrier.

### 6. Add temperature and physical units

Introduce

\[
\pi_T(\theta)=\frac{e^{-E(\theta)/(k_BT)}}{Z_T}
\]

and define dimensionless energy \(U=E/(k_BT)\). Animate high temperature, low temperature, and the \(T\to0\) optimization limit.

### 7. State minimal mathematical conditions

A proper equilibrium requires

\[
0<Z=\int e^{-U(\theta)}d\theta<\infty.
\]

Mention that:

- \(U=+\infty\) can represent forbidden states;
- differentiability is unnecessary merely to define equilibrium;
- differentiability will matter when forces are introduced;
- an equilibrium distribution must be confined or otherwise normalizable.

### 8. Include the coordinate caveat

A density is relative to a reference measure. Under \(\phi=g(\theta)\),

\[
p_\phi(\phi)=p_\theta(\theta)\left|\frac{d\theta}{d\phi}\right|.
\]

Therefore energy gains a Jacobian term. Keep this brief and promise to revisit it during information geometry.

### 9. Julia implementation

The complete script is `examples/energy_landscapes.jl`; reusable numerical functions belong in `src/BayesThroughPhysics.jl`.

Core utilities:

```julia
energy_from_logweight(logweight) = -logweight

function normalize_logweights(logweights)
    m = maximum(logweights)
    logZ = m + log(sum(exp.(logweights .- m)))
    return exp.(logweights .- logZ), logZ
end
```

Explain log-sum-exp and why direct exponentiation underflows.

Produce linked views of:

1. unnormalized weight \(f(\theta)\);
2. energy \(U(\theta)=-\log f(\theta)\);
3. normalized probability \(e^{-U}/Z\).

### 10. Preview the physical meaning of KL

Introduce the desired equilibrium \(\pi\) and an arbitrary current distribution \(\rho\). Ask how much physical work is stored in their mismatch. Preview without fully deriving:

\[
\mathcal F[\rho]-\mathcal F[\pi]
=k_BT\,\mathrm{KL}(\rho\Vert\pi).
\]

This establishes the KL/free-energy thread early without overloading the first post.

### 11. Closing transition

End with:

> We now know how to turn any positive probability model into an energy landscape. Bayesian inference contributes one additional fact: the model naturally arrives already factorized. The prior and likelihood multiply as probabilities, so they add as physical energies. In the Gaussian case, that addition can be built literally—with springs.

The next post is **Bayes' Theorem Adds Energies**.

## Required artifacts

- explanatory article;
- complete Julia script;
- static probability-to-energy figure;
- temperature animation;
- discrete energy-level diagram;
- multimodal double-well demonstration;
- tests that reconstructed distributions sum or integrate to one.

## Final unresolved question

The post should leave the reader asking:

> If a posterior can be represented as equilibrium in an energy landscape, what physical process carries an arbitrary initial distribution into that equilibrium?

That question drives the rest of the series.

