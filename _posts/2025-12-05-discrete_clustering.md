---
layout: post
title:  "Simplex Projection (Sparsemax) for Differentiable Discrete Cluster Assignments"
date:   2025-11-11 12:00:00 +0800
last_modified_at: 2025-11-11 12:00:00 +0800
categories: [Statistical Modeling, Julia]
---

<style>
.figure-container {
  margin: 20px auto;
  width: 90%;
  text-align: center; 
}

.figure-container img {
  display: block; 
  margin: 0 auto;
  max-width: 100%;
  height: auto;
  border-radius: 4px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.figure-container figcaption {
  text-align: center; 
  font-style: italic; 
  font-size: 0.9em;
  margin-top: 8px;
  color: #666;
}

.code-comparison-table {
  width: 100%;
  border-collapse: collapse;
  margin: 20px 0;
  font-size: 0.9em;
}

.code-comparison-table th, .code-comparison-table td {
  border: 1px solid #ddd;
  padding: 8px;
  text-align: left;
}

.code-comparison-table th {
  background-color: #f2f2f2;
}

pre {
  margin: 0; 
}
</style>

## Introduction

Sampling the posterior for a likelihood conditional on discrete cluster assignment is a notoriously difficult problem for probabilistic models. Such discrete samples create a gradient discontinuity that prevents the use of gradient-based Hamiltonian Monte Carlo (HMC) samplers. This is unfortunate given the far superior efficiency of gradient-based samplers for MCMC.

<div class="figure-container">
  <img width="500" src="/assets/images/sparsemax/demo.png" alt="Simulated 3 component 2D multivariate gaussian mixture model"/>
  <figcaption>Figure 1: Simulated 3-component 2-dimensional Gaussian Mixture Model.</figcaption>
</div>

Consider the distribution shown above, which represents a 3-component, 2-dimensional Gaussian Mixture Model (GMM). Turing.jl, a Julia package for Bayesian inference, is notable for its superlative ability to easily sample the full posterior of cluster assignment for each observation. However, the sampling time for even small datasets is enormously restrictive.

It is for this reason that most GMMs are implemented as **marginalized models**, as shown below:

```julia
@model function gmm_standard(x, K)
    D, N = size(x)

    μ ~ filldist(MvNormal(Zeros(D), I), K)
    w ~ Dirichlet(K, 1.0)

    x ~ MixtureModel([MvNormal(μ[:, k], I) for k in 1:K], w)
end
```

## The Smearing Problem

Standard marginalized models suffer from the fact that the Dirichlet distribution and/or any Softmax-based learned weights tend to be "dense," or non-sparse. This results in probability mass being smeared across clusters that should essentially have zero probability.

We can observe this behavior in the REPL execution below. Note how softmax assigns substantial non-zero probability to every element, whereas project_to_simplex (Sparsemax) is capable of allocating true zeros.

<table class="code-comparison-table">
<thead>
<tr>
<th>Function Call</th>
<th>Output Vector</th>
<th>Interpretation</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>softmax([1, 2, 3])</code></td>
<td><code>[0.090, 0.245, 0.665]</code></td>
<td><strong>Dense</strong>: All clusters have mass.</td>
</tr>
<tr>
<td><code>project_to_simplex([1, 2, 3])</code></td>
<td><code>[0.0, 0.0, 1.0]</code></td>
<td><strong>Sparse</strong>: Converges to a "hard" assignment (One-hot).</td>
</tr>
<tr>
<td><code>project_to_simplex([1, 2.9, 3])</code></td>
<td><code>[0.0, 0.45, 0.55]</code></td>
<td><strong>Mixture</strong>: Sparse, but handles ambiguity.</td>
</tr>
</tbody>
</table>

## The Sparsemax Solution

<p><a href="https://arxiv.org/pdf/1309.1541" style="text-decoration: underline;">Wang and Carreira-Perpiñán (2013)</a> provide an interesting method for projecting any vector of reals onto a probability simplex. This projection has the fascinating property of being able to project—<b>differentiably</b>—to a one-hot vector.</p>

If inputs fall within a specific radius of the $\max x$, the vector retains mixture characteristics; otherwise, it snaps to a one-hot encoding. This property gives the method something along the lines of a "super-power," allowing for both mixtures and discrete cluster assignments while maintaining the differentiability required for HMC.

**Note:** You can find my Julia implementation of the algorithm described in the original paper, including a custom ChainRules pullback for fast differentiation, **[here](https://github.com/dan-sprague/DirichletDiffusion/blob/main/src/sparsemax.jl)**.

We can implement this in a Turing model as follows:

```julia
@model function gmm_sparsemax(x, K, temperature)
    D, N = size(x)

    μ ~ filldist(MvNormal(Zeros(D), I), K)
    
    # Learnable scaling factor for the logits
    α ~ Exponential(0.5)
    
    logits ~ filldist(Gumbel(), K)

    w = project_to_simplex(logits ./ α)

    x ~ MixtureModel([MvNormal(μ[:, k], I) for k in 1:K], w)
end
```

## Results & Validation

This non-linear transformation can be applied to induce sparsity in the learned weight vector for the marginalized GMM. It is interesting to note the similarity to the stick-breaking process used in the marginal approximation to the "infinite" Gaussian Mixture Model. Therefore, I sought to determine if Sparsemax would produce similar results to stick-breaking and improve model fitting over a standard Dirichlet marginalized GMM.

To do this, I estimated a Standard GMM, a Stick-breaking GMM, and a Sparsemax GMM with $K = 5$ to examine how robust each model was to misspecification (fitting 5 clusters to data generated from 3). Cluster assignments were visualized by taking the weights—e.g., `w = sparsemax(logits / α)`—and visualizing them as a proportion of a unit stick.

### Posterior Weights Analysis

The results below show that the standard GMM fails to learn the correct weight structure, assigning significant weight to two clusters that do not actually exist.

<div class="figure-container">
<img width="100%" src="/assets/images/sparsemax/gmm_discrete_heatmap.png" alt="Ground truth vs posterior weights heatmap"/>
<figcaption>Figure 2: Heatmap comparison of learned weights. Note the noise in the Standard GMM vs. the sparsity in Stick-Breaking and Sparsemax.</figcaption>
</div>

### Cluster Separation

Both the stick-breaking and Sparsemax models induce significant sparsity into the model while remaining fully differentiable. Indeed, it appears that the overall results for stick-breaking and Sparsemax are nearly identical. Both have nearly identical distributions for the weights $w$, and both appear to discriminate cluster membership well, even for low-separation data.

<div class="figure-container">
<img width="100%" src="/assets/images/sparsemax/gmm_grid_comparison_final.png" alt="Grid comparison of GMM fit"/>
<figcaption>Figure 3: Points colored by most likely cluster assignment with opacity indicating confidence of assignment.</figcaption>
</div>

## Discussion & Conclusion

The simplex projection (Sparsemax) offers a compelling alternative to traditional Softmax or Dirichlet-based parameterizations. By allowing the model to hit "hard zeros" in the weight vector, we achieve a level of interpretability usually reserved for discrete sampling methods, without sacrificing the gradients necessary for efficient MCMC.

While the Stick-breaking process yields similar results in this experiment, Sparsemax offers distinct theoretical advantages. Stick-breaking imposes an ordering on the clusters (the "rich get richer" phenomenon), which can sometimes be undesirable depending on the prior knowledge of the data. Sparsemax, conversely, treats the logits symmetrically before projection.

For practitioners using Julia and Turing.jl, this implies that we can build models that are both sparse (interpretable) and fast (differentiable). Future work might explore how this projection behaves in higher-dimensional latent spaces, such as those found in Variational Autoencoders (VAEs), where "disentanglement" is often a primary goal.