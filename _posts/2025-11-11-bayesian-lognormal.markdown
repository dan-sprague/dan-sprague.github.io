---
layout: post
title:  "Bayesian Uncertainty Estimation for Small Sample LogNormal Data"
date:   2025-11-11 12:00:00 +0800
last_modified_at: 2025-11-11 12:00:00 +0800
categories: [Statistical Modeling, Julia]
---

<style>
/* Add responsive design styles */
img {
  max-width: 100%;
  height: auto;
}
.figure-container {
  margin-bottom: 20px;
  width: 100%;
}
</style>

<br>

**Key findings:**

- **Appropriate Transformation**: LogNormal distributed data must have uncertainty quantified in the log transformed space.
- **Uncertainty Underestimation**: Naive frequentist maximum likelihood 95% CI underestimate true uncertainty of right-skewed data.
- **Bayesian Posterior Estimation**: A statistical method is proposed that better estimates the underlying distirbution and uncertainty for lognormal data.
- **Consistent improvement**: The Bayesian approach provides superior distribution approximation across diverse parameter regimes
<br><br>
<hr>

## Overview

<p>Biological data often spans multiple orders of magnitude and is right-skewed away from the mean. Standard visualization practice transforms the data to log-space while adjusting axis labels to display the original scale (Figure 1). While this transformation aids visual interpretation, it introduces a subtle but important consideration for uncertainty estimation.</p>


<div class="figure-container">
  <img src="/assets/images/plot.svg" alt="Prior distribution comparisons"/>
  <figcaption style="text-align: center; font-style: plain; font-size: 0.9em;">Figure 1: Simulated data often seen in biological contexts.</figcaption>
</div>

<p>Statistical inference for lognormal data must be conducted on the log-transformed scale where the parameters estimating the average then follow a normal distribution. Consider the two groups in Figure 1 with medians at 10 and 7000 in the original scale. Correct inference requires evaluating the difference $\log(7000) - \log(10) \approx 6.55 - 2.30 = 4.25$ on the log scale where the data are normally distributed and can be compared against a null standard normal. Moreover, naively applying standard statistical methods (like frequentist confidence intervals) directly to lognormal data can lead to inferential errors, as we demonstrate in the following section.</p>
<br>

A bayesian method is proposed that better estimates the uncertainty from log-normal data.


<hr>

## Standard CI Underestimate Uncertainty

Inferring 95% confidence intervals prior to log-transforming the data leads to incorrect 95% CI (Figure 2, left). This is shown that as the data becomes more skewed with $\sigma$, the odds that the 95% CI contains the true parameter value goes down. Even when using the correct approach to calculate SEM in the log scale (Figure 2, center), small sample confidence intervals fail to capture upper tail behavior (Figure 2, right). 
<br><br>
To demonstrate these issues, 10,000 datasets were simulated for various sample sizes (n = 3, 5, 10, 20, 50) and scale parameters (σ = 0.25, 0.5, 1.0, 2.0). For each dataset, we computed 95% confidence intervals using the standard $\bar{x} \pm t_{\alpha/2, n-1} \cdot \text{SEM}$ formula on both scales.

<div class="figure-container">
  <img src="/assets/images/coverage_combined.svg" alt="Coverage probability comparison"/>
  <figcaption style="text-align: left; font-style: plain; font-size: 0.9em;">Figure 2: Three problems with frequentist confidence intervals for lognormal data. Left:  Estimated 95% CIs fail to contain the true mean 95% of the time when CI is computed on original scale. Middle: Correct nominal coverage when CI is computed on log scale. Right: Even with correct log-scale CIs, the upper bound systematically falls below the true 95th percentile.</figcaption>
</div>

For lognormal biological data where upper tail behavior matters (e.g., maximum drug concentrations, peak immune responses), care must be taken when plotting and analyzing such data.
<hr>

## Model Specification

Bayesian models allow us to place common sense priors against the data that prevent model overfitting, particularly in the case of small data. In the case of the data shown in Figure 1, two important priors admit themselves.

1. The group means are, a priori, equally likely between $y_{min}$ and $y_{max}$ and follow a T-Distribution due to the small sample size. 
2. The scale of the log normal is, a priori, likely to be close to standard.

**Mean Parameters**: The group mean parameters are modeled as shifted T-Distributions, with locations $\mu_j$ given uniform priors $\nu$ over the observed log-data range, while the degress of freedom parameter $\tau$ is estimated from the data.

**Scale Parameters**: The group scale parameters $\sigma_j$ use a Gamma$(\alpha,\beta)$ prior parameterized such that the prior mode is equal to 1. This is derived with $\alpha = (1/\beta) + 1$, where $\beta \sim \text{Exponential}(1)$. This enforces a prior for standard scale while allowing the model to estimate uncertainty in the scale parameter. The gamma distribution was chosen because the exponential distribution either has a mode at 0 (Figure 3, blue) which is not our prior belief, or a mean that isn't 1 (red). 

<div class="figure-container">
  <img src="/assets/images/prior_comparison.svg" alt="Prior distribution comparisons"/>
  <figcaption style="text-align: center; font-style: plain; font-size: 0.9em;">Figure 3: Comparison of prior distributions for location and scale parameters across groups.</figcaption>
</div>

The unique parameterization of the Gamma distribution represents the ideal behavior for the scale prior. The posterior for each group is the estimated using the data. 

<hr>

### Probabilistic Model

The model is specified as follows:

<table style="width: 100%; border-collapse: collapse;">
<tr style="border-bottom: 1px solid #ddd;">
  <td style="padding: 10px; vertical-align: top; width: 30%;"><strong>T-Distributed Means</strong></td>
  <td style="padding: 10px; vertical-align: top;">
<div style="text-align: left; display: inline-block;">
$$
\begin{aligned}
\tau &\sim \text{LocationScale}(1, 1, \text{Exponential}(1/29)) \\
\nu &\sim \text{Uniform}(\min(\log y), \max(\log y)) \quad \text{for } j = 1,\ldots,6 \\
\mu_j &\sim \text{LocationScale}(\nu, 1, \text{TDist}(\tau)) \quad \text{for } j = 1,\ldots,6 \\
\end{aligned}
$$
</div>
  </td>
</tr>
<tr style="border-bottom: 1px solid #ddd;">
  <td style="padding: 10px; vertical-align: top;"><strong>Standard scale prior</strong></td>
  <td style="padding: 10px; vertical-align: top;">
<div style="text-align: left; display: inline-block;">
$$
\begin{aligned}
\beta &\sim \text{Exponential}(1) \\
\alpha &= \frac{1}{\beta} + 1 \\
\sigma_j &\sim \text{Gamma}(\alpha, \beta) \quad \text{for } j = 1,\ldots,6 \\
\end{aligned}
$$
</div>
  </td>
</tr>
<tr>
  <td style="padding: 10px; vertical-align: top;"><strong>Likelihood</strong></td>
  <td style="padding: 10px; vertical-align: top;">
<div style="text-align: left; display: inline-block;">
$$
\begin{aligned}
y_i &\sim \text{LogNormal}(\mu_{c_i}, \sigma_{c_i})
\end{aligned}
$$
</div>
  </td>
</tr>
</table>

where $c_i$ denotes the class (group) assignment for observation $i$.

## Julia Implementation

The Turing.jl package enables easy specification and fast sampling of this relatively simple model.

```julia
using Turing
using Distributions

@model function lognormal_model(class, y)
    # Number of unique groups (6 total)
    n_groups = length(unique(class))

    # Data range for uniform prior on location
    y_min = minimum(log.(y))
    y_max = maximum(log.(y))

    # Degrees of freedom for t-distribution
    τ ~ LocationScale(1, 1, Exponential(1/29))

    # Priors on location parameters - uniform from min to max of log(y)
    ν ~ filldist(Uniform(y_min, y_max), n_groups)
    μ ~ arraydist(LocationScale.(ν, 1.0, TDist(τ)))

    # Mode = 1 prior for standard lognormal scale
    β ~ Exponential()
    α = (1 / β) + 1
    σ ~ filldist(Gamma(α,β), n_groups)

    y ~ product_distribution(LogNormal.(μ[class], σ[class]))
end

model = lognormal_model(class_data, y_data)
chain = sample(model, NUTS(), 5000;drop_warmup=true)
```

<br><br>
# Results

**Comparing 95%CI**
<div class="figure-container">
  <img src="/assets/images/comparison_plot.svg" alt="Model fit comparison"/>
  <figcaption style="text-align: center; font-style: plain; font-size: 0.9em;">Figure 4: Posterior predictive distributions compared to observed data for each group.</figcaption>
</div>

**Evaluating fitted distribution against simulated ground truth**

To quantify the improvement of the hierarchical Bayesian approach over standard maximum likelihood estimation across different levels of data skewness, we simulated 30 groups with n=3 samples for each of four scale parameters (σ = 0.25, 0.5, 1.0, 2.0). For each group, we calculated the Kullback-Leibler (KL) divergence between each method's estimated distributions and the true underlying distributions. Lower KL divergence indicates better approximation of the true distribution.

<div class="figure-container">
  <img src="/assets/images/kl_divergence_comparison.svg" alt="KL divergence comparison"/>
  <figcaption style="text-align: center; font-style: plain; font-size: 0.9em;">Figure 5: Comparison of average KL divergence between maximum likelihood (red) and Bayesian hierarchical (blue) models across different scale parameters. The Bayesian approach consistently outperforms ML across all levels of data skewness, with ML showing high variance while Bayesian maintains stable, low KL divergence.</figcaption>
</div>

<hr>

## Discussion

The hierarchical structure allows for partial pooling of information across groups while maintaining group-specific estimates. The uniform prior on location parameters bounded by the observed data range provides a weakly informative prior that regularizes extreme estimates without imposing strong assumptions.

The mode-at-one parameterization for the scale parameter represents a principled choice for lognormal data, as it centers the prior on a neutral scaling assumption while allowing the data to drive the posterior away from this default when warranted.

This substantial improvement in KL divergence demonstrates that the Bayesian hierarchical approach provides superior approximation of the true underlying distributions, especially in small-sample scenarios where maximum likelihood estimation is unstable.

<hr>

## Methods

Analysis was performed using [Turing.jl](https://turing.ml/) for probabilistic programming. Code and data are available at [link to repository].

### Inference Details

- Sampler: NUTS with automatic differentiation
- Chains: 4 independent chains
- Iterations: 2000 per chain (1000 warmup)
- Convergence diagnostics: $\hat{R} < 1.01$ for all parameters

<hr>
