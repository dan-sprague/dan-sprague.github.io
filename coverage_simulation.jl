using Distributions
using Statistics
using Random
using CairoMakie

Random.seed!(42)

"""
Simulate coverage probabilities for 95% confidence intervals
on lognormal data with varying sample sizes and σ values.
Compares 95% CI on original scale vs log scale.
"""
function simulate_coverage(μ, σ, n, n_sims=10000)
    # True parameters
    true_log_mean = μ
    true_mean = exp(μ + σ^2/2)  # True mean on original scale

    # Storage for both approaches
    coverage_log = 0
    coverage_orig = 0
    lower_viol_log = 0
    upper_viol_log = 0
    lower_viol_orig = 0
    upper_viol_orig = 0

    for _ in 1:n_sims
        # Generate lognormal data
        log_data = rand(Normal(μ, σ), n)
        orig_data = exp.(log_data)

        # Approach 1: 95% CI on log scale (correct)
        m_log = mean(log_data)
        se_log = std(log_data) / sqrt(n)
        t_crit = quantile(TDist(n-1), 0.975)
        ci_lower_log = m_log - t_crit * se_log
        ci_upper_log = m_log + t_crit * se_log

        if ci_lower_log <= true_log_mean <= ci_upper_log
            coverage_log += 1
        elseif true_log_mean < ci_lower_log
            lower_viol_log += 1
        else
            upper_viol_log += 1
        end

        # Approach 2: 95% CI on original scale (incorrect for lognormal)
        m_orig = mean(orig_data)
        se_orig = std(orig_data) / sqrt(n)
        ci_lower_orig = m_orig - t_crit * se_orig
        ci_upper_orig = m_orig + t_crit * se_orig

        if ci_lower_orig <= true_mean <= ci_upper_orig
            coverage_orig += 1
        elseif true_mean < ci_lower_orig
            lower_viol_orig += 1
        else
            upper_viol_orig += 1
        end
    end

    return (
        log_scale = (coverage=coverage_log/n_sims, lower=lower_viol_log/n_sims, upper=upper_viol_log/n_sims),
        orig_scale = (coverage=coverage_orig/n_sims, lower=lower_viol_orig/n_sims, upper=upper_viol_orig/n_sims)
    )
end

# Simulation parameters
μ = 0.0  # True mean on log scale
sample_sizes = [3, 5, 10, 20, 50]
σ_values = [0.25, 0.5, 1.0, 2.0]

# Run simulations
results = Dict()
for σ in σ_values
    results[σ] = Dict()
    for n in sample_sizes
        println("Running σ=$σ, n=$n")
        results[σ][n] = simulate_coverage(μ, σ, n)
    end
end

# Wait to create figure until we have both analyses done

println("\nCoverage Results Summary:")
println("=" ^ 70)
for σ in σ_values
    println("\nσ = $σ:")
    for n in sample_sizes
        r = results[σ][n]
        println("  n=$n:")
        println("    Log scale:  Coverage=$(round(r.log_scale.coverage, digits=3)), Lower=$(round(r.log_scale.lower, digits=3)), Upper=$(round(r.log_scale.upper, digits=3))")
        println("    Orig scale: Coverage=$(round(r.orig_scale.coverage, digits=3)), Lower=$(round(r.orig_scale.lower, digits=3)), Upper=$(round(r.orig_scale.upper, digits=3))")
    end
end

# ============================================================================
# Part 2: Small Sample Problem - CI upper bound vs true 95th percentile
# ============================================================================

"""
Simulate how often the 95% CI upper bound is below the true 95th percentile
"""
function simulate_ci_vs_percentile(μ, σ, n, n_sims=10000)
    # True 95th percentile of the lognormal distribution
    true_dist = LogNormal(μ, σ)
    true_95_percentile = quantile(true_dist, 0.95)

    # Count how often CI upper bound is below true 95th percentile
    ci_below_count = 0

    for _ in 1:n_sims
        # Generate lognormal data
        log_data = rand(Normal(μ, σ), n)

        # Compute 95% CI on log scale
        m_log = mean(log_data)
        se_log = std(log_data) / sqrt(n)
        t_crit = quantile(TDist(n-1), 0.975)
        ci_upper_log = m_log + t_crit * se_log

        # Transform back to original scale
        ci_upper_orig = exp(ci_upper_log)

        # Check if CI upper bound is below true 95th percentile
        if ci_upper_orig < true_95_percentile
            ci_below_count += 1
        end
    end

    proportion_below = ci_below_count / n_sims

    return proportion_below, true_95_percentile
end

# Run small sample analysis
println("\n" ^ 2)
println("=" ^ 70)
println("Small Sample Problem: CI Upper Bound vs True 95th Percentile")
println("=" ^ 70)

small_sample_results = Dict()
for σ in σ_values
    small_sample_results[σ] = Dict()
    for n in sample_sizes
        prop_below, true_p95 = simulate_ci_vs_percentile(μ, σ, n)
        small_sample_results[σ][n] = (proportion_below=prop_below, true_p95=true_p95)
        println("σ=$σ, n=$n: $(round(prop_below*100, digits=1))% of CI upper bounds < true 95th percentile")
    end
end

# ============================================================================
# Create combined figure with all three panels
# ============================================================================

fig = Figure(size=(900, 300))

# Plot colors and markers
colors = [:blue, :green, :orange, :red]
markers = [:circle, :rect, :diamond, :utriangle]

# Panel 1: Original scale coverage (LEFT)
ax1 = Axis(fig[1, 1],
    xlabel="Sample Size (n)",
    ylabel="Coverage Probability",
    title="Frequentist 95% CI\nOriginal Scale",
    xticks=sample_sizes
)

# Panel 2: Log scale coverage (MIDDLE)
ax2 = Axis(fig[1, 2],
    xlabel="Sample Size (n)",
    ylabel="Coverage Probability",
    title="Frequentist 95% CI\nLog Scale",
    xticks=sample_sizes
)

# Panel 3: Small sample problem (RIGHT)
ax3 = Axis(fig[1, 3],
    xlabel="Sample Size (n)",
    ylabel="Proportion Below\nTrue 95th %ile",
    title="CI Upper Bound vs\nTrue 95th Percentile",
    xticks=sample_sizes
)

# Add nominal coverage lines
hlines!(ax1, [0.95], color=:black, linestyle=:dash, linewidth=1.5)
hlines!(ax2, [0.95], color=:black, linestyle=:dash, linewidth=1.5)

# Plot all three panels
for (i, σ) in enumerate(σ_values)
    # Panel 1: Original scale coverage
    coverages_orig = [results[σ][n].orig_scale.coverage for n in sample_sizes]
    lines!(ax1, sample_sizes, coverages_orig,
           color=colors[i], linewidth=2, label="σ = $σ")
    scatter!(ax1, sample_sizes, coverages_orig,
             color=colors[i], marker=markers[i], markersize=10)

    # Panel 2: Log scale coverage
    coverages_log = [results[σ][n].log_scale.coverage for n in sample_sizes]
    lines!(ax2, sample_sizes, coverages_log,
           color=colors[i], linewidth=2, label="σ = $σ")
    scatter!(ax2, sample_sizes, coverages_log,
             color=colors[i], marker=markers[i], markersize=10)

    # Panel 3: Small sample problem
    proportions = [small_sample_results[σ][n].proportion_below for n in sample_sizes]
    lines!(ax3, sample_sizes, proportions,
           color=colors[i], linewidth=2, label="σ = $σ")
    scatter!(ax3, sample_sizes, proportions,
             color=colors[i], marker=markers[i], markersize=10)
end

# Add legends
axislegend(ax1, position=:rb, labelsize=10)
axislegend(ax2, position=:rb, labelsize=10)
axislegend(ax3, position=:rt, labelsize=10)

# Set y-axis limits
ylims!(ax1, 0.0, 1.0)
ylims!(ax2, 0.0, 1.0)
ylims!(ax3, 0.0, 1.0)

save("/Users/dansprague/Documents/dan-sprague.github.io/assets/images/coverage_combined.svg", fig)

println("\nFigure saved: coverage_combined.svg")
