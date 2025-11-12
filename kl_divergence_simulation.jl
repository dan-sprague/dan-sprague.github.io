using Turing
using Distributions
using CairoMakie
using Statistics
using Random
using Printf

Random.seed!(42)

# Define the Bayesian model from the blog post
@model function lognormal_model(class, y)
    # Number of unique groups
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
    σ ~ filldist(Gamma(α, β), n_groups)

    y ~ product_distribution(LogNormal.(μ[class], σ[class]))
end

# Generate data for one group with specified sigma
function generate_group_data(n=3, true_σ=1.0)
    # Random true mean (on log scale)
    true_μ = randn() * 2

    # Generate data
    y = rand(LogNormal(true_μ, true_σ), n)

    return y, true_μ, true_σ
end

# Calculate KL divergence for normal distributions
function calculate_kl_divergence(true_μ, true_σ, est_μ, est_σ)
    # KL(P||Q) for normal distributions:
    # KL = log(σ_Q/σ_P) + (σ_P^2 + (μ_P - μ_Q)^2)/(2σ_Q^2) - 1/2

    σ_P = true_σ
    μ_P = true_μ
    σ_Q = est_σ
    μ_Q = est_μ

    kl = log(σ_Q / σ_P) + (σ_P^2 + (μ_P - μ_Q)^2) / (2 * σ_Q^2) - 0.5

    return kl
end

println("Starting simulation with varying sigma values...")
println("=" ^ 70)

# Different sigma values to test
sigma_values = [0.25, 0.5, 1.0, 2.0]
n_groups_per_sigma = 30
n_per_group = 3

# Storage for results
results = Dict(
    "sigma" => Float64[],
    "method" => String[],
    "kl" => Float64[]
)

for σ in sigma_values
    println("\nProcessing σ = $σ...")

    for i in 1:n_groups_per_sigma
        println("  Group $i / $n_groups_per_sigma...")

        # Generate data for this group with specified sigma
        y, true_μ, true_σ = generate_group_data(n_per_group, σ)

        # Maximum Likelihood estimation
        log_y = log.(y)
        μ_ml = mean(log_y)
        σ_ml = std(log_y)

        # Bayesian estimation
        class = ones(Int, n_per_group)  # All same group
        model = lognormal_model(class, y)

        # Sample from posterior
        chain = sample(model, NUTS(), 1000, progress=false, discard_initial=500)

        μ_bayes = mean(chain["μ[1]"])
        σ_bayes = mean(chain["σ[1]"])

        # Calculate KL divergences
        kl_ml = calculate_kl_divergence(true_μ, true_σ, μ_ml, σ_ml)
        kl_bayes = calculate_kl_divergence(true_μ, true_σ, μ_bayes, σ_bayes)

        # Store results
        push!(results["sigma"], σ)
        push!(results["method"], "ML")
        push!(results["kl"], kl_ml)

        push!(results["sigma"], σ)
        push!(results["method"], "Bayesian")
        push!(results["kl"], kl_bayes)
    end
end

# Calculate statistics for each sigma and method
sigma_labels = String[]
ml_means = Float64[]
ml_ci_half_widths = Float64[]
bayes_means = Float64[]
bayes_ci_half_widths = Float64[]

for σ in sigma_values
    # Get ML results for this sigma
    ml_mask = (results["sigma"] .== σ) .& (results["method"] .== "ML")
    ml_kls = results["kl"][ml_mask]

    # Get Bayesian results for this sigma
    bayes_mask = (results["sigma"] .== σ) .& (results["method"] .== "Bayesian")
    bayes_kls = results["kl"][bayes_mask]

    push!(sigma_labels, "σ=$σ")
    push!(ml_means, mean(ml_kls))
    push!(ml_ci_half_widths, 1.96 * std(ml_kls) / sqrt(length(ml_kls)))
    push!(bayes_means, mean(bayes_kls))
    push!(bayes_ci_half_widths, 1.96 * std(bayes_kls) / sqrt(length(bayes_kls)))

    println("\nσ = $σ:")
    @printf("  ML - Mean: %.4f, 95%% CI: %.4f\n", mean(ml_kls), 1.96 * std(ml_kls) / sqrt(length(ml_kls)))
    @printf("  Bayesian - Mean: %.4f, 95%% CI: %.4f\n", mean(bayes_kls), 1.96 * std(bayes_kls) / sqrt(length(bayes_kls)))
end

# Create grouped bar chart
fig = Figure(size=(400, 300))

ax = Axis(fig[1, 1],
    xlabel="Scale Parameter",
    ylabel="KL Divergence",
    title="Similarity to Simulated\nGround Truth",
    xticks=(1:length(sigma_values), sigma_labels),
    titlesize=14,
    xlabelsize=12,
    ylabelsize=12
)

# Remove frame/border
hidespines!(ax, :t, :r)

# Plot grouped bars
bar_width = 0.35
x_positions = 1:length(sigma_values)

# ML bars (red, left)
barplot!(ax, x_positions .- bar_width/2, ml_means, color=:red, width=bar_width, label="ML")
errorbars!(ax, x_positions .- bar_width/2, ml_means, ml_ci_half_widths, color=:black, linewidth=2, whiskerwidth=8)

# Bayesian bars (blue, right)
barplot!(ax, x_positions .+ bar_width/2, bayes_means, color=:blue, width=bar_width, label="Bayesian")
errorbars!(ax, x_positions .+ bar_width/2, bayes_means, bayes_ci_half_widths, color=:black, linewidth=2, whiskerwidth=8)

# Add legend
axislegend(ax, position=:lt)

save("/Users/dansprague/Documents/dan-sprague.github.io/assets/images/kl_divergence_comparison.svg", fig)

println("\n\nFigure saved: kl_divergence_comparison.svg")
