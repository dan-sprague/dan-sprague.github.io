using Turing, Distributions, LinearAlgebra, Random, Statistics
using CairoMakie
using Colors
using StatsFuns: logsumexp
using FillArrays
using ADTypes
using Bijectors

# ==========================================
# 1. Global Settings & Data Generation
# ==========================================
Random.seed!(42)

const K_fit = 5   
const D = 5        
const N_samples = 200      
const σ_true = 0.5         
const temperatures = [1.0] 

println("--- Generating Data (σ = $σ_true) ---")

μ_true = [[2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]]
Σ_true = [1.0 0.0; 0.0 1.0] .* σ_true

n_per_cluster = div(N_samples, 3)
X_data = hcat([rand(MvNormal(m, Σ_true), n_per_cluster) for m in μ_true]...)
X_data = X_data[:, shuffle(1:size(X_data, 2))]

println("Data Shape: $(size(X_data))")

# ==========================================
# 2. Vectorized Helpers
# ==========================================


function stick_breaking_transform(v::AbstractVector)
    K = length(v) + 1
    w = Vector{eltype(v)}(undef, K)
    remaining = one(eltype(v))
    for k in 1:(K-1)
        w[k] = v[k] * remaining
        remaining *= (1 - v[k])
    end
    w[K] = remaining
    return w
end

function project_to_simplex(y::Vector{T}) where T <: Real

    μ = sort(y, rev = true)
    n = length(y)  

    ρ = 1
    current_sum = zero(T)
    sum_at_ρ = zero(T)

    for j in 1:n  
        current_sum += μ[j]

        if μ[j] + (1 / j) * (1 - current_sum) > 0
            ρ = j
            sum_at_ρ = current_sum
        end
    end

    λ = (1 / ρ) * (1 - sum_at_ρ)

    return max.(y .+ λ, zero(T))
end

# ==========================================
# 3. Models (Unified Format)
# ==========================================

@model function gmm_standard(x, K)
    D, N = size(x)

    μ ~ filldist(MvNormal(Zeros(D), I), K)
    w ~ Dirichlet(K, 1.0)

    x ~ MixtureModel([MvNormal(μ[:, k], I) for k in 1:K], w)
end

@model function gmm_stick_breaking(x, K)
    D, N = size(x)

    μ ~ filldist(MvNormal(Zeros(D), I), K)

    alpha = 1.0
    v ~ filldist(Beta(1, alpha), K - 1)
    w = stick_breaking_transform(v)

    x ~ MixtureModel([MvNormal(μ[:, k], I) for k in 1:K], w)
end

@model function gmm_sparsemax(x, K, temperature)
    D, N = size(x)

    μ ~ filldist(MvNormal(Zeros(D), I), K)

    α = 0.5

    logits ~ filldist(Gumbel(),K)

    w = project_to_simplex(logits ./ α)

    x ~ MixtureModel([MvNormal(μ[:, k], I) for k in 1:K], w)
end



# ==========================================
# 4. Execution
# ==========================================

results_dict = Dict()

function get_mean_weights(chain, sym)
    vec(mean(Array(group(chain, sym)), dims=1))
end

function get_sparsemax_attention_weights(chain, X, K)
    logits_samples = Array(group(chain, :logits))
    n_samples = size(logits_samples, 1)

    # For each MCMC sample, compute weights from logits
    weights_samples = zeros(n_samples, K)
    for s in 1:n_samples
        logits_s = logits_samples[s, :]
        weights_samples[s, :] = project_to_simplex(logits_s)
    end

    # Average across MCMC samples
    return vec(mean(weights_samples, dims=1))
end

function assign_cluster_colors(μ_post, μ_true, K)
    n_true = length(μ_true)
    cluster_color_idx = zeros(Int, K)

    remaining = collect(1:K)
    for true_idx in 1:n_true
        if isempty(remaining)
            break
        end
        distances = [norm(μ_post[:, k] - μ_true[true_idx]) for k in remaining]
        closest_idx = argmin(distances)
        fitted_k = remaining[closest_idx]
        cluster_color_idx[fitted_k] = true_idx
        deleteat!(remaining, closest_idx)
    end

    next_color = n_true + 1
    for k in remaining
        cluster_color_idx[k] = next_color
        next_color += 1
    end

    return cluster_color_idx
end

# Helper to compute cluster responsibilities (soft assignment probabilities)
function compute_responsibilities(X, μ_post, weights, K)
    N = size(X, 2)
    D = size(X, 1)
    Σ = Matrix(I, D, D)  # Unit covariance

    log_probs = zeros(K, N)
    for k in 1:K
        dist = MvNormal(μ_post[:, k], Σ)
        for i in 1:N
            log_probs[k, i] = log(weights[k] + 1e-10) + logpdf(dist, X[:, i])
        end
    end

    responsibilities = zeros(K, N)
    for i in 1:N
        log_sum = logsumexp(log_probs[:, i])
        for k in 1:K
            responsibilities[k, i] = exp(log_probs[k, i] - log_sum)
        end
    end

    return responsibilities
end

function plot_gmm_assignment!(ax, chain, weights, title_str, X, K, colors)
    μ_samples = Array(group(chain, :μ))
    D = size(X, 1)  
    μ_post = reshape(vec(mean(μ_samples, dims=1)), D, K)
    N = size(X, 2)

    cluster_color_idx = assign_cluster_colors(μ_post, μ_true, K)

    responsibilities = compute_responsibilities(X, μ_post, weights, K)

    for i in 1:N
        probs = responsibilities[:, i]
        k_best = argmax(probs)
        p_best = probs[k_best]

        color_idx = cluster_color_idx[k_best]

        scatter!(ax, [X[1, i]], [X[2, i]],
                color=colors[color_idx],
                markersize=8,
                alpha=p_best)
    end

    for k in 1:K
        color_idx = cluster_color_idx[k]
        scatter!(ax, [μ_post[1, k]], [μ_post[2, k]],
                color=colors[color_idx],
                marker=:xcross,
                markersize=20,
                strokewidth=3)
    end
end

println("\nFitting Standard (Dirichlet)...")
chain_std = sample(gmm_standard(X_data, K_fit), NUTS(500,.65), 500, progress=true)
w_std = get_mean_weights(chain_std, :w)
results_dict[:std] = w_std

println("\nFitting Stick-Breaking (Beta)...")
chain_sb = sample(gmm_stick_breaking(X_data, K_fit), NUTS(500,0.65), 500; adtype=AutoForwardDiff(), progress=true)
v_samples = Array(group(chain_sb, :v))
w_samples = [stick_breaking_transform(v_samples[i, :]) for i in 1:size(v_samples, 1)]
w_sb = mean(w_samples)
results_dict[:sb] = w_sb

results_dict[:sp] = []
for t in temperatures
    println("\nFitting Sparsemax (T=$t)...")
    chain_sp = sample(gmm_sparsemax(X_data, K_fit, t), NUTS(500,0.65), 500; adtype=AutoForwardDiff(), progress=true)
    w_out = get_sparsemax_attention_weights(chain_sp, X_data, K_fit)
    push!(results_dict[:sp], w_out)
end

println("\nGenerating Plot...")

# ==========================================
# 4. Visualization (Discrete Matrix Construction)
# ==========================================
println("\nGenerating Discrete Matrix Heatmap...")

model_labels = [
    "Ground Truth",
    "Standard (Dirichlet)",
    "Stick-Breaking",
    "Sparsemax", # T=1.0
]

weights_raw = [
    [1/3,1/3,1/3,0.0,0.0],
    results_dict[:std],
    results_dict[:sb],
    results_dict[:sp][1],
]

N_models = length(weights_raw)

resolution = 500 

discrete_data = zeros(Int, resolution, N_models)

for (model_idx, w_vec) in enumerate(weights_raw)
    w_sorted = sort(w_vec, rev=true)
    
    current_x = 1
    
    for (rank, weight) in enumerate(w_sorted)
        n_blocks = round(Int, weight * resolution)
        
        end_x = min(current_x + n_blocks - 1, resolution)
        
        if end_x >= current_x
            discrete_data[current_x:end_x, model_idx] .= rank
        end
        
        current_x = end_x + 1
    end
    
    if current_x <= resolution
        discrete_data[current_x:end, model_idx] .= length(w_sorted)
    end
end


rank_colors = [:cornflowerblue, :orange, :mediumseagreen, :purple, :firebrick]

f = Figure(size = (1000, 300), backgroundcolor = :white)

ax = Axis(f[1, 1],
    title = "Cluster Weight Dominance (Discrete Proportions)",
    yticks = (1:N_models, model_labels),
    yreversed = true, 
    aspect = 5.0, 
    xgridvisible = false, ygridvisible = false,
    xticksvisible = false, yticksvisible = false
)

hm = heatmap!(ax, 1:resolution, 1:N_models, discrete_data,
    colormap = rank_colors,
    colorrange = (1, K_fit) # Ensure integers 1..K map to the specific colors
)

leg_elements = [PolyElement(color = rank_colors[i], strokecolor = :transparent) for i in 1:K_fit]
leg_labels = ["Group $i" for i in 1:K_fit]
Legend(f[1, 2], leg_elements, leg_labels, "Cluster ID", framevisible=false)

save("gmm_discrete_heatmap.png", f)
println("Done. Plot saved to gmm_discrete_heatmap.png")
display(f)
# ==========================================
# 8. Two-Row Comparison (High vs Low Separation)
# ==========================================

function generate_data(means, n_samples, σ_val)
    n_per = div(n_samples, length(means))
    Σ = [1.0 0.0; 0.0 1.0] .* σ_val
    X = hcat([rand(MvNormal(m, Σ), n_per) for m in means]...)
    return X[:, shuffle(1:size(X, 2))],n_per
end

println("Generating Datasets...")

μ_far = [[2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]]
X_far = generate_data(μ_far, 500, 0.5)

μ_close = [[1.25, 1.25], [-1.25, -1.25], [1.25, -1.25]]
X_close = generate_data(μ_close, 500, 0.5)

function fit_models_for_data(X_in, label)
    println("\nFitting models for: $label")
    
    print("Standard... ")
    c_std = sample(gmm_standard(X_in, K_fit), NUTS(500,0.65), 500; progress=false)
    
    print("Stick-Breaking... ")
    c_sb = sample(gmm_stick_breaking(X_in, K_fit), NUTS(500,0.65), 500; adtype=AutoForwardDiff(), progress=false)
    
    print("Sparsemax... ")
    c_sp = sample(gmm_sparsemax(X_in, K_fit, 1.0), NUTS(500,0.65), 500; adtype=AutoForwardDiff(), progress=false)
    
    println("Done.")
    return c_std, c_sb, c_sp
end

chain_std_far, chain_sb_far, chain_sp_far = fit_models_for_data(X_far, "High Separation")

chain_std_close, chain_sb_close, chain_sp_close = fit_models_for_data(X_close, "Low Separation")

println("\nGenerating Grid Plot...")

f_grid = Figure(size = (1400, 900), backgroundcolor = :white)

function plot_row!(fig, row_idx, X, c_std, c_sb, c_sp, row_title)
    # 1. Standard
    ax1 = Axis(fig[row_idx, 1], xlabel="Dim 1", ylabel="Dim 2", title="Standard (Dirichlet)")
    w_std = get_mean_weights(c_std, :w)
    plot_gmm_assignment!(ax1, c_std, w_std, "", X, K_fit, rank_colors)

    # 2. Stick Breaking
    ax2 = Axis(fig[row_idx, 2], xlabel="Dim 1", ylabel="Dim 2", title="Stick-Breaking")
    v_samples = Array(group(c_sb, :v))
    w_samples_sb = [stick_breaking_transform(v_samples[i, :]) for i in 1:size(v_samples, 1)]
    w_sb = mean(w_samples_sb)
    plot_gmm_assignment!(ax2, c_sb, w_sb, "", X, K_fit, rank_colors)

    # 3. Sparsemax
    ax3 = Axis(fig[row_idx, 3], xlabel="Dim 1", ylabel="Dim 2", title="Sparsemax (T=1.0)")
    w_sp = get_sparsemax_attention_weights(c_sp, X, K_fit)
    plot_gmm_assignment!(ax3, c_sp, w_sp, "", X, K_fit, rank_colors)

    Label(fig[row_idx, 0], row_title, rotation = pi/2, font = :bold, fontsize=18)
end

plot_row!(f_grid, 1, X_far, chain_std_far, chain_sb_far, chain_sp_far, "High Sep (±2.0)")

plot_row!(f_grid, 2, X_close, chain_std_close, chain_sb_close, chain_sp_close, "Low Sep (±1.0)")

Legend(f_grid[1:2, 4], 
    [PolyElement(color = rank_colors[i], strokecolor = :transparent) for i in 1:K_fit],
    ["Rank $i" for i in 1:K_fit],
    "Cluster Rank",
    framevisible = false
)

colsize!(f_grid.layout,1,Aspect(1,1.0))
colsize!(f_grid.layout,2,Aspect(1,1.0))
colsize!(f_grid.layout,3,Aspect(1,1.0))
resize_to_layout!(f_grid)
save("gmm_grid_comparison_final.png", f_grid)
println("Done. Saved to gmm_grid_comparison.png")
display(f_grid)



fig = Figure(size=(600,300))
ax = Axis(fig[1,2],
    width = 150,
    height = 150,
    xlabel = "Dim 1",
    ylabel = "Dim 2")


for k in 1:3
    x1 = rand(MvNormal(μ_close[k],I),250)
    color_idx = rank_colors[k]

    for i in 1:250
        scatter!(ax, [x1[1, i]], [x1[2, i]],
            color=color_idx,
            markersize=8,
            alpha=1)
    end
end

ax2 = Axis(fig[1,1],
    width = 150,
    height = 150,
    xlabel = "Dim 1",
    ylabel = "Dim 2")


for k in 1:3
    x1 = rand(MvNormal(μ_close[k],I),250)
    color_idx = rank_colors[k]

    for i in 1:250
        scatter!(ax2, [x1[1, i]], [x1[2, i]],
            color=:grey,
            markersize=8,
            alpha=1)
    end
end
resize_to_layout!(fig)
fig
save("demo.png",fig)