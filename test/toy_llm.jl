using LinearAlgebra
using Random
using Distributions
using Debugger
using LogExpFunctions
"""
Abstract type for energy-based language models
Allows different energy functions to be implemented
"""
abstract type EnergyBasedLM end

"""
Autoregressive energy-based LM with general energy function
Parameters:
- V: vocabulary (set of possible token values)
- n: sequence length
- B: temperature parameter
- energy_fn: function that computes E(x_k | x_{<k})
"""
struct AutoregressiveEBLM <: EnergyBasedLM
    V::Vector{Float64}                    # Vocabulary
    n::Int                         # Sequence length
    B::Float64                     # Temperature parameter
    π::Vector{Float64}              # Prior distribution over vocabulary
    J::Matrix{Float64}              # Interaction matrix
end


function (model::AutoregressiveEBLM)(x_k, prefix,k)
    -sum(@. model.J[k,:] * (x_k * prefix))
end

"""
Compute conditional probability distribution
p(x_k | x_{<k}) = exp(-B * E(x_k | x_{<k})) / Z_k
where Z_k = ∑_{v ∈ V} exp(-B * E(v | x_{<k}))
"""
function conditional_distribution(model::AutoregressiveEBLM, sequence::Vector,k::Int)

    mask = vcat(ones(Int64,k-1), zeros(Int64,model.n-k+1))
    prefix = sequence .* mask

    @show prefix 

    # Compute unnormalized probabilities for each vocabulary item
    energies = model.(model.V, Ref(prefix),Ref(k))
    
    softmax(B .* energies)
end

"""
Generate sequence by sampling token by token:
x_k ~ p(· | x_{<k})
"""
function generate(model::AutoregressiveEBLM)
    sequence = zeros(Int64,model.n)


    sequence[1] = model.V[rand(Categorical(model.π))]

    for k in 2:model.n
        # Get distribution over next token
        probs = conditional_distribution(model, sequence,k)
        
        # Sample and append token
        idx = rand(Categorical(probs))
        sequence[k] = model.V[idx]
    end
    
    return sequence
end

"""
Compute probability of a complete sequence:
p(x) = ∏_{k=1}^n p(x_k | x_{<k})
"""
function sequence_probability(model::AutoregressiveEBLM, sequence)
    log_prob = 0.0

    log_prob += log(model.π[argmax(V .== sequence[1])])
    
    for k in 2:lastindex(sequence)
        probs = conditional_distribution(model, sequence,k)
        token_idx = argmax(V .== sequence[k])
        log_prob += log(probs[token_idx])
    end
    
    return exp(log_prob)
end


"""
Calculate probabilities for all possible sequences
Returns:
- sequences: Matrix where each column is a sequence
- probs: Vector of corresponding probabilities
"""
function enumerate_sequences(model::AutoregressiveEBLM)
    n_sequences = 2^model.n
    sequences = zeros(model.n, n_sequences)
    probs = zeros(n_sequences)
    
    # Generate all possible sequences
    for i in 0:(n_sequences-1)
        # Convert integer to binary sequence using -1,1
        sequence = [(i >> j) & 1 == 1 ? 1.0 : -1.0 for j in 0:(model.n-1)]
        sequences[:, i+1] = sequence
        probs[i+1] = sequence_probability(model, sequence)
    end
    
    return sequences, probs

end





sum(probs)

# Pairwise interaction model
n = 5
B = 1.0             # Temperature parameter
V = [-1.0, 1.0]  # Extended vocabulary
J = randn(n, n)       # Random interaction matrix
J = tril(J,0)           # Make lower triangular for autoregressive property
π = [0.7, 0.3]        # Uniform prior
model = AutoregressiveEBLM(V,n, B, π, J)
seq = generate(model)
prob = sequence_probability(model, seq)


seqs,probs = enumerate_sequences(model)
using Plots
histogram(probs)



"""
Generate sequence by sampling token by token:
x_k ~ p(· | x_{<k})
"""
function mostlikely(model::AutoregressiveEBLM)
    sequence = zeros(Int64,model.n)


    sequence[1] = model.V[rand(Categorical(model.π))]

    for k in 2:model.n
        # Get distribution over next token
        probs = conditional_distribution(model, sequence,k)
        
        # Sample and append token
        idx = argmax(probs)
        sequence[k] = model.V[idx]
    end
    
    return sequence
end

s = mostlikely(model)

sequence_probability(model,s) == maximum(probs)

