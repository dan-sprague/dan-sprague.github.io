
using LinearAlgebra
using Random
using Distributions
using Debugger
using LogExpFunctions
using TensorOperations
using ForwardDiff
newaxis = [CartesianIndex(1)]
abstract type EnergyBasedLM end

struct ThinkingMachine <: EnergyBasedLM
    V::Matrix{Float64}                   # Vocabulary
    n::Int
    β::Float64                     # Temperature parameter
    π::Vector{Float64}              # Prior distribution over vocabulary
    J::Matrix{Float64}

    U::Function
    ∇U::Function
end


function ThinkingMachine(V,n,β,π,J)
    U(x) = -sum(J * (2 .* x'x .- 1))
    ∇U(x) = -4J * x'

    ThinkingMachine(V,n,β,π,J,U,∇U)
end

"""
Calculate probabilities for all possible sequences
Returns:
- sequences: Matrix where each column is a sequence
- probs: Vector of corresponding probabilities
"""
function enumerate_sequences(model::ThinkingMachine)
    n_sequences = 2^model.n
    sequences = zeros(model.n, n_sequences)
    energy = zeros(n_sequences)
    
    # Generate all possible sequences
    for i in 0:(n_sequences-1)
        # Convert integer to binary sequence using -1,1
        sequence = [(i >> j) & 1 == 1 ? 1.0 : -1.0 for j in 0:(model.n-1)]
        sequences[:, i+1] = sequence
        sequence = sequence' .== model.V
        energy[i+1] = model.U(sequence)
    end
    
    return sequences, energy

end


function proj(x)
    2 .* (x .> 0) .- 1
end

n = 5
Σ = 2 
β = 1.0             # Temperature parameter
V = [[0 1];[1 0]]  # Extended vocabulary
J = rand(Normal(),n, n)       # Random interaction matrix       
π = [0.7, 0.3]  


model = ThinkingMachine(V,n, β, π, J);


seq = [(1 >> j) & 1 == 1 ? 1.0 : -1.0 for j in 0:(model.n-1)]
seq = seq' .== [1,-1]



μ(x;α=0.5) = x - (α * model.∇U(x)/2)

model.V * μ(seq)

μ(seq)
test = μ(seq)
model.V .- μ(seq)


grad = model.∇U(seq)



test2 = repeat(model.V,1,1,5)

test2 .- reshape(test,2,1,5) 

