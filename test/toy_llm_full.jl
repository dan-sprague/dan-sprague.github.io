
using LinearAlgebra
using Random
using Distributions
using Debugger
using LogExpFunctions

abstract type EnergyBasedLM end

struct EBLM <: EnergyBasedLM
    V::Vector{Float64}                    # Vocabulary
    n::Int                         # Sequence length
    β::Float64                     # Temperature parameter
    π::Vector{Float64}              # Prior distribution over vocabulary
    J::Matrix{Float64}

    U::Function
    ∇U::Function
end




function EBLM(V,n,β,π,J)
    U(x) = -β*(x' * J * x)
    ∇U(x) = -β*(J + J')*x

    EBLM(V,n,β,π,J,U,∇U)
end


n = 5
β = 1.0             # Temperature parameter
V = [-1.0, 1.0]  # Extended vocabulary
J = rand(Gumbel(),n, n)       # Random interaction matrix       
π = [0.7, 0.3]  


model = EBLM(V,n, β, π, J)

model.U([1,-1,1,1,1])
model.∇U([1,-1,1,1,1])


"""
Calculate probabilities for all possible sequences
Returns:
- sequences: Matrix where each column is a sequence
- probs: Vector of corresponding probabilities
"""
function enumerate_sequences(model::EBLM)
    n_sequences = 2^model.n
    sequences = zeros(model.n, n_sequences)
    energy = zeros(n_sequences)
    
    # Generate all possible sequences
    for i in 0:(n_sequences-1)
        # Convert integer to binary sequence using -1,1
        sequence = [(i >> j) & 1 == 1 ? 1.0 : -1.0 for j in 0:(model.n-1)]
        sequences[:, i+1] = sequence
        energy[i+1] = model.U(sequence)
    end
    
    return sequences, energy

end

function proj(x)
    2 .* (x .> 0) .- 1
end



function MUCOLA(model::EBLM,x;α=10.0)
    x_prime = x - ((α/2) * model.∇U(x)) + (sqrt(α) *randn(model.n))

    rand() < exp(model.U(x) - model.U(x_prime)) ? 2 .* (x_prime .> 0) .- 1 : x
    
end

function NCG(model::EBLM,x;α=0.5)
        μₓ = x - (α * model.∇U(x) / 2)

        q = -(1/(2 * α) .* norm(model.V .- μₓ)^2)
        
        @show q 
        
    
end



NCG(model,xnew)

seqs,energy = enumerate_sequences(model)

x = seqs[:,rand(1:size(seqs,2))]
xnew = copy(x)
N = 100
samples = zeros(model.n,N)
energies = zeros(N)
samples[:,1] .= xnew
energies[1] = model.U(xnew)
Threads.@threads for i in 2:N
    samples[:,i] .= sample(model,samples[:,i-1])
    energies[i] = model.U(samples[:,i-1])
end

using Plots
histogram(energy,normed=true)
histogram!(energies,normed=true)