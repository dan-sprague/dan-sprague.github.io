using Plots
using LinearAlgebra

# Define potential energy function (same as before - mixture of Gaussians with valleys)
function U(q)
    x, y = q
    g1 = 3 * exp(-((x-1.5)^2 + (y-1.5)^2) /4)  # Tall peak
    g2 = 2 * exp(-((x+1)^2 + (y+1)^2) / 3)      # Medium peak
    g3 = 4 * exp(-((x-6)^2 + (y-3)^2) / 3)                      # Wavy surface
    return -(g1 + g2 + g3)
end

# Gradient of potential energy
function ∇U(q)
    ε = 1e-3
    x, y = q
    ∂x = (U([x + ε, y]) - U([x - ε, y])) / (2ε)
    ∂y = (U([x, y + ε]) - U([x, y - ε])) / (2ε)
    return [∂x, ∂y]
end

# HMC step
function hmc_step(q, ε=0.05, L=20)
    # Initial momentum
    p = randn(2)
    
    # Store initial position and momentum
    current_q = copy(q)
    current_p = copy(p)
    
    # Leapfrog integration
    current_p .-= ε * ∇U(current_q) / 2  # Half step for momentum
    
    for i in 1:L
        current_q .+= ε * current_p       # Full step for position
        if i != L
            current_p .-= ε * ∇U(current_q) # Full step for momentum
        end
    end
    
    current_p .-= ε * ∇U(current_q) / 2  # Half step for momentum
    
    return current_q
end

# Generate samples
function generate_trajectory(n_steps; start=[-2.0, -2.0])
    trajectory = zeros(2, n_steps)
    q = copy(start)
    trajectory[:, 1] = q
    
    for i in 2:n_steps
        q = hmc_step(q)
        trajectory[:, i] = q
    end
    
    return trajectory
end

function hill_climb_step(q, step_size=0.1)
    grad = ∇U(q)                          # Get gradient
    grad_norm = norm(grad)                # Get magnitude
    
    # If gradient is very small, we're at a local maximum
    if grad_norm < 1e-5
        return q
    end
    
    # Move in direction of gradient
    grad_normalized = grad / grad_norm    # Unit vector in gradient direction
    return q - step_size * grad_normalized  # Take step in that direction
end

q = [-2.0, -2.0]
qs = [q]
for i in 1:1000
    q = hill_climb_step(q)
    push!(qs,q)
end



# Visualization
function plot_hmc_trajectory(n_steps=100; view_angle=(45,45))
    gr(size=(800,600),dpi=300)  # Set backend and plot size
    # Generate contour data
    x = range(-10, 10, length=1000)
    y = range(-10, 10, length=1000)
    z = [U([i, j]) for i in x, j in y]
    
    # Generate trajectory
    trajectory = generate_trajectory(n_steps)
    
    # Create 3D surface plot
    p1 = surface(x, y, -z',
                 color=:viridis,
                 title="HMC Trajectory",
                 camera=(45, 45),  # 3D viewing angle
                 xlabel="x",
                 ylabel="y",
                 zlabel="p(x)",
                 legend=:topright)
    
    # Plot trajectory in 3D
    # Get z-coordinates for trajectory points
    z_traj = [U([trajectory[1,i], trajectory[2,i]]) for i in 1:size(trajectory,2)]
    
    # Plot 3D trajectory
    plot!(p1, trajectory[1,:], trajectory[2,:], -z_traj,
          linewidth=3, color=:red, label="HMC Path")
    
    # Plot starting point
    scatter!(p1, [trajectory[1,1]], [trajectory[2,1]], [-z_traj[1]],
            color=:green, label="Start", markersize=6)
    scatter!(p1, trajectory[1,:], trajectory[2,:], -z_traj,
    color=:grey, label="Samples", markersize=3)
    
    xlims!(-10,10)
    ylims!(-10,10)
    return p1
end

function climb(q)
    q = copy(q)
    for i in 1:1000
        q += 0.01 * ∇U(q)
    end
end



# Generate and display the plot
p = plot_hmc_trajectory(100)
savefig(p, "hmc_trajectory.png",)


using Random
x = range(-10, 10, length=1000)
y = range(-10, 10, length=1000)
z = [U([i, j]) for i in x, j in y]
    # Generate trajectory
    
    # Create 3D surface plot
p1 = surface(x, y, -z',
    color=:viridis,
    title="Gradient Ascent",
    camera=(45, 45),  # 3D viewing angle
    xlabel="x",
    ylabel="y",
    zlabel="p(x)",
    legend=:topright,
    dpi=300
    )

scatter!([-2],[-2],[-U([-2,-2])],color=:green,label="Start")
Q = reduce(hcat,qs)
scatter!([Q[1,end]],[Q[2,end]],[-U(Q[:,end])],color=:red,label="End")
plot!(Q[1,:],Q[2,:],-U.(eachcol(Q)),linewidth=3,color=:red,label="Gradient Ascent Path")


savefig(plot(p1,p,colorbar=false,size=(600,300),dpi=300),"path_opt.png")
