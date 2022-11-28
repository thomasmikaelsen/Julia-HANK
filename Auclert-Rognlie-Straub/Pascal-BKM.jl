# Dependencies
using Distributions
using Plots
using DataFrames
using Random

max_iter=1000 #number of iterations for the simulation
a = 0.5
b = 0.05
sigma_shock=1.0 #variance of shocks
mu_shock=0. #mean of shocks
Random.seed!(1234)
d = Normal(mu_shock, sigma_shock)

function iter_x(x_min1::Float64, a::Float64, b::Float64)
    """
    Function to find the next iteration of x_{t} = a x_{t-1} + b x_{t-1}^2
    x_min1::Float64: x_{t-1}
    a::Float64 
    b::Float64 
    """
    return a*x_min1 + b*x_min1^2
end

# Simulation of an MIT Shock
# We assume that after max_iter_mit periods, the economy is back at the steady-state
max_iter_mit = 25
x_mit=zeros(max_iter_mit)
# Initial shock
z_t=zeros(max_iter_mit)
z_t[1] = sigma_shock #a 1 std. deviation
x_mit[1] = 0 #steady-state

for i=2:max_iter_mit
    x_mit[i] = iter_x(x_mit[i-1], a, b) + z_t[i-1]
end

# Scaled-version of the impulse response:
x_mit_scaled = x_mit./z_t[1];

# Scaled-version of the impulse response:
p0 = plot(x_mit_scaled, label="x scaled", xlabel="t")
title!("MIT shock")

# Function to calculate the path of xt using the BKM algorithm
function BKM_path!(XT::Array{Float64,1}, x_scaled::Array{Float64,1}, shocks::Array{Float64,1})
    """
    XT::Array{Float64,1}: array to store the evolution of the variable xt
    x_scaled::Array{Float64,1}: a scaled MIT shock
    shocks::Array{Float64,1}: sequence of shocks
    """
    # get the length of x_scaled
    len_x_scaled = length(x_scaled)
    max_iter = length(XT)
    
    # Loop over time periods periods
    for t=2:max_iter
        # Superposition of MIT shocks:
        for k=1:t
            # After some time, we assume that the effect of past shocks vanishes:
            if k<=len_x_scaled
                XT[t]+=x_scaled[k]*shocks[t-k+1]
            end
        end
    end
    
end

XT = zeros(max_iter) # Initialization
shocks_t = rand(d, max_iter).*0.5 # Series of shocks
@time BKM_path!(XT, x_mit_scaled, shocks_t) # Solving using BKM:
x_true = zeros(max_iter) # True value of the series
for i=2:max_iter
    x_true[i] = iter_x(x_true[i-1], a, b) + shocks_t[i-1]
end
x_true

# Let's store statistics on error:
diff_BKM = x_true - XT
max_abs_err_BKM = maximum(abs.(diff_BKM))
min_abs_err_BKM = minimum(abs.(diff_BKM))
mean_abs_err_BKM = mean(abs.(diff_BKM))
median_abs_err_BKM = median(abs.(diff_BKM))

p1 = plot(XT[2:end], label="x_t BKM")
plot!(x_true[2:end], label="x_t true")
title!("BKM with b=$(b)")
p1

