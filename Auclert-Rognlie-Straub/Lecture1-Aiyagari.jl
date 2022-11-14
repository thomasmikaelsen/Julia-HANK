using Tullio, BenchmarkTools, LinearAlgebra, Interpolations, Plots, Optim, Roots

function DiscretizeAssets(amin, amax, Na)
    """"
    Double exponential spaced grid 
    """
    ubar = log(1 + log(1 + amax - amin))
    ugrid = range(0, ubar, length = Na)
    agrid = zeros(length(ugrid))
    @tullio agrid[i] = amin + exp(exp(ugrid[i])-1) - 1
    return agrid
end
#Agrid = DiscretizeAssets(0, 100, 100)
# Discretize income state 
function RouwenhorstP(N, p)
    """
    # Rouwenhorst method (Kopecky and Suen): Models sum of N-1 independent, binary states (0 or 1), 
    # each of which have probability p of staying a current value, 1-p of switching
    """
    # Base case 
    P = [p 1-p 
         1-p p]

    # Build Pn recursively 
    for n in 3:N
        P_old = copy.(P)
        P = zeros(n,n)

        P[1:end-1, 1:end-1] += p * P_old 
        P[1:end-1, 2:end] += (1-p) * P_old 
        P[2:end, 1:end-1] += (1-p) * P_old 
        P[2:end, 2:end] += p * P_old 
        P[2:end-1, :] /= 2
    end 
    return P
end


# Stationary distribution of markov chain 
function StationaryMarkov(P)
    """
    We know the eigenvalue associated with the stationary distribution is 1, so we impose 
    that the first entry in the stationary distribution is non-zero (1 in this case)
    and solve the resulting n-1 dimensional linear problem (I - P'[2:end])x = 0 which 
    is invertible, if the Ergodic theorem holds (i.e. P is aperiodic and irreducible <= Pij > 0)
    """
    Π = [1; (I - P'[2:end,2:end]) \ Vector(P'[2:end,1])]
    return Π = Π ./ sum(Π)
end

#P = RouwenhorstP(10, 1/3)
#Π = StationaryMarkov(P)

function DiscretizeIncome(ρ, σ, Ns)
    """
    # Match AR(1) process using Rouwenhorst: set p = (1+ρ)/2 and scale states s by α = 2σ/√(N-1)
    """
    p = (1+ρ)/2

    # Assume states are 0 to Ns-1, then scale using α 
    s = 0:Ns-1
    α = (2*σ)/sqrt(Ns-1)
    s = α .* s 

    # Markov transition matrix and stationary distribution 
    P = RouwenhorstP(Ns, p)
    Π = StationaryMarkov(P)

    # s is log income, get income y and scale so that mean is 1
    y = exp.(s)
    y /= dot(Π, y)

    return y, Π, P
end 


#y, Π, Pi = DiscretizeIncome(0.975, 0.7, 7)



# EGM 
function EulerBack(Va, P, Agrid, y, r, β, σ)
    # 1. Discounted, expected value of Va tomorrow for all possible a', given today's state 
    Wa = (β .* P) * Va 

    # 2. Endogenous consumption from EGM equation
    c_endo = Wa.^(-σ)

    # 3. Cash-on-hand grid 
    coh = zeros(length(y), length(Agrid))
    @tullio coh[i,:] = y[i] .+ (1+r)*Agrid  

    a = copy.(coh)
    # interpolations[state] : c_endo[state,a'] + a' = coh ↦ a'
    interpolations = [linear_interpolation(c_endo[s,:] + Agrid, Agrid, extrapolation_bc = Line()) for s in 1:length(y)]
    # Evaluatng the interpolation at a coh[s,a] point implies an implicit a' which is collected in a[s,:]
    @tullio a[s, :] = interpolations[s](coh[s,:])

    # 4. Enforce borrowing constraint 
    a = max.(a, Agrid[1])
    c = coh - a

    # 5. Use envelop to get derivative of value function 
    Va = (1+r) * c.^(-1/σ)

    return Va, a, c
end

function PolicySS(P, Agrid, y, r, β, σ, tol=1e-9, maxiter=10000)
    # Initial guess for Va: consume 5% of cash-on-hand and get Va from envelope 
    coh = zeros(length(y), length(Agrid))
    @tullio coh[i,:] = y[i] .+ (1+r)*Agrid  
    c = 0.05*coh 
    Va = (1+r) * c.^(-1/σ)
    a_old = copy.(coh)

    # Iterate until maximum distance between two iterations fall below tol, or maxiter is reached 
    for it in 1:maxiter 
        Va, a, c = EulerBack(Va, P, Agrid, y, r, β, σ)
        if it > 1 && maximum(abs.(a - a_old)) < tol
            return Va, a, c 
        end 
        a_old = copy.(a)
    end
end




# Solve the model
Agrid = DiscretizeAssets(0, 10000, 500)
y, Π, P = DiscretizeIncome(0.975, 0.7, 7)
r = 0.01/4
β = 1 - 0.08/4
σ = 1

Va, a, c = PolicySS(P, Agrid, y, r, β, σ);
Va[3,67]

###################
###### PLOTS ######
###################
# Consumption policy c(a,s)
plt = plot(Agrid, c[1,:], label = "y = $(round(y[1], digits=2))")
for s in 2:length(y)
    plot!(Agrid, c[s,:], label = "y = $(round(y[s], digits=2))")
end
plt

# Consumption policy at the bottom 
plt = plot(Agrid[1:120], c[1,1:120], label = "y = $(round(y[1], digits=2))")
for s in 2:length(y)
    plot!(Agrid[1:120], c[s,1:120], label = "y = $(round(y[s], digits=2))")
end
plt

# Consumption is especially concave in low states 
plt = plot(Agrid[1:120], c[1,1:120], label = "y = $(round(y[1], digits=2))")

# Not in high states (note we're even closer to 0 assets here than in the previous plot)
plt = plot(Agrid[1:10], c[7,1:10], label = "y = $(round(y[7], digits=2))")

# Net savings = apol(a,s) - a for a ∈ Agrid, i.e. whether agents save more than their current assets. If NS > 0, it means that ppl 
# save more than the assets they currently have. We see that in the low states, everyone is a net borrower: if they have 0 assets, they save 0 assets.
# And if they are in the low state net savings is decreasing in assets, that is if I have 20 assets today, I save approx 19 assets 
# because I want to consume more today. I.e. I run down my assets in the low state because I expect to reach a higher state in the future and.
# In the high state, I save more than the assets I currently have because of precaution: I might reach a lower state in the future. Unless
# I have very high assets in the high state, in which case I run down my assets because I can consume more without fear of having too few assets if I hit a low state.
plt = plot(Agrid[1:300], a[1,1:300] - Agrid[1:300], label = "y = $(round(y[1], digits=2))")
plot!([0; Agrid[300]], [0 ; 0], lw = 2, lc =:black, label = "")
for s in 2:length(y)
    plot!(Agrid[1:300], a[s,1:300] - Agrid[1:300], label = "y = $(round(y[s], digits=2))")
end
plt


# Marginal Propensities to Consume (MPC) by hand 
mpcs = copy.(c)
for s in 1:length(y)
    # Away from boundaries: symmetric differences. MPC[n] = c[n+1] - c[c-1] / (Agrid[n+1] - Agrid[n-1]) / 1+r
    mpcs[s, 2:end-1] = (c[s,3:end] - c[s,1:end-2]) ./ (Agrid[3:end] - Agrid[1:end-2]) / (1+r)
    # At boundary: asymmetric differences. MPC[1] = c[2] - c[1] / Agrid[2] - Agrid[1] / 1+r
    mpcs[s, 1] = (c[s, 2] - c[s, 1]) ./ (Agrid[2] - Agrid[1]) / (1+r)
    mpcs[s, end] = (c[s, end] - c[s, end-1]) ./ (Agrid[end] - Agrid[end-1]) / (1+r)
end
# Impose that everything is consumed (MPC = 1) at the borrowing constraint 
mpcs[findall(==(Agrid[1]), a)] .= 1.0

# MPC plots. Concave policy functions imply decreasing MPC in assets. The MPCs jump from 1 to ≈ 0.5 as households go from constrained
# to "not constrained but probably constrained tomorrow" because these hhs will smooth consumption between those two states. This jump 
# happens when time and income are discrete 
plt = plot(Agrid[1:50], mpcs[1,1:50], xlabel = "Assets", ylabel = "MPC", label = "y = $(round(y[1], digits=2))")
for s in 2:length(y)
    plot!(Agrid[1:50], mpcs[s,1:50], label = "y = $(round(y[s], digits=2))")
end
plt



##########################
###### DISTRIBUTION ######
##########################

# Dealing with off-grid policies 
function GetLottery(a; Agrid)
    ss = searchsortedfirst(Agrid, a)                                # i such that A_i-1 a <= A_i
    if ss == 1     
        index = 1
        π = 1                                                       # If a' <= amin => assign all weight to amin 
    elseif ss >= length(Agrid)+1
        index = length(Agrid)-1
        π = 0                                                       # If a' >= amax => assign all weight to amax 
    else 
        index = ss - 1                                              # Index such that Agrid[index] <= a <= Agrid[index+1]
        π = (Agrid[index+1] - a)/(Agrid[index+1]-Agrid[index])      # Lottery probability
    end
    return index, π
end

#function GetLottery(a; Agrid)
#    index = searchsortedfirst(Agrid, a) - 1                        # Index i such that a_i <= a <= a_i+1
#    index = min(max(index,1),499)                                            
#    π = (Agrid[index+1] - a)/(Agrid[index+1]-Agrid[index])
#    π = max(min(π,1),0)
#    return index, π
#end

#searchsortedfirst(Agrid, a[7,500])
#GetLottery.(9999.9999; Agrid=Agrid)
#Agrid[498]
#Agrid[499]
#GetLottery(Agrid[end]; Agrid=Agrid)

#GetLottery(10000; Agrid=Agrid)

#searchsortedfirst(Agrid,10000)

#Lottery  = GetLottery.(a; Agrid)
#length(Lottery[1,:])
#length(Lottery[:,1])


function ForwardPolicy(D, Lottery)
    """
    Given a discretized distribution D_t(s,a) with mass on each gridpoint, send π(s,a) mass to gridpoint
    i(s,a) and 1-π(s,a) to gridpoint i(s,a)+1. Return D_t^end(s,a') before the s' draw 
    I.e. hhs with (s,a) are sent to (s,a') by the policy, with Agrid[index] <= a' <= Agrid[index+1] for some index. 
    Use the Lottery to assign π(s,a) mass to (s,Agrid[index]) and 1-π(s,a) mass to (s,Agrid[index+1])
    """
    Dend = zeros(length(D[:,1]), length(D[1,:]))
    for s in 1:length(Lottery[:,1])
        for a in 1:length(Lottery[1,:])
            # Send π(s,a) of the mass to gridpoint i(s,a)
            Dend[s, Lottery[s,a][1]] += Lottery[s,a][2] * D[s,a]

            # Send 1-π(s,a) of the mass to gridpoint i(s,a)+1
            Dend[s, Lottery[s,a][1]+1] += (1-Lottery[s,a][2]) * D[s,a]
        end
    end
    return Dend
end

#Dend = ForwardPolicy(ones(7,500)./sum(ones(7,500)), Lottery)

function ForwardIteration(D, P, Lottery)
    """
    Input: Discretized distribution D_t(s,a), state transition matrix P and the Lottery.
    Output: D_t+1(s',a') i.e. the full iteration one period forward. ForwardPolicy tells us 
    (s,a') given (s,a). For each s, the hh can end up in one of length(y) states s' tomorrow 
    so the mass assigned to (s',a') should be the sum of all the points (si,a') * P(s' ∣ si)
    """ 
    Dend = ForwardPolicy(D, Lottery)
    return P' * Dend                            # Each masspoint Dend[s,a'] has probability P(s' ∣  s) of being in (s',a')
end

# Good guess since its the right distribution on s and it's uniform on a (max entropy)
#D0 = ones(7,500) 
#@tullio D0[i,:] = (Π[i] / 500) * D0[i,:]

function SSDistribution(P, apol, Agrid, tol=1e-10, maxiter=10000)
    Lottery = GetLottery.(apol; Agrid)
    Π = StationaryMarkov(P)
    # Initial guess: stationary over y, uniform over a 
    D = ones(length(P[:,1]),length(Agrid)) 
    @tullio D[i,:] = (Π[i] / length(Agrid)) * D[i,:]

    # Iterate until convergence 
    for it in 1:maxiter
        D_new = ForwardIteration(D, P, Lottery)
        if maximum(abs.(D_new - D)) < tol 
            return D_new 
        end 
        D = copy.(D_new)
    end
    return D
end

# Plot CDF below assets = 20 since almost everyone have assets below 
λss = SSDistribution(P, a, Agrid)
@tullio λcdf[i] := sum(λss[:,i])
λcdf = cumsum(λcdf)
i = searchsortedfirst(Agrid, 20)
plot(Agrid[1:i], λcdf[1:i])

# Plot CDF below assets = 2 
i = searchsortedfirst(Agrid, 2)
plot(Agrid[1:i], λcdf[1:i])

# Plot CDF by income state: normalize by 1/(probability of being in that state)
# Weird behavior with middle state y = 1.39, the stair-like graph indicate that there are mass points 
# above a = 0 for y = 1.39. This is because for each y < 1.39 there are mass points at 
# a = 0. As these households jump to state y = 1.39 they save for one, two, three,... periods until 
# they reach the next asset level at which there is a kink and here they all save the same but for slightly 
# fewer periods, and so on until the graph smoothes out. 
# Furthermore, there's a kink in y=1.39 around 1.15 because for y=1.39 the target asset is slightly above 1.
# Hence people with y=1.39 and assets below or above that level will both converge to that point. However,
# they won't cross and since more people converge from below than from above, the density is kinked. 
plt = plot(Agrid[1:150], cumsum(λss[1,1:150])./Π[1], label = "y = $(round(y[1],digits=2))")
for s in 2:length(y)
    plot!(Agrid[1:150], cumsum(λss[s,1:150])./Π[s], label = "y = $(round(y[s],digits=2))")
end 
plt


# Assets by income, relative to income 
for s in 1:length(y)
    println("Ave assets at y = $(round(y[s],digits=2)): ", round(dot(Agrid, λss[s,:]) ./ Π[s],digits=2))
end

# Total assets held in each state 
totalassets = sum(λss * Agrid)
for s in 1:length(y)
    println("Total assets at y = $(round(y[s],digits=2))  ($(round(Π[s]*100,digits=1))% of hhs): ", round(dot(Agrid, λss[s,:]),digits=2))
end

##########################################
#### AGGREGATING TO FULL STEADY STATE ####
##########################################

function SteadyState(P, Agrid, y, r, β, σ)
    Va, apol, cpol = PolicySS(P, Agrid, y, r, β, σ)
    D = SSDistribution(P, apol, Agrid)

    return (; D = D, Va = Va, apol = apol, cpol = cpol, A = dot(apol, D),
             C = dot(cpol, D), P = P, Agrid = Agrid, y = y, r = r, β = β,
             σ = σ)
end

#ss = SteadyState(P, Agrid, y, r, β, σ);
#ss.D

# Comparative statics in r: Steady state assets 
rs = r .+ range(-0.02, 0.015, length=15)
Ar = zeros(15)
Threads.@threads for i in 1:15 
    Ar[i] = SteadyState(P, Agrid, y, rs[i], β, σ).A
end
plot(rs, Ar, title="Comparative Statics in r", 
     ylabel="Aggregate assets over quarterly income", xlabel = "Real interest rate",
     legend=false)
vline!([β^(-1)-1])

# Comparative statics in income risk: We manipulate income risk by changing σ in the income process
# As income risk increases, assets increas for precautionary reasons  
σincs = range(0.3,1.2,length=8)
incomes = [DiscretizeIncome(0.975, i, 7) for i in σincs]
Ar = zeros(8)
Threads.@threads for i in 1:8 
    Ar[i] = SteadyState(incomes[i][3], Agrid, incomes[i][1], r, β, σ).A
end
plot(σincs, Ar, title="Comparative Statics in income risk", 
     ylabel="Assets over quarterly income", xlabel = "Cross-sectional standard dev. of income",
     legend=false)

# Comparative statics in intertemporal substitution 
# Assets are decreasing in EIS, or equivalently, increasing in risk aversion
# The more risk averse you are, the more you want to smooth consumption intertemporally
# The higher the elasticity of inter-temporal substitution, the more willing you are to 
# vary consumption over time and so you need fewer assets to dampen shocks and 
# assets held are assets not converted into consumption 
σstemp = range(0.4,2,length=10)
Ar = zeros(10)
Threads.@threads for i in 1:10 
    Ar[i] = SteadyState(P, Agrid, y, r, β, σstemp[i]).A
end
plot(σstemp, Ar, title="Comparative Statics in EIS", 
     ylabel="Assets over quarterly income", xlabel = "Elasticity of Intertemporal Substitution",
     legend=false)



#########################
###### CALIBRATION ######
#########################
"""
Inspired by (McKay, Nakamura, Steinsson) we want to calibrate the model such that 
total assets are 140% of annual GDP. Since income in this model is quarterly, we want 
A such that 4*1.4 = A = 5.6. We allow β to vary in order to do this 
""";
# Find β such that assets match target 
β_calib = find_zero(x ->  SteadyState(P, Agrid, y, r, x, σ).A - 5.6, [0.98 0.995])

# Verify we get the right assets 
ss = SteadyState(P, Agrid, y, r, β_calib, σ)
A, C = ss.A, ss.C
# Check aggregate steady-state budget balance: C = 1+rA 
isapprox(C, 1+r*A)


#################################
###### GENERAL EQUILIBRIUM ######
#################################
"""
We vary income risk, keeping everything else constant. We vary r to satisfy asset market clearing for 
each income risk. r enters the HH problem as a separate variable and in the government budget constraint. 
This means that the tax changes with r. As income risk increases, household want to save more and so
r falls to make sure that the market still clears. 
"""
B = 5.6                     # Bonds supplied by government, paying real interest r 
τ = r*B                     # Labor tax to balance steady-state government budget 
e = y
β_ge = find_zero(x ->  SteadyState(P, Agrid, (1-τ)*e, r, x, σ).A - B, [0.98 0.995])    # Calibrate β to be consistent with asset market clearing A = B
ss_ge = SteadyState(P, Agrid, (1-τ)*e, r, β_ge, σ)
A_ge, C_ge = ss_ge.A, ss_ge.C
isapprox(C_ge, 1)
isapprox(A_ge, B)

# GE counterfactuals: As income risk increases, so does the demand for assets to help smooth consumption.
# This brings down the return on assets. 
function GECounter(β_ge, Agrid, B, σ)
    σincs = range(0.3,1.2,length=8)
    r_ge = zeros(8)
    Threads.@threads for i in 1:length(σincs)
        println(i)
        ei, Πi, Pi = DiscretizeIncome(0.975, σincs[i], 7)
        r_ge[i] = find_zero(x ->  SteadyState(Pi, Agrid, (1-x*B)*ei, x, β_ge, σ).A - B, [-0.02 0.015])
    end
    return r_ge
end

rge = GECounter(β_ge, Agrid, B, σ)
rge = 4 .* rge

plot(σincs, rge, xlabel = "Cross-sectional standard deviation of income", 
     ylabel = "Equilibrium real interest rate (annualize)", legend = false)

#################################
###### EXPECTATION VECTORS ######
#################################
function ExpectationPolicy(Xend, ai, a_pi)
    """
    Goes from X_t^(end)(s,a') - before an s' draw - to X_t(s,a), i.e. the opposite way of ForwardPolicy. 
    Takes a lottery - i.e. all elements a_i that are being mapped to tomorrow from states today along with the weight given to them - 
    and creates the expected value today. The intuition is this: a policy x(s,a) maps (s,a) to something off grid x(s,a) = (s,a')
    that lands between two grid points a_i and a_(i+1), with weight a_pi(a,s) and (1-a_pi(a,s)) respectively. 
    Now, take any grid point tomorow a_i that is hit by x(s,a) for some (s,a). We know that a_i(s,a) <= x(s,a) < a_(i+1)(s,a) and that 
    each grid point is assigned weights a_pi(a,s) and (1-a_pi(a,s)) respectively, so we can calculate the expected value of 
    x(s,a) as the mixture of a_i(s,a) and a_(i+1)(s,a). I think this expectation is actually identical to x(s,a), but it will depend
    on how the weights are constructed.
    """
    X = zero(Xend)
    for s in 1:length(a_i[:,1])
        for a in 1:length(ai[1,:])
            X[s,a] = a_pi[s,a] * Xend[s, a_i[s,a]] + (1-a_pi[s,a])*Xend[s, a_i[s,a]+1]
        end
    end
    return X
end


function ExpectationIteration(X, Pi, a_i, a_pi)
    """ 
    ExpectationPolicy takes X_t^(end)(s,a') and returns X_t(s,a). We now need to obtain X_t^(end)(s,a') from X_t+1(s',a'). 
    This is done by taking expectations, similarly to ExpectationPolicy which is also an expectation. 
    Note how to go forward we used P' to get distributions, so to reverse these operations, we take expectations using P. 
    Px gives a vector of conditional expectations. P'x gives a distribution. 
    Also note that the order of operations is reversed. Before, we did ForwardPolicy(s,a) ↦ (s,a') then 
    ForwardIteration(s,a') ↦ (s',a'). Now, we go from ExpectationIteration(s',a') ↦ (s,a'), then ExpectationPolicy(s,a') ↦ (s,a) 
    because we have to undo the operations in order. 
    """
    Xend = Pi * X
    return ExpectationPolicy(Xend, a_i, a_pi)
end

function ExpectationVectors(X, Pi, a_i, a_pi, T)
    curlyE = [zero(X) for t in 1:T]
    curlyE[1] = X 

    for j in 2:T
        curlyE[j] = ExpectationIteration(curlyE[j-1], Pi, a_i, a_pi) 
    end
    return curlyE
end

## Autocorrleation of consumption 
ctilde = c .- dot(λss,c)             # Demeaned consumption 
a_i = [GetLottery(a; Agrid = Agrid)[1] for a in a]
a_pi = [GetLottery(a; Agrid = Agrid)[2] for a in a]
E_ctilde = ExpectationVectors(ctilde, P, a_i, a_pi, 40)

Autocov_c = zeros(40)
for j in 1:40
    Autocov_c[j] = dot(λss, ctilde .* E_ctilde[j])     # ctilde .* E_ctilde[j] is the product of consumption in state (s,a) today with consumption in state (s,a) tomorrow. Multiply with distribution to get average autocovariance
end
# Autocorrelation is autocovariance divided by variance, and variance is autocovariance with lag 0
Autocorr_c = Autocov_c ./ Autocov_c[1]

plot(1:40, Autocorr_c, title = "Autocorrelation of consumption", ylabel = "Correlation", xlabel = "Horizon in quarters")

