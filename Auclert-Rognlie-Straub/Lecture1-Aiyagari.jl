using Tullio, BenchmarkTools, LinearAlgebra, Interpolations, Plots, Optim, Roots, Parameters

@with_kw struct IncompleteMarketsParam
    r = 0.01/4
    β = 1 - 0.08/4
    σ = 1                                       # EIS
    amin = 0.0
    amax = 10000
    Na = 500                                    # Number of assets
    Ns = 7                                      # Number of income states
    ρ = 0.975                                   # Income process persistence
    σ_y = 0.7                                   # Income risk 
end         


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
#Agrid = DiscretizeAssets(0, 10000, 500)
#y, Π, P = DiscretizeIncome(0.975, 0.7, 7)
#r = 0.01/4
#β = 1 - 0.08/4
#σ = 1

#Va, a, c = PolicySS(P, Agrid, y, r, β, σ);
#Va[3,67]





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


##########################################
#### AGGREGATING TO FULL STEADY STATE ####
##########################################

function SteadyState(Params)
    Params = params
    @unpack r, β, σ, amin, amax, Na, ρ, σ_y, Ns = Params
    Agrid = DiscretizeAssets(amin,amax,Na)
    y, py, Py = DiscretizeIncome(ρ, σ_y, Ns)

    Va, apol, cpol = PolicySS(Py, Agrid, y, r, β, σ)
    D = SSDistribution(Py, apol, Agrid)

    return (; D = D, Va = Va, apol = apol, cpol = cpol, A = dot(apol, D),
             C = dot(cpol, D), P = Py, Agrid = Agrid, y = y, r = r, β = β,
             σ = σ)
end

function SteadyStateManual(P, Agrid, y, r, β, σ)
    Va, apol, cpol = PolicySS(P, Agrid, y, r, β, σ)
    D = SSDistribution(P, apol, Agrid)

    return (; D = D, Va = Va, apol = apol, cpol = cpol, A = dot(apol, D),
             C = dot(cpol, D), P = P, Agrid = Agrid, y = y, r = r, β = β,
             σ = σ)
end

#params = IncompleteMarketsParam()
#ss = SteadyState(params);

#####################
#### CALIBRATION ####
#####################
function BetaCalib(P,Agrid,y,r,σ,beta_low,beta_high,Assetlevel)
    β_calib = find_zero(x ->  SteadyState(P, Agrid, y, r, x, σ).A - Assetlevel, [beta_low beta_high])
    ss = ss = SteadyState(P, Agrid, y, r, β_calib, σ)
    A, C = ss.A, ss.C
    # Check aggregate steady-state budget balance: C = 1+rA 
    if isapprox(C, 1+r*A) == 1
        println("Calibration was succesful")
    else 
        println("Calibration failed")
    end
    return β_calib
end

#################################
###### GENERAL EQUILIBRIUM ######
#################################

function BetaCalibGE(Pi, Agrid, τ, e, r, σ, B)
    β_ge = find_zero(x ->  SteadyState(Pi, Agrid, (1-τ)*e, r, x, σ).A - B, [0.98 0.995])    # Calibrate β to be consistentwith asset market clearing A = B
    ss_ge = SteadyState(Pi, Agrid, (1-τ)*e, r, β_ge, σ)
    A_ge, C_ge = ss_ge.A, ss_ge.C

    if isapprox(C_ge, 1) == 1 & isapprox(A_ge, B) == 1
        println("Calibration succesful")
    else 
        println("Calibration unsuccesful")
    end
    return β_ge
end


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

#rge = GECounter(β_ge, Agrid, B, σ)
#rge = 4 .* rge

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