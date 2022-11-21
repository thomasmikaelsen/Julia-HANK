include("Lecture1-Aiyagari.jl")
params = IncompleteMarketsParam()
ss = SteadyState(params);
ss.y

### PARTIAL EQ AND 1-time MIT SHOCK ###
# Assume all incomes increase by 1% at time 6
T = 11
ys = repeat(ss.y',11)
ys[5,:] = ys[5,:] * 1.01 

# Initial Va and containers 
Va = ss.Va
a = [zeros(7,500) for i in 1:T]
c = [zeros(7,500) for i in 1:T];
for t in T:-1:1
    Va, a[t], c[t] = EulerBack(Va, ss.P, ss.Agrid, ys[t,:], ss.r, ss.β, ss.σ)
end

@tullio c_impulse[i] := 100 .* (c[i] - ss.cpol) ./ ss.cpol;

i_ave = searchsortedfirst(ss.Agrid, ss.A)
plt = plot([c_impulse[i][1,i_ave] for i in 1:11],title="Consumption Policy Response to 1% y increase")
for s in [4 7]
    plot!([c_impulse[i][s,i_ave] for i in 1:11])
end
plt

D = copy.(a)
D[1] = ss.D

for t in 1:T-1
    lottery = GetLottery.(a[t]; Agrid = ss.Agrid)
    D[t+1] = ForwardIteration(D[t], ss.P, lottery)
end

A, C = zeros(T), zeros(T)
for t in 1:T
    A[t] = sum(D[t] .* a[t])
    C[t] = sum(D[t] .* c[t])
end

C_impulse = 100*(C .- ss.C) ./ ss.C
A_impulse = 100*(A .- ss.A) ./ ss.C
plt = plot(C_impulse, label = "C", title = "% change in response to +1% increase at t=6")
plot!(A_impulse, label = "A")

function yPolicyImpulse(ss, shocksize, shocktime, T)
    ns = length(ss.apol[:,1])
    na = length(ss.apol[1,:])
    Va = [zeros(ns,na) for i in 1:T]
    Va[T] = ss.Va
    a = [zeros(ns,na) for i in 1:T]
    c = [zeros(ns,na) for i in 1:T]
    ys = repeat(ss.y',T)
    ys[shocktime,:] = ys[shocktime,:] * (1 + shocksize) 
    for t in T:-1:1
        if t == T
            Va[t], a[t], c[t] = EulerBack(ss.Va, ss.P, ss.Agrid, ys[t,:], ss.r, ss.β, ss.σ)
        else 
            Va[t], a[t], c[t] = EulerBack(Va[t+1], ss.P, ss.Agrid, ys[t,:], ss.r, ss.β, ss.σ)
        end
    end
    return Va, a, c
end

Va_alt, a_alt, c_alt = yPolicyImpulse(ss, 0.01, 5, 11);

function yDistributionImpulse(ss, apol, T)
    D = copy.(apol)
    D[1] = ss.D
    
    for t in 1:T-1
        lottery = GetLottery.(apol[t]; Agrid = ss.Agrid)
        D[t+1] = ForwardIteration(D[t], ss.P, lottery)
    end
    return D
end

function yHouseholdImpulse(ss, shocksize, shocktime, T)
    Va, a, c = yPolicyImpulse(ss, shocksize, shocktime, T)
    D = yDistributionImpulse(ss, a, T)
    @tullio A[t] := sum(D[t] .* a[t])
    @tullio C[t] := sum(D[t] .* c[t])
    return (; Va = Va, a = a, c = c, D = D, A = A, C = C)
end

y_impulse = yHouseholdImpulse(ss, 0.01, 5, 11);

C_impulse = 100*(y_impulse.C .- ss.C) ./ ss.C
A_impulse = 100*(y_impulse.A .- ss.A) ./ ss.C
plt = plot(C_impulse, label = "C", title = "% change in response to +1% increase at t=5")
plot!(A_impulse, label = "A")


### GE IMPULSE AND PERSISTENT SHOCK ###
ss = SteadyState(params)
B = 5.6
τ = ss.r*B
e = ss.y
function BetaCalibGE(Pi, Agrid, τ, e, r, σ, B)
    β_ge = find_zero(x ->  SteadyStateManual(Pi, Agrid, (1-τ)*e, r, x, σ).A - B, [0.98 0.995])    # Calibrate β to be consistentwith asset market clearing A = B
    ss_ge = SteadyStateManual(Pi, Agrid, (1-τ)*e, r, β_ge, σ)
    A_ge, C_ge = ss_ge.A, ss_ge.C

    if isapprox(C_ge, 1) == 1 & isapprox(A_ge, B) == 1
        println("Calibration succesful")
    else 
        println("Calibration unsuccesful")
    end
    return β_ge, ss_ge, e
end
β_ge, ss_ge, ss_ge_y = BetaCalibGE(ss.P, ss.Agrid, τ, e, ss.r, ss.σ, B)

# Persistent shock to TFP 
T = 300
Zs = [1 + 0.01*0.95^(t-1) for t in 1:T];

# TFP impulse functions: given a TFP sequence and a guess of rs, calculate policies for the period 1:T
function zPolicyImpulse(ss, y, rs, zs, T)
    # Productivity affects taxes through affecting r 
    taus = rs * 5.6
    # Productivity affects income through taxes
    @tullio ys[i] := (zs[i] - taus[i]) * y 
    ns = length(ss.apol[:,1])
    na = length(ss.apol[1,:])
    Va = [zeros(ns,na) for i in 1:T]
    Va[T] = ss.Va
    a = [zeros(ns,na) for i in 1:T]
    c = [zeros(ns,na) for i in 1:T]
    for t in T:-1:1
        if t == T
            Va[t], a[t], c[t] = EulerBack(ss.Va, ss.P, ss.Agrid, ys[t], rs[t], ss.β, ss.σ)
        else 
            Va[t], a[t], c[t] = EulerBack(Va[t+1], ss.P, ss.Agrid, ys[t], rs[t], ss.β, ss.σ)
        end
    end
    return Va, a, c, taus, ys
end

Va_z, a_z, c_z, taus_z, ys_z = zPolicyImpulse(ss_ge, ss_ge_y, ss_ge.r*ones(T), Zs, T);


function zHouseholdImpulse(ss, y, rs, zs, T)
    Va, a, c, taus, ys = zPolicyImpulse(ss, y, rs, zs, T)
    # Same for y and z
    D = yDistributionImpulse(ss, a, T)
    @tullio A[t] := sum(D[t] .* a[t])
    @tullio C[t] := sum(D[t] .* c[t])
    return (; Va = Va, a = a, c = c, D = D, A = A, C = C, taus = taus, ys = ys)
end

Va_z, a_z, c_z, D_z, A_z, C_z, taus_z, ys_z = zHouseholdImpulse(ss_ge,ss_ge_y,ss_ge.r*ones(T),Zs,T);

function zImpulseMap(ss, y, rs, zs, T)
    impulse = zHouseholdImpulse(ss, y, rs, zs, T)
    return impulse.A .- 5.6, impulse
end
excess, impulse_z = zImpulseMap(ss_ge, ss_ge_y, ss_ge.r*ones(T), Zs, T);
plot(excess, title = "Excess demand for assets, given TFP and guess of r")


### Ad hoc iteration on r 
# Doesn't plot like the lecture but returns are the same 
function AdHocIteration(ss, y, Zs, T)
    rs = ss_ge.r * ones(T)
    for it in 1:400
        excess, impulse = zImpulseMap(ss, y, rs, Zs, T)
        if maximum(abs.(excess)) < 5e-4
            println("Converged after $(it) iterations")
            return rs
        end
        rs[2:end] -= 0.002*excess[1:end-1]
    end
end

rs = AdHocIteration(ss_ge, ss_ge_y, Zs, T)


### Use Sequence space jacobian to update guess 
J = zeros(T,T);
h = 1e-4
# No shock to r and no TFP shock: result is same as lecture to 11th decimal place
no_shock_excess, no_shock_imp = zImpulseMap(ss_ge, ss_ge_y, ss_ge.r*ones(T), ones(T), T);
# Shock to r at different times: gives Jacobian
@Threads.threads for tshock in 1:T
    # @tullio rs[i] = ss_ge.r*ones(T)[i] + h*(tshock == i)
    J[:,tshock] = (zImpulseMap(ss_ge, ss_ge_y, ss_ge.r*ones(T) .+ h * (1:T .== tshock), ones(T), T)[1] - no_shock_excess)/h
end
J[1:10,1]


plt = plot(J[1:60,1])
for j in [6 11 16 21]
    plot!(J[1:60,j])
end
plt

# Use Jacobian to update guess 
rs = ss_ge.r * ones(T)
Jbar = J[1:end-1,2:end]
errs = Float64[]
err = 0.0
for it in 1:30
    excess, impulse = zImpulseMap(ss_ge, ss_ge_y, rs, Zs, T)
    err = maximum(abs.(excess[1:end-1]))
    push!(errs, err)
    if err < 1e-10
        println("Asset market clearing up to 12 digits after $(it) iterations")
        break 
    end 
    rs[2:end] -= Jbar\excess[1:end-1]
end

### FAKE NEWS ALGORITHM: CALCULATE JACOBIANS FASTER ###
""" 
Fake news: J requires computing policy response to shocks at s=T-1,s=T-2,... 
For each of these, we de backward iteration over all T periods. Unnecessary due to symmetry:
Use that policy response at t to time s shock = response at t-1 to shock at s-1 
So, suppose we shock the economy at time T. Backward iterate to get policy responses for 
T, T-1,...,1 = pol_T,T , pol_T,T-1,...,pol_T,1. Then policy response at time 20 to a shock 
at time 169 is pol_T,T-(169-20) = pol_T,T-149
Recall that policy responses are forward-looking only, not backward-looking, so policy responses 
 after the shock are 0. The converse holds for distributions 
"""
# Test that impulses only depend on distance to shock 
imp4 = zImpulseMap(ss_ge, ss_ge_y, ss_ge.r*ones(T) .+ h * (1:T .== 5), ones(T), T)[2].a
@tullio impulse_s4[i] := (zImpulseMap(ss_ge, ss_ge_y, ss_ge.r*ones(T) .+ h * (1:T .== 5), ones(T), T)[2].a[i] - ss_ge.apol)/h;
@tullio impulse_s5[i] := (zImpulseMap(ss_ge, ss_ge_y, ss_ge.r*ones(T) .+ h * (1:T .== 6), ones(T), T)[2].a[i] - ss_ge.apol)/h;
@tullio imps4[i] := impulse_s4[i][4,51]
@tullio imps5[i] := impulse_s5[i][4,51]
plt = plot(4:-1:-4,imps4[1:9])
plot!(5:-1:-4,imps5[1:10])

# Assume shock at latest possible time T
da = zImpulseMap(ss_ge, ss_ge_y, ss_ge.r*ones(T) .+ h * (1:T .== T), ones(T), T)[2].a
@tullio da[i] = da[i] - ss_ge.apol

# 
J_alt = zeros(T,T);
# Start with steady state policy - will add the impulse to this for each shock
a_ss = [ss_ge.apol .* ones(params.Ns,params.Na) for t in 1:T]

@Threads.threads for s in 1:T                       # For a shock at time s...
    a = copy.(a_ss)                                 # Start with steady state policy 
    a[1:s] += da[T-s+1:end]                           # In period 1, distance is s-1, so add policy T-(s-1)
    D = yDistributionImpulse(ss_ge, a, T)
    A = zeros(T)
    for t in 1:T
        A[t] = sum(D[t] .* a[t])
    end
    J_alt[:, s] = (A .- B - no_shock_excess)/h
end

## FAKE NEWS: Distribution part ##
"""
Policy responses only depend on distance to shock. However, distribution evolves
up to, so J_t,s ≂̸ J_t-1,s-1. But this difference is only due to the extra anticipation effect 
from the distribution and this can be calculated as below.
"""
# Given a shock at time s = 10
a = copy.(a_ss)
a[1] += da[T-10]
D = yDistributionImpulse(ss_ge, a, T)
A = zeros(T)
for t in 1:T
    A[t] = sum(D[t] .* a[t])
end
anticipation_effect = (A .- B - no_shock_excess)/h

# Fake news matrix 
F = copy.(J)
F[2:end,2:end] -= J[1:end-1,1:end-1]


# Calculating entire jacobian using this insight 
F_alt = zeros(T,T);
@Threads.threads for s in 1:T
    a = copy.(a_ss)
    a[1] += da[T+1-s]
    D = yDistributionImpulse(ss_ge, a, T)
    A = zeros(T)
    for t in 1:T
        A[t] = sum(D[t] .* a[t])
    end
    F_alt[:,s] = (A .- B - no_shock_excess)/h
end

function JFromF(F)
    J = copy.(F)
    for t in 2:T 
        J[2:end, t] += J[1:end-1,t-1]
    end
    return J
end

Jff = JFromF(F)

### USE EXPECTATION VECTORS TO GET DISTRIBUTION ###
y, pi, Pi = DiscretizeIncome(params.ρ, params.σ_y, params.Ns)
a_i = [GetLottery(a; Agrid = ss_ge.Agrid)[1] for a in ss_ge.apol]
a_pi = [GetLottery(a; Agrid = ss_ge.Agrid)[2] for a in ss_ge.apol]
curlyE = ExpectationVectors(ss_ge.apol, Pi, a_i, a_pi, T);
@tullio cE1[i] := curlyE[i][4,1] 
@tullio cE151[i] := curlyE[i][4,151] 
@tullio cE251[i] := curlyE[i][4,251] 
plt = plot(cE1[1:131])
plot!(cE151[1:131])
plot!(cE251[1:131])

# Full fake news algorithm 
F = zeros(T,T)
lottery = GetLottery.(ss_ge.apol; Agrid = ss_ge.Agrid)
D1_noshock = ForwardIteration(ss_ge.D, Pi, lottery)
for s in 1:T-1
    # F(1,s) : change in asset policy times steady state incoming distribution 
    F[1,s] = sum(ss_ge.D .* da[T-s+1]) / h 

    # Change in D_1 from this cnage 
    lottery_shock = GetLottery.(ss_ge.apol + da[T-s+1]; Agrid = ss_ge.Agrid)
    dD1 = ForwardIteration(ss_ge.D, Pi, lottery_shock) - D1_noshock
    
    # Use expectation vectors to project effect on aggregate 
    curlye = curlyE[1:T-1]
    @tullio f[i] := sum(curlye[i] .* dD1) / h
    F[2:end,s] = f 
end
J_alt = JFromF(F)

"""
Fake news algorithm:
For each shocked inuput i (=r) and policy of interest o (=a):
1. For each i, iterate backward from shock at T-1 to find:
    1a. Effect on policies o, aggregated using SS distribution, at each horizon s, stored in Y_s^(o,i)
    1b. Effect on one-period-ahead distribution at each horizon s, stored in D_s^i

2. For each o, do expectation iteration to obtain expectation vectors at horizon t up to T-1, store in E_t^0 

3. Assemble fake new matrix F^(o,i) for each pair (o,i) using the formula 

4. Use recursion to obtain J 
"""