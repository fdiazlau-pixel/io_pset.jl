module io_pset_3
# Installing packages
using CSV, DataFrames, GLM, LinearAlgebra,Chain, DataFramesMeta,Statistics, Optim
##############################
# BLP — Piece 1: Setup + Step-1
##############################

# ----- Load data -----
product_original = CSV.read(joinpath(@__DIR__, "..", "data", "product_data.csv"), DataFrame)
agent_original   = CSV.read(joinpath(@__DIR__, "..", "data", "agent_data.csv"),   DataFrame)

# Outside share per market
outside = combine(groupby(product_original, :market_ids),
                  :shares => (s -> 1 .- sum(s)) => :share_outside)

df_product = leftjoin(product_original, outside, on = :market_ids)
df_product = select(df_product, Not(1))  # drop index col if present
df_agents  = select(agent_original,  Not(1))

# ----- Parameter container -----
struct Params
    β0_0::Float64        # linear mean intercept (used later in IV)
    α0::Float64          # linear mean price coeff (used later in IV)
    β0_inc::Float64      # intercept × income
    β_sugar_inc::Float64 # sugar × income
    α_inc::Float64       # price × income
    α_ν::Float64         # st.dev. of random price coeff (>0)
end

# Example starting values
θ = Params(0.0, -5.0, 0.2, 0.1, -0.3, 0.75)

# Build Γ and α_ν from θ (order of characteristics: [1, sugar, price])
Γ   = [θ.β0_inc; θ.β_sugar_inc; θ.α_inc]  # 3×1
α_ν = θ.α_ν

# Initial δ guess: log share ratio
δ_0 = log.(df_product.shares) .- log.(df_product.share_outside)

# ===============================
# Step-1: predicted shares s̃(δ; Γ, α_ν)
# ===============================

"""
Simulated BLP shares for ONE market (vectorized, numerically stable).

Inputs:
- sugar, price, δ  :: AbstractVector{<:Real} (length J)
- Γ                 :: Vector{<:Real}  [β0_inc, β_sugar_inc, α_inc]
- α_ν               :: Real            st.dev. of random price coeff
- income, ν, w      :: AbstractVector{<:Real} (length I, e.g., 20)

Return: Vector{Float64} of length J with market shares.
"""
function blp_shares_one_market(
    sugar::AbstractVector, price::AbstractVector, δ::AbstractVector,
    Γ::AbstractVector, α_ν::Real,
    income::AbstractVector, ν::AbstractVector, w::AbstractVector
)
    J = length(δ);  I = length(income)

    # J×3 design: [1  sugar  price]
    X = hcat(ones(J), sugar, price)                  # J×3

    # 3×I loadings: Γ*income' + [0,0,α_ν]*ν'
    ΓD = Γ .* permutedims(income)                    # 3×I
    Σν = [0.0, 0.0, α_ν] .* permutedims(ν)           # 3×I
    B  = ΓD + Σν                                     # 3×I

    # Utilities for all products × agents (outside u=0)
    U = δ .* ones(1, I) .+ X * B                     # J×I

    # Stable softmax with outside option
    m     = maximum(U; dims=1)                       # 1×I
    num   = exp.(U .- m)                             # J×I
    denom = exp.(-m) .+ sum(num; dims=1)             # 1×I
    P     = num ./ denom                             # J×I

    return vec(P * w)                                # J×1 → Vector
end

"""
Step-1 for ALL markets. Returns ŝ with same row order as df_product.
"""
function step1_predicted_shares(df_product::DataFrame, df_agents::DataFrame,
                                δ::AbstractVector{<:Real},
                                Γ::AbstractVector{<:Real}, α_ν::Real)
    @assert length(δ) == nrow(df_product)
    s_sim = similar(df_product.shares)

    for subP in groupby(df_product, :market_ids)
        idx   = parentindices(subP)[1]
        δmkt  = δ[idx]
        sugar = subP.sugar
        price = subP.prices

        A = df_agents[df_agents.market_ids .== subP.market_ids[1], :]
        s_sim[idx] = blp_shares_one_market(
            sugar, price, δmkt,
            Γ, α_ν,
            A.income, A.nodes0, A.weights
        )
    end
    return s_sim
end

# ---- quick smoke test: compute ŝ at the initial δ_0 ----
ŝ = step1_predicted_shares(df_product, df_agents, δ_0, Γ, α_ν)
@show minimum(ŝ) maximum(ŝ)  # should lie in (0,1)

##############################
# BLP — Piece 2: δ inversion
##############################

const εshare = eps(Float64)  # tiny guard for logs

"""
Invert δ via Berry's contraction for all markets.

Inputs:
- df_product, df_agents : DataFrames
- δ0  :: Vector{<:Real}         starting guess (row order of df_product)
- Γ   :: Vector{<:Real}         [β0_inc, β_sugar_inc, α_inc]
- α_ν :: Real                   st.dev. of random price RC

Keyword args:
- tol     :: Real  stopping tolerance on ||δ^{r+1} - δ^{r}||_∞  (default 1e-12)
- maxit   :: Int   maximum iterations (default 20_000)
- damping :: Real  optional damping in (0,1] (default 1.0)
- verbose :: Bool  print progress (default false)

Returns: δ̂ :: Vector{Float64}
"""
function invert_delta!(df_product::DataFrame, df_agents::DataFrame,
                       δ0::AbstractVector{<:Real},
                       Γ::AbstractVector{<:Real}, α_ν::Real;
                       tol::Real = 1e-12, maxit::Int = 20_000,
                       damping::Real = 1.0, verbose::Bool = false)

    @assert length(δ0) == nrow(df_product)
    δ     = copy(float.(δ0))
    s_obs = df_product.shares

    for r in 1:maxit
        # Step 1: predicted shares at current δ
        s_hat = step1_predicted_shares(df_product, df_agents, δ, Γ, α_ν)

        # Contraction update
        δ_new = δ .+ damping .* (log.(s_obs .+ εshare) .- log.(s_hat .+ εshare))

        # Convergence check (∞-norm)
        gap = maximum(abs.(δ_new .- δ))
        if verbose && (r == 1 || r % 50 == 0)
            @info "δ-contraction iter $r   ||Δ||∞ = $gap"
        end
        if gap < tol
            return δ_new
        end

        # Basic sanity
        if any(!isfinite, δ_new)
            error("Non-finite δ at iter $r — try smaller damping or different start.")
        end

        δ .= δ_new
    end
    @warn "δ inversion hit maxit without meeting tol."
    return δ
end

"""
Convenience wrapper that calls invert_delta! with your δ_0.
"""
compute_delta(df_product, df_agents, δ_start, Γ, α_ν; kwargs...) =
    invert_delta!(df_product, df_agents, δ_start, Γ, α_ν; kwargs...)


δ_hat = compute_delta(df_product, df_agents, δ_0, Γ, α_ν; tol=1e-12, damping=1.0, verbose=true)
# check that simulated shares now match observed shares closely
s_hat = step1_predicted_shares(df_product, df_agents, δ_hat, Γ, α_ν)
@show maximum(abs.(log.(df_product.shares) .- log.(s_hat)))  # should be < ~1e-12

##############################
# BLP — Piece 3: Linear IV step
##############################

# Build X (mean-utility regressors) and Z (instruments)
# X = [1, price];  Z = [sugar, demand_instruments0:19]
function build_XZ(df::DataFrame)
    N = nrow(df)
    X = hcat(ones(N), df.prices)                         # N×2

    # grab all columns that start with "demand_instruments"
    instr_cols = filter(n -> occursin("demand_instruments", String(n)), names(df))
    Z = hcat(df.sugar, Matrix(select(df, instr_cols)))   # N×(1+20)

    return (Matrix{Float64}(X), Matrix{Float64}(Z))
end

# 2SLS / one-step GMM: θ̂ = (X'Z W Z'X)^(-1) X'Z W Z'δ  with  W=(Z'Z)^(-1)
# Returns residuals ξ̂ = δ - Xθ̂  and θ̂ (vector length 2)
# X = [1, price]; Z = [sugar, demand_instruments0:19]
function build_XZ(df::DataFrame)
    N = nrow(df)
    X = hcat(ones(N), df.prices) |> Matrix{Float64}          # N×2
    instr_cols = filter(c -> startswith(String(c), "demand_instruments"),
                        names(df))
    Z = hcat(df.sugar, Matrix(df[:, instr_cols])) |> Matrix{Float64}  # N×21
    return X, Z
end

# 2SLS / one-step GMM with W = (Z'Z)^(-1)
function compute_xi(df_product::DataFrame, δ::AbstractVector{<:Real})
    @assert length(δ) == nrow(df_product)
    X, Z = build_XZ(df_product)

    ZZ = Z' * Z                     # 21×21
    ZX = Z' * X                     # 21×2
    XZ = X' * Z                     # 2×21
    Zδ = Z' * δ                     # 21×1   ← note the transpose!

    W  = ZZ \ I                     # = inv(ZZ), but stable

    A  = (XZ * W * ZX)              # 2×2
    b  = (XZ * W * Zδ)              # 2×1
    θ1 = A \ b                      # [β0_0, α0]

    ξ  = δ .- X * θ1
    return ξ, θ1
end

ξ̂, θ1̂ = compute_xi(df_product, δ_hat)   # use the δ from your inner loop
println("θ1 (linear) = [β0_0, α0] = ", vec(θ1̂))
println("mean|ξ| = ", mean(abs.(ξ̂)), "   max|ξ| = ", maximum(abs.(ξ̂)))

##############################
# BLP — Piece 4: GMM objective + optimizer
##############################


# Precompute X, Z and a default weight W = (Z'Z)^(-1)
const X_global, Z_global = build_XZ(df_product)
const Nobs = nrow(df_product)
const W1 = inv(Z_global' * Z_global)     # one-step GMM weight

# Map θ2 ↔ (Γ, α_ν).  We parameterize α_ν = exp(ρ) to keep it > 0.
Γ_from(θ2_raw)   = @view θ2_raw[1:3]               # [β0_inc, β_sugar_inc, α_inc]
α_ν_from(θ2_raw) = exp(θ2_raw[4])                  # α_ν > 0

"""
Compute GMM criterion Q(θ2_raw) with given weight matrix W.

θ2_raw = [β0_inc, β_sugar_inc, α_inc, ρ] where α_ν = exp(ρ).
δ_start is a warm start for the contraction (use δ_0 or last δ̂).
"""
function gmm_objective(θ2_raw::AbstractVector{<:Real};
                       δ_start::AbstractVector{<:Real}=δ_0,
                       W::AbstractMatrix{<:Real}=W1,
                       tol::Real=1e-12, maxit::Int=20_000, damping::Real=1.0)

    Γ   = collect(Γ_from(θ2_raw))
    α_ν = α_ν_from(θ2_raw)

    # inner loop: invert δ for (Γ, α_ν)
    δ̂ = compute_delta(df_product, df_agents, δ_start, Γ, α_ν;
                       tol=tol, maxit=maxit, damping=damping, verbose=false)

    # linear IV: residuals ξ̂ = δ̂ - X θ1̂
    ξ̂, _ = compute_xi(df_product, δ̂)

    # moments & criterion
    g = (Z_global' * ξ̂) / Nobs
    return dot(g, W * g)         # scalar Q
end

"""
One-step GMM estimation of θ2, followed by optional two-step reweighting.

Returns a NamedTuple with θ2_hat, Γ_hat, α_ν_hat, θ1_hat, δ_hat, ξ_hat, Q, and (optionally) Q2.
"""
using Optim

function estimate_blp!(θ2_start_raw::AbstractVector{<:Real};
                       two_step::Bool=true, damping::Real=1.0, tol::Real=1e-12)

    # ---- First step: one-step GMM with W1 ----
    opt1 = optimize(
        θ -> gmm_objective(θ; δ_start=δ_0, W=W1, tol=tol, damping=damping),
        θ2_start_raw,
        NelderMead(),
        Optim.Options(iterations=2000, store_trace=true, show_trace=true)
    )
    θ2_hat_raw = Optim.minimizer(opt1)

    # Recover objects at the first-step optimum
    Γ_hat   = collect(Γ_from(θ2_hat_raw))
    α_ν_hat = α_ν_from(θ2_hat_raw)
    δ_hat   = compute_delta(df_product, df_agents, δ_0, Γ_hat, α_ν_hat; tol=tol, damping=damping)
    ξ_hat, θ1_hat = compute_xi(df_product, δ_hat)
    Q1 = gmm_objective(θ2_hat_raw; δ_start=δ_hat, W=W1, tol=tol, damping=damping)

    if !two_step
        return (θ2_hat_raw=θ2_hat_raw, Γ_hat=Γ_hat, α_ν_hat=α_ν_hat,
                θ1_hat=θ1_hat, δ_hat=δ_hat, ξ_hat=ξ_hat, Q=Q1)
    end

    # ---- Second step: optimal weight W2 = Ω^{-1} using ξ̂ ----
    Ω  = (Z_global' * Diagonal(vec(ξ_hat .^ 2)) * Z_global) / Nobs
    W2 = inv(Ω)

    opt2 = optimize(
        θ -> gmm_objective(θ; δ_start=δ_hat, W=W2, tol=tol, damping=damping),
        θ2_hat_raw,
        NelderMead(),
        Optim.Options(iterations=2000, store_trace=true, show_trace=true)
    )
    θ2_hat2_raw = Optim.minimizer(opt2)

    # Final objects at two-step optimum
    Γ_hat2   = collect(Γ_from(θ2_hat2_raw))
    α_ν_hat2 = α_ν_from(θ2_hat2_raw)
    δ_hat2   = compute_delta(df_product, df_agents, δ_hat, Γ_hat2, α_ν_hat2; tol=tol, damping=damping)
    ξ_hat2, θ1_hat2 = compute_xi(df_product, δ_hat2)
    Q2 = gmm_objective(θ2_hat2_raw; δ_start=δ_hat2, W=W2, tol=tol, damping=damping)

    return (θ2_hat_raw=θ2_hat2_raw, Γ_hat=Γ_hat2, α_ν_hat=α_ν_hat2,
            θ1_hat=θ1_hat2, δ_hat=δ_hat2, ξ_hat=ξ_hat2, Q=Q2, W2=W2)
end

θ2_start_raw = [θ.β0_inc, θ.β_sugar_inc, θ.α_inc, log(θ.α_ν)]
results = estimate_blp!(θ2_start_raw; two_step=true, damping=1.0, tol=1e-12)

println("\n==== BLP results ====")
println("θ1 (linear) [β0_0, α0] = ", vec(results.θ1_hat))
println("Γ [β0_inc, β_sugar_inc, α_inc] = ", results.Γ_hat)
println("α_ν (st.dev. price RC) = ", results.α_ν_hat)
println("Q = ", results.Q)

##############################
# BLP — Piece 5: Elasticities
##############################

# Compute probabilities for ONE market at (δ, Γ, α_ν) and params θ1=(β0_0, α0)
# Returns: P (J×I), s (J-vector), α_i (I-vector)
function probs_one_market(subP::DataFrame, A::DataFrame,
                          δmkt::AbstractVector, Γ::AbstractVector, α_ν::Real)
    J = nrow(subP); I = nrow(A)
    sugar = subP.sugar; price = subP.prices
    X = hcat(ones(J), sugar, price)     # J×3

    income = A.income; ν = A.nodes0; w = A.weights
    ΓD = Γ .* permutedims(income)       # 3×I
    Σν = [0.0, 0.0, α_ν] .* permutedims(ν)
    B  = ΓD + Σν                        # 3×I

    U = δmkt .* ones(1, I) .+ X * B     # J×I
    m = maximum(U; dims=1)
    num = exp.(U .- m)
    denom = exp.(-m) .+ sum(num; dims=1)
    P = num ./ denom                    # J×I
    s = vec(P * w)                      # J×1
    αi = (@view B[3, :])                # price loading per agent (I)
    return P, s, αi, w
end

"""
Elasticity matrix for ONE market at the estimated parameters.

Inputs:
- market_id::AbstractString
- df_product, df_agents : DataFrames
- δ_hat :: Vector (stacked)
- Γ_hat :: Vector{Float64} [β0_inc, β_sugar_inc, α_inc]
- α_ν_hat :: Real
Returns:
- E :: J×J matrix (ε_{jk})
"""
function elasticities_market(market_id::AbstractString,
                             df_product::DataFrame, df_agents::DataFrame,
                             δ_hat::AbstractVector,
                             Γ_hat::AbstractVector, α_ν_hat::Real)

    subP = filter(:market_ids => ==(market_id), df_product)
    idx  = parentindices(groupby(df_product, :market_ids)
           |> (g -> first(filter(x -> x.market_ids[1] == market_id, g))).parent)[1]
    δmkt = δ_hat[idx]
    P, s, αi, w = probs_one_market(subP, filter(:market_ids => ==(market_id), df_agents),
                                   δmkt, Γ_hat, α_ν_hat)

    J, I = size(P)
    # Own and cross derivatives (heterogeneous logit):
    # ∂s_j/∂p_j = Σ_i w_i * α_i * P_{ij} * (1 - P_{ij})
    # ∂s_k/∂p_j = - Σ_i w_i * α_i * P_{ik} * P_{ij}   (k ≠ j)
    dsdpj = zeros(J, J)
    for j in 1:J
        # own
        dsdpj[j, j] = sum(@. w * αi * P[j, :] * (1 - P[j, :]))
        # cross
        for k in 1:J
            if k != j
                dsdpj[k, j] = -sum(@. w * αi * P[k, :] * P[j, :])
            end
        end
    end
    p = subP.prices
    E = zeros(J, J)
    for j in 1:J, k in 1:J
        E[j, k] = dsdpj[j, k] * (p[k] / s[j])   # ε_{jk} = ∂s_j/∂p_k * (p_k / s_j)
    end
    return E, s, p
end

# --- example usage (replace market id if needed) ---
# E, s_mkt, p_mkt = elasticities_market("C01Q1", df_product, df_agents, results.δ_hat, results.Γ_hat, results.α_ν_hat)
# println("Own-price elasticities (diag): ", diag(E))

# 1) Fit diagnostic: simulated vs observed shares at optimum
s_fit = step1_predicted_shares(df_product, df_agents, results.δ_hat, results.Γ_hat, results.α_ν_hat)
println("max |log share error| = ",
        maximum(abs.(log.(df_product.shares) .- log.(s_fit))))

# 2) Price coefficient by income (sanity on α_inc scale)
using Statistics
println("income: mean=", mean(df_agents.income), "  sd=", std(df_agents.income))
# Effective α_i range (10th–90th percentile)
using StatsBase
q10, q90 = quantile(df_agents.income, [0.1, 0.9])
α10 = results.θ1_hat[2] + results.Γ_hat[3]*q10      # α0 + α_inc*income
α90 = results.θ1_hat[2] + results.Γ_hat[3]*q90
println("α_i at income p10≈$q10 : $α10   | p90≈$q90 : $α90")

s_fit = step1_predicted_shares(df_product, df_agents,
                               results.δ_hat, results.Γ_hat, results.α_ν_hat)
println("max |log share error| = ",
        maximum(abs.(log.(df_product.shares) .- log.(s_fit))))

        using Statistics
println("income: mean=", mean(df_agents.income), "  sd=", std(df_agents.income))

# effective price coeff at income quantiles
using StatsBase
q10, q90 = quantile(df_agents.income, [0.10, 0.90])
α0   = results.θ1_hat[2]
αinc = results.Γ_hat[3]
println("α_i at p10 ≈ ", q10, "  → ", α0 + αinc*q10)
println("α_i at p90 ≈ ", q90, "  → ", α0 + αinc*q90)

# probabilities + shares helper (unchanged)
function probs_one_market(subP::DataFrame, A::DataFrame,
                          δmkt::AbstractVector, Γ::AbstractVector, α_ν::Real)
    J = nrow(subP); I = nrow(A)
    sugar = subP.sugar; price = subP.prices
    X = hcat(ones(J), sugar, price)

    income = A.income; ν = A.nodes0; w = A.weights
    ΓD = Γ .* permutedims(income)
    Σν = [0.0, 0.0, α_ν] .* permutedims(ν)
    B  = ΓD + Σν

    U = δmkt .* ones(1, I) .+ X * B
    m = maximum(U; dims=1)
    num = exp.(U .- m)
    denom = exp.(-m) .+ sum(num; dims=1)
    P = num ./ denom
    s = vec(P * w)
    αi = @view B[3, :]
    return P, s, αi, w
end

function elasticities_market(market_id::AbstractString,
                             df_product::DataFrame, df_agents::DataFrame,
                             δ_hat::AbstractVector,
                             Γ_hat::AbstractVector, α_ν_hat::Real)
    idx  = findall(df_product.market_ids .== market_id)
    subP = df_product[idx, :]
    δmkt = δ_hat[idx]
    A    = df_agents[df_agents.market_ids .== market_id, :]

    P, s, αi, w = probs_one_market(subP, A, δmkt, Γ_hat, α_ν_hat)

    J, I = size(P)
    dsdp = zeros(J, J)
    for j in 1:J
        dsdp[j,j] = sum(@. w * αi * P[j,:] * (1 - P[j,:]))
        for k in 1:J
            if k != j
                dsdp[k,j] = -sum(@. w * αi * P[k,:] * P[j,:])
            end
        end
    end
    p = subP.prices
    E = [ dsdp[j,k] * (p[k] / s[j]) for j in 1:J, k in 1:J ]
    return E, s, p
end

E, s_mkt, p_mkt = elasticities_market("C01Q1",
                                      df_product, df_agents,
                                      results.δ_hat, results.Γ_hat, results.α_ν_hat)

println("Own-price elasticities (diag): ", diag(E))

# ====== Part (b): own-price elasticities scatter for market "C01Q1" ======
using Plots, DataFrames

# -- helper uses total price coeff per agent (α0 + α_inc*income + α_ν*ν)
function probs_one_market(subP::DataFrame, A::DataFrame,
                          δmkt::AbstractVector, Γ::AbstractVector, α_ν::Real,
                          α0::Real)
    J = nrow(subP); I = nrow(A)
    sugar = subP.sugar; price = subP.prices
    X = hcat(ones(J), sugar, price)

    income = A.income; ν = A.nodes0; w = A.weights
    ΓD = Γ .* permutedims(income)             # 3×I
    Σν = [0.0, 0.0, α_ν] .* permutedims(ν)    # 3×I
    B  = ΓD + Σν                               # heterogeneity part

    U = δmkt .* ones(1, I) .+ X * B
    m = maximum(U; dims=1)
    num = exp.(U .- m)
    denom = exp.(-m) .+ sum(num; dims=1)
    P = num ./ denom
    s = vec(P * w)

    αi_tot = α0 .+ @view(B[3, :])             # total price coeff per agent
    return P, s, αi_tot, w
end

function elasticities_market(market_id::AbstractString,
                             df_product::DataFrame, df_agents::DataFrame,
                             δ_hat::AbstractVector,
                             Γ_hat::AbstractVector, α_ν_hat::Real,
                             θ1_hat::AbstractVector)
    α0 = θ1_hat[2]
    idx  = findall(df_product.market_ids .== market_id)
    subP = df_product[idx, :]
    δmkt = δ_hat[idx]
    A    = df_agents[df_agents.market_ids .== market_id, :]

    P, s, αi, w = probs_one_market(subP, A, δmkt, Γ_hat, α_ν_hat, α0)

    J, I = size(P)
    dsdp = zeros(J, J)
    for j in 1:J
        dsdp[j,j] = sum(@. w * αi * P[j,:] * (1 - P[j,:]))               # own
        for k in 1:J
            if k != j
                dsdp[k,j] = -sum(@. w * αi * P[k,:] * P[j,:])            # cross
            end
        end
    end
    p = subP.prices
    E = [ dsdp[j,k] * (p[k] / s[j]) for j in 1:J, k in 1:J ]             # ε_{jk}
    return E, s, p, subP.product_ids
end

# --- compute & plot for C01Q1 ---
E, s_mkt, p_mkt, pid_mkt = elasticities_market("C01Q1",
    df_product, df_agents, results.δ_hat, results.Γ_hat, results.α_ν_hat, results.θ1_hat)

own = diag(E)
df_plot = DataFrame(product_id = pid_mkt, price = p_mkt, own_elast = own)

# scatter
scatter(df_plot.price, df_plot.own_elast;
        xlabel = "Price", ylabel = "Own-price elasticity",
        legend = false, title = "Own-price elasticities vs Price — market C01Q1",
        markershape = :circle, markersize = 5)
annotate!.(df_plot.price, df_plot.own_elast, Ref((" ", 8)))  # (optional) labels off for cleanliness

# Save if you like:
# png("reports/figs/own_elasticities_C01Q1.png")

# quick sanity print
println("count(own ε > 0) = ", sum(df_plot.own_elast .> 0), "  of  ", nrow(df_plot))
first(df_plot, 5)

end
