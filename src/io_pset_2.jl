module io_pset_2
# Installing packages
using CSV, DataFrames, GLM, LinearAlgebra,Chain, DataFramesMeta
agent_original   = CSV.read(joinpath(@__DIR__, "..", "data", "agent_data.csv"), DataFrame)
describe(agent_original)
product_original   = CSV.read(joinpath(@__DIR__, "..", "data", "product_data.csv"), DataFrame)
describe(product_original)

outside = @chain product_original begin
    groupby(:market_ids)
    combine(:shares => (s -> 1 - sum(s)) => :share_outside)
end

df_product = leftjoin(product_original, outside, on = :market_ids)
df_product = select(df_product, Not(1))
df_agents = agent_original
df_agents = select(df_agents, Not(1))
describe(df_product)

# === parameter container (linear & nonlinear) ===
struct Params
    β0_0::Float64          # linear: mean intercept (enters δ)
    α0::Float64            # linear: mean price coeff (enters δ)
    β0_inc::Float64        # nonlinear: intercept × income
    β_sugar_inc::Float64   # nonlinear: sugar × income
    α_inc::Float64         # nonlinear: price × income
    α_ν::Float64           # nonlinear: stdev for price RC
end

# example starting values (tune as you like)
θ = Params(0.0, -5.0, 0.2, 0.1, -0.3, 0.75)

# === build Γ and Σ given θ, using order [1, sugar, price] ===
Γ = [θ.β0_inc; θ.β_sugar_inc; θ.α_inc]           # rows: 1, sugar, price
Σ = Diagonal([0.0, 0.0, θ.α_ν])
δ_0 = log.(df_product.shares).-log.(df_product.share_outside)  # initial guess for δ

# === simulated shares given δ and θ ===
function simulate_shares(df_product::DataFrame, df_agents::DataFrame,
                         θ::Params, δ::AbstractVector{<:Real})
    @assert length(δ) == nrow(df_product)
    s_sim = similar(df_product.shares)  # same length, Float64

    for subP in groupby(df_product, :market_ids)
        # rows of this market in the parent df
        ridx = parentindices(subP)[1]         # row indices in df_product
        J    = nrow(subP)

        # agents for this market
        agents = @subset(df_agents, :market_ids .== subP.market_ids[1])
        income = agents.income
        ν      = agents.nodes0
        w      = agents.weights

        # precompute agent-specific terms
        base_i   = θ.β0_inc .* income                         # intercept × inc
        sugar_i  = θ.β_sugar_inc .* income                    # sugar × inc (agent factor)
        aprice_i = θ.α_inc .* income .+ θ.α_ν .* ν            # price × (inc+rand)

        # product vectors
        sugar = subP.sugar
        price = subP.prices
        δmkt  = δ[ridx]

        # accumulate market shares over agents
        share_j = zeros(Float64, J)
        for i in eachindex(income)
            # utilities for all products for agent i (outside good u=0)
            u = δmkt .+ base_i[i] .+ sugar .* sugar_i[i] .+ price .* aprice_i[i]

            # numerically stable softmax with outside option
            m = max(0.0, maximum(u))
            denom = exp(-m) + sum(@. exp(u - m))
            p = @. exp(u - m) / denom

            share_j .+= w[i] .* p
        end
        s_sim[ridx] = share_j
    end
    return s_sim
end

end
