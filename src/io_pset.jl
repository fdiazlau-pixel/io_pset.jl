module io_pset
 # Installing packages
using CSV, DataFrames, GLM


agent_original   = CSV.read(joinpath(@__DIR__, "..", "data", "agent_data.csv"), DataFrame)
describe(agent_original)
product_original   = CSV.read(joinpath(@__DIR__, "..", "data", "product_data.csv"), DataFrame)
describe(product_original)

agent_original.income

x = randn(100)
y = 0.9 .* x + 0.5 * rand(100)
df = DataFrame(x = x, y = y)
ols = lm(@formula(y~x), df) # R-style notation

df_product = select(product_original, Not(1))

ols_io = lm(@formula(log(shares)~sugar+prices),df_product)



end
