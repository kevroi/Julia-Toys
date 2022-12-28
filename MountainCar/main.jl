using ReinforcementLearning
using Flux
using Statistics
using Plots
using SparseArrays

include("tiling.jl")
include("agent.jl")

env = MountainCarEnv()
S = state_space(env)

ntilings = 8
ntiles = 8
tiling = Tiling(
    (
        range(r.left, step=(r.right-r.left)/ntiles, length=ntiles+2)
        for r in S
    )...
)
offset = map(x-> x.right - x.left, S) ./ (ntiles * ntilings)
tilings = [tiling - offset .* (i-1) for i in 1:ntilings]

X = range(S[1].left, stop=S[1].right, length=40)
Y = range(S[2].left, stop=S[2].right, length=40)
n = 10

plot(X, Y, -show_approximation(n), linetype=:wireframe,
	xlabel="Position", ylabel="Velocity", zlabel="cost-to-go", title="Episode $n")
