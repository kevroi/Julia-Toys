# Based on the Pluto Notebook from ReinforcementLearningAnIntroduction.jl package

# using Flux
# using Statistics
# using Plots
# using SparseArrays

include("MountainCarEnv.jl")

env = MountainCarEnv()
S = state_space(env)
show(env)

# Tile Coding the state space
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