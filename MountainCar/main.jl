# Based on the Pluto Notebook from ReinforcementLearningAnIntroduction.jl package

using Flux
using Statistics
using Plots
using SparseArrays

env = MountainCarEnv()
S = state_space(env)