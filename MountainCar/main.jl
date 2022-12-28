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

# cost to go
plot(X, Y, -show_approximation(n), linetype=:wireframe,
	xlabel="Position", ylabel="Velocity", zlabel="cost-to-go", title="Episode $n")
savefig("costtogo.png")

# steps per episode
fig_10_2 = plot(legend=:topright, xlabel="Episode", ylabel="Avg. steps per episode")
n_runs = 5  # quite slow here, need revisit
for α in [0.1/8, 0.2/8, 0.5/8]
    avg_steps_per_episode = zeros(501)
    for _ in 1:n_runs
        local env, agent = create_env_agent(α)
        hook = StepsPerEpisode()
        run(agent, env, StopAfterEpisode(500; is_show_progress=false),hook)
        avg_steps_per_episode .+= hook.steps
    end
    plot!(fig_10_2, avg_steps_per_episode[1:end-1] ./ n_runs, yscale=:log10, label="α=$α")
end
savefig("steps_ep.png")