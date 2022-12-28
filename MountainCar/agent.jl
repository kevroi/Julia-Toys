function create_env_agent(α=2e-4, n=0)
    env = StateTransformedEnv(
        MountainCarEnv(;max_steps=10000),
        state_mapping=s -> sparse(map(t -> encode(t, s), tilings), 1:8, ones(8), 81, 8) |> vec
    )

    agent = Agent(
        policy=QBasedPolicy(
            learner=TDLearner(
                approximator=LinearQApproximator(
                    n_state=81*8,
                    n_action=3,
                    opt = Descent(α)
                    ),
                method=:SARSA,
                n=n
                ),
            explorer=GreedyExplorer()
            ),
        trajectory=VectorSARTTrajectory(;state=Vector{Int})
    )

    env, agent
end


function show_approximation(n)
    env, agent = create_env_agent()
    run(agent, env, StopAfterEpisode(n))
    [
		agent.policy.learner.approximator(env.state_mapping([p, v])) |> maximum
        for p in X, v in Y
	]
end