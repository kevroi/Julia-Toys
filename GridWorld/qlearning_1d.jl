using Random
using POMDPs, POMDPModelTools, QuickPOMDPs, DiscreteValueIteration, TabularTDLearning, POMDPPolicies

# State data type
struct State
    x::Int
end

# State space
null = State(-1)
S = [[State(x) for x in 1:7]..., null]

# Action data type
@enum Action LEFT RIGHT
# Action space
A = [LEFT, RIGHT]

const Movements = Dict(
                        LEFT => State(-1),
                        RIGHT => State(1)
                        )

Base.:+(s1::State, s2::State) = State(s1.x + s2.x)

# Transition dynamics
function P(s::State, a::Action)
    if R(s) != 0
        return Deterministic(null)
    end

    dim_A = length(A)
    next_states = Vector{State}(undef, dim_A+1)
    probs = zeros(dim_A+1)

    for (index, a_prime) in enumerate(A)
        prob = (a_prime==a) ? 0.8 : 0.2
        dest = s + Movements[a_prime]
        next_states[index+1] = dest

        if 1<= dest.x <= 7
            probs[index+1] += prob
        end
    end

    next_states[1] = s
    probs[1] = 1 - sum(probs)

    return SparseCat(next_states, probs)
end

# Reward function
function R(s, a = missing)
    if s == State(1)
        return -1
    elseif s == State(7)
        return 1
    end
    return 0
end

gamma = 0.99

terminated(s::State) = s == null

abstract type GridWorld <: MDP{State, Action} end

### Q-learning Algorithm ###
q_mdp = QuickMDP(GridWorld,
                states=S,
                actions=A,
                transition=P,
                reward=R,
                discount=gamma,
                initialstate=S,
                isterminal=terminated
)

Random.seed!(1)
lr = 0.9
n_episodes = 20
q_solver = QLearningSolver(n_episodes=n_episodes,
                        learning_rate=lr,
                        exploration_policy=EpsGreedyPolicy(q_mdp, 0.1),
                        verbose=false,
)
q_policy= solve(q_solver, q_mdp)
