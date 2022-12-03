#=

|_|_|_|G|
|_|_|_|X|
|_|B|_|_|
|S|_|_|_|

=#




using POMDPs, POMDPModelTools, QuickPOMDPs, DiscreteValueIteration

# custom data type for State
struct State
    x::Int
    y::Int
end

# define a null state and a 4 by 4 GridWorld state space (vector of 17 states)
null = State(-1, -1)
S = [
    [State(x,y) for x=1:4, y=1:4]..., null
    ]


# create an enumerate data type for Action (directions along with their index)
@enum Action UP DOWN LEFT RIGHT

# define action space using 4 instances of this data type
A = [UP, DOWN, LEFT, RIGHT]

# create dictionary encoding the effect of each action
const Movements = Dict(
                        UP => State(0,1),
                        DOWN => State(0,-1),
                        LEFT => State(-1,0),
                        RIGHT => State(1,0)
                        )

# Add a new method to + operator to add a movement to a current State
Base.:+(s1::State, s2::State) = State(s1.x+s2.x, s1.y+s2.y)

# define a transition function for this GridWorld
function P(s::State, a::Action)
    # end the game when the agent lands on a tile with non-zero reward
    if R(s) != 0
        return Deterministic(null)
    end

    dim_A = length(A)
    next_states = Vector{State}(undef, dim_A+1) # 5 element vector: current and neighbour states
    probs = zeros(dim_A+1)

    # transition dynamics
    for (index, a_prime) in enumerate(A)
        prob = (a_prime == a) ? 0.7 : 0.1 # 70% chance of moving in the direction a (the argument to P(s,a))
        dest = s + Movements[a_prime]
        next_states[index+1] = dest

        if dest.x==1 && dest.y==1
            probs[index+1] = 0
        elseif 1 <= dest.x <= 4 && 1 <= dest.y <= 4
            probs[index+1] += prob
        end
    end

    #reflecting boundaries
    next_states[1] = s
    probs[1] = 1 - sum(probs)

    # return sparse categorical dist 
    return SparseCat(next_states, probs)
end

# define Reward function
function R(s, a=missing)
    if s == State(4,4)
        return 10
    elseif s == State(4,3)
        return -100
    end
    return 0
end

gamma = 0.99

# tells us whether is terminated
terminated(s::State) = s==null

# custom data type for env
abstract type GridWorld <: MDP{State, Action} end

# create MDP using Quick MDP constructor
mdp = QuickMDP(
                GridWorld,
                states = S,
                actions = A,
                transition = P,
                reward = R,
                discount = gamma,
                isterminal = terminated
                )

# Solve MDP for this small env using Value Iteration Algorithm (Iterative sol to Bellman Equation)
solver = ValueIterationSolver(max_iterations=30)
policy = solve(solver, mdp)
value_view = [S policy.util]