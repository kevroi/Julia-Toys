# Based on the implementation in the ReinforcementLearningEnvironments.jl package
export MountainCarEnv

struct MountainCarEnvParams
    min_pos::float
    max_pos::float
    max_speed::float
    goal_pos::float
    goal_velocity::float
    power::float
    gravity::float
    max_steps::Int

    # default constructor with params from Sutton & Barto
    function MountainCarEnvParams(
        min_pos=-1.2,
        max_pos=0.6,
        max_speed=0.07,
        goal_pos=0.5,
        goal_velocity=0.0,
        power=0.001,
        gravity=0.0025,
        max_steps=200
    )
        P = new(min_pos,
                max_pos,
                max_speed,
                goal_pos,
                goal_velocity,
                power,
                gravity,
                max_steps,
                )
        return P
    end
end


mutable struct MountainCarEnv
    params::MountainCarEnvParams
    state::Vector{float} # 2D vector of position and speed
    action::ACT
    done::Bool
    t::Int
    rng::AbstractRNG

    # default values from Sutton & Barto, using a default constructor
    function MountainCarEnv(kwargs...)
        params = MountainCarEnvParams(kwargs...),
        state = zeros(float, 2)
        action = 0,
        done = False,
        t = 0,
        rng = Random.GLOBAL_RNG

        E = new(params,
                state,
                action,
                done,
                t,
                rng
                )

        return E
    end

end


# Define State Space
function state_space(env::MountainCarEnv)
    (env.params.min_pos .. env.params.max_pos) ×
    (-env.params.max_speed .. env.params.max_speed)
end

# Define Action Space: backwards, zero and forward throttle
action_space(::MountainCarEnv) = Base.OneTo(3)
action_space(::MountainCarEnv) = -1.0 .. 1.0

# Reward function
reward(env::MountainCarEnv) = env.done ? 0.0 : -1.0


function (env::MountainCarEnv)(a::Int)
    @assert a in action_space(env)
    env.action = a
    step!(env, a - 2)
end


function step!(env::MountainCarEnv, force)
    env.t += 1
    x, v = env.state
    v += force * env.params.power + cos(3 * x) * (-env.params.gravity)
    v = clamp(v, -env.params.max_speed, env.params.max_speed)
    x += v
    x = clamp(x, env.params.min_pos, env.params.max_pos)
    if x == env.params.min_pos && v < 0
        v = 0
    end
    env.done =
        x >= env.params.goal_pos && v >= env.params.goal_velocity ||
        env.t >= env.params.max_steps
    env.state[1] = x
    env.state[2] = v
    nothing
end


function reset!(env::MountainCarEnv)
    env.state[1] = 0.2 * rand(env.rng, T) - 0.6
    env.state[2] = 0.0
    env.done = false
    env.t = 0
    nothing
end