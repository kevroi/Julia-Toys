# Based on the implementation in the ReinforcementLearningEnvironments.jl package

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
        T=Float64,
        min_pos=-1.2,
        max_pos=0.6,
        max_speed=0.07,
        goal_pos=0.5,
        goal_velocity=0.0,
        power=0.001,
        gravity=0.0025,
        max_steps=200
    )
        P = new(
            min_pos,
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
                zeros(T, 2),
                0,
                false, 0)

        return E
    end

end


function step(env::MountainCarEnv, force)
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