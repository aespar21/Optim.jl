log_temperature(t::Real) = 1 / log(t)

constant_temperature(t::Real) = 1.0

function default_neighbor!(x::Array, x_proposal::Array)
    @assert size(x) == size(x_proposal)
    for i in 1:length(x)
        @inbounds x_proposal[i] = x[i] + randn()
    end
    return
end

immutable SimulatedAnnealing <: Optimizer
    neighbor!::Function
    temperature::Function
    keep_best::Bool # not used!?
end

function SimulatedAnnealing(; neighbor! = nothing,
                     neighbor = default_neighbor!,
                     temperature = log_temperature,
                     keep_best::Bool = true)
    neighbor = get_neighbor(neighbor!, neighbor)
    SimulatedAnnealing(neighbor, temperature, keep_best)
end
type SimulatedAnnealingState{T}
    @add_generic_fields()
    iteration::Int
    x_current::Array{T}
    x_proposal::Array{T}
    f_x_current::T
    f_proposal::T
end

function initial_state{T}(method::SimulatedAnnealing, options, f, initial_x::Array{T})
    # Count number of parameters
    n = length(initial_x)
    f.f_x = value(f, initial_x)
    SimulatedAnnealingState("Simulated Annealing", n, copy(initial_x), 1, copy(initial_x), copy(initial_x), f.f_x, f.f_x)
end

function update_state!(f, state::SimulatedAnnealingState, method::SimulatedAnnealing)

    # Determine the temperature for current iteration
    t = method.temperature(state.iteration)

    # Randomly generate a neighbor of our current state
    method.neighbor!(state.x_current, state.x_proposal)

    # Evaluate the cost function at the proposed state
    state.f_proposal = value(f, state.x_proposal)
    if state.f_proposal <= state.f_x_current
        # If proposal is superior, we always move to it
        copy!(state.x_current, state.x_proposal)
        state.f_x_current = state.f_proposal

        # If the new state is the best state yet, keep a record of it
        if state.f_proposal < f.f_x
            f.f_x = state.f_proposal
            copy!(state.x, state.x_proposal)
        end
    else
        # If proposal is inferior, we move to it with probability p
        p = exp(-(state.f_proposal - state.f_x_current) / t)
        if rand() <= p
            copy!(state.x_current, state.x_proposal)
            state.f_x_current = state.f_proposal
        end
    end
    state.iteration += 1
    false
end
