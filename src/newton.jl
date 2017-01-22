immutable Newton{L<:Function} <: Optimizer
    linesearch!::L
    resetalpha::Bool
end
#= uncomment v0.8.0
Newton(; linesearch::Function = LineSearches.hagerzhang!) =
Newton(linesearch)
=#
function Newton(; linesearch! = nothing, linesearch::Function = LineSearches.hagerzhang!,
                resetalpha = true)
    linesearch = get_linesearch(linesearch!, linesearch)
    Newton(linesearch,resetalpha)
end

type NewtonState{T}
    @add_generic_fields()
    x_previous::Array{T}
    f_x_previous::T
    F
    Hd
    s::Array{T}
    @add_linesearch_fields()
end

function initial_state{T}(method::Newton, options, d, initial_x::Array{T})
    n = length(initial_x)
    # Maintain current gradient in gr
    s = similar(initial_x)
    x_ls, g_ls = similar(initial_x), similar(initial_x)
    f_x_previous = NaN
    value_grad!(d, initial_x)
    hessian!(d, initial_x)
    NewtonState("Newton's Method",
              length(initial_x),
              copy(initial_x), # Maintain current state in state.x
              copy(initial_x), # Maintain current state in state.x_previous
              T(NaN), # Store previous f in state.f_x_previous
              copy(hessian(d)),
              copy(hessian(d)),
              similar(initial_x), # Maintain current search direction in state.s
              @initial_linesearch()...) # Maintain a cache for line search results in state.lsr
end

function update_state!{T}(d, state::NewtonState{T}, method::Newton)
    lssuccess = true
    # Search direction is always the negative gradient divided by
    # a matrix encoding the absolute values of the curvatures
    # represented by H. It deviates from the usual "add a scaled
    # identity matrix" version of the modified Newton method. More
    # information can be found in the discussion at issue #153.
    state.F, state.Hd = ldltfact!(Positive, hessian(d))
    state.s[:] = -(state.F\grad(d))

    # Refresh the line search cache
    dphi0 = vecdot(grad(d), state.s)
    LineSearches.clear!(state.lsr)
    push!(state.lsr, zero(T), value(d), dphi0)

    # Determine the distance of movement along the search line
    lssuccess = do_linesearch(state, method, d)

    # Maintain a record of previous position
    copy!(state.x_previous, state.x)

    # Update current position # x = x + alpha * s
    LinAlg.axpy!(state.alpha, state.s, state.x)
    (lssuccess == false) # break on linesearch error
end
