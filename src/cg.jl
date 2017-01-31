#
# Conjugate gradient
#
# This is an independent implementation of:
#   W. W. Hager and H. Zhang (2006) Algorithm 851: CG_DESCENT, a
#     conjugate gradient method with guaranteed descent. ACM
#     Transactions on Mathematical Software 32: 113–137.
#
# Code comments such as "HZ, stage X" or "HZ, eqs Y" are with
# reference to a particular point in this paper.
#
# Several aspects of the following have also been incorporated:
#   W. W. Hager and H. Zhang (2012) The limited memory conjugate
#     gradient method.
#
# This paper will be denoted HZ2012 below.
#
# There are some modifications and/or extensions from what's in the
# paper (these may or may not be extensions of the cg_descent code
# that can be downloaded from Hager's site; his code has undergone
# numerous revisions since publication of the paper):
#
# cgdescent: the termination condition employs a "unit-correct"
#   expression rather than a condition on gradient
#   components---whether this is a good or bad idea will require
#   additional experience, but preliminary evidence seems to suggest
#   that it makes "reasonable" choices over a wider range of problem
#   types.
#
# linesearch: the Wolfe conditions are checked only after alpha is
#   generated either by quadratic interpolation or secant
#   interpolation, not when alpha is generated by bisection or
#   expansion. This increases the likelihood that alpha will be a
#   good approximation of the minimum.
#
# linesearch: In step I2, we multiply by psi2 only if the convexity
#   test failed, not if the function-value test failed. This
#   prevents one from going uphill further when you already know
#   you're already higher than the point at alpha=0.
#
# both: checks for Inf/NaN function values
#
# both: support maximum value of alpha (equivalently, c). This
#   facilitates using these routines for constrained minimization
#   when you can calculate the distance along the path to the
#   disallowed region. (When you can't easily calculate that
#   distance, it can still be handled by returning Inf/NaN for
#   exterior points. It's just more efficient if you know the
#   maximum, because you don't have to test values that won't
#   work.) The maximum should be specified as the largest value for
#   which a finite value will be returned.  See, e.g., limits_box
#   below.  The default value for alphamax is Inf. See alphamaxfunc
#   for cgdescent and alphamax for linesearch_hz.


immutable ConjugateGradient{T, Tprep<:Union{Function, Void}, L<:Function} <: Optimizer
    eta::Float64
    P::T
    precondprep!::Tprep
    linesearch!::L
end
#= uncomment for v0.8.0
function ConjugateGradient(;
                           linesearch = LineSearches.hagerzhang!,
                           eta::Real = 0.4,
                           P::Any = nothing,
                           precondprep = (P, x) -> nothing)
    ConjugateGradient(Float64(eta),
                                 P, precondprep,
                                 linesearch)
end
=#
function ConjugateGradient(; linesearch! = nothing,
                             linesearch = LineSearches.hagerzhang!,
                             eta::Real = 0.4,
                             P::Any = nothing,
                             precondprep! = nothing,
                             precondprep = (P, x) -> nothing)

    linesearch = get_linesearch(linesearch!, linesearch)
    precondprep = get_precondprep(precondprep!, precondprep)
    ConjugateGradient(Float64(eta),
                      P, precondprep,
                      linesearch)
end

type ConjugateGradientState{T,N,G}
    @add_generic_fields()
    x_previous::Array{T,N}
    g_previous::G
    f_x_previous::T
    y::Array{T,N}
    py::Array{T,N}
    pg::Array{T,N}
    s::Array{T,N}
    @add_linesearch_fields()
end


function initial_state{T}(method::ConjugateGradient, options, d, initial_x::Array{T})
    value_grad!(d, initial_x)
    pg = copy(gradient(d))
    @assert typeof(value(d)) == T
    # Output messages
    if !isfinite(value(d))
        error("Must have finite starting value")
    end
    if !all(isfinite, gradient(d))
        @show gradient(d)
        @show find(!isfinite.(gradient(d)))
        error("Gradient must have all finite values at starting point")
    end

    # Determine the intial search direction
    #    if we don't precondition, then this is an extra superfluous copy
    #    TODO: consider allowing a reference for pg instead of a copy
    method.precondprep!(method.P, initial_x)
    A_ldiv_B!(pg, method.P, gradient(d))

    ConjugateGradientState("Conjugate Gradient",
                         length(initial_x),
                         copy(initial_x), # Maintain current state in state.x
                         similar(initial_x), # Maintain previous state in state.x_previous
                         similar(gradient(d)), # Store previous gradient in state.g_previous
                         T(NaN), # Store previous f in state.f_x_previous
                         similar(initial_x), # Intermediate value in CG calculation
                         similar(initial_x), # Preconditioned intermediate value in CG calculation
                         pg, # Maintain the preconditioned gradient in pg
                         -pg, # Maintain current search direction in state.s
                         @initial_linesearch()...) # Maintain a cache for line search results in state.lsr
end

function update_state!{T}(d, state::ConjugateGradientState{T}, method::ConjugateGradient)
        # Search direction is predetermined

        # Determine the distance of movement along the search line
        lssuccess = perform_linesearch!(state, method, d)

        # Maintain a record of previous position
        copy!(state.x_previous, state.x)

        # Update current position # x = x + alpha * s
        LinAlg.axpy!(state.alpha, state.s, state.x)

        # Maintain a record of the previous gradient
        copy!(state.g_previous, gradient(d))

        # Update the function value and gradient
        state.f_x_previous  = value(d)
        value_grad!(d, state.x)

        # Check sanity of function and gradient
        if !isfinite(value(d))
            error("Function value must be finite")
        end

        # Determine the next search direction using HZ's CG rule
        #  Calculate the beta factor (HZ2012)
        # -----------------
        # Comment on py: one could replace the computation of py with
        #    ydotpgprev = vecdot(y, pg)
        #    vecdot(y, py)  >>>  vecdot(y, pg) - ydotpgprev
        # but I am worried about round-off here, so instead we make an
        # extra copy, which is probably minimal overhead.
        # -----------------
        method.precondprep!(method.P, state.x)
        dPd = dot(state.s, method.P, state.s)
        etak::T = method.eta * vecdot(state.s, state.g_previous) / dPd
        @simd for i in 1:state.n
            @inbounds state.y[i] = gradient(d, i) - state.g_previous[i]
        end
        ydots = vecdot(state.y, state.s)
        copy!(state.py, state.pg)        # below, store pg - pg_previous in py
        A_ldiv_B!(state.pg, method.P, gradient(d))
        @simd for i in 1:state.n     # py = pg - py
           @inbounds state.py[i] = state.pg[i] - state.py[i]
        end
        betak = (vecdot(state.y, state.pg) - vecdot(state.y, state.py) * vecdot(gradient(d), state.s) / ydots) / ydots
        beta = max(betak, etak)
        @simd for i in 1:state.n
            @inbounds state.s[i] = beta * state.s[i] - state.pg[i]
        end
        lssuccess == false # break on linesearch error
end

update_g!(d, state, method::ConjugateGradient) = nothing
