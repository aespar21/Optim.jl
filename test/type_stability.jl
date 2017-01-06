let
    function rosenbrock{T}(x::Vector{T})
        o = one(T)
        c = convert(T,100)
        return (o - x[1])^2 + c * (x[2] - x[1]^2)^2
    end

    function rosenbrock_gradient!{T}(x::Vector{T}, storage::Vector{T})
        o = one(T)
        c = convert(T,100)
        storage[1] = (-2*o) * (o - x[1]) - (4*c) * (x[2] - x[1]^2) * x[1]
        storage[2] = (2*c) * (x[2] - x[1]^2)
    end

    function rosenbrock_hessian!{T}(x::Vector{T}, storage::Matrix{T})
        o = one(T)
        c = convert(T,100)
        f = 4*c
        storage[1, 1] = (2*o) - f * x[2] + 3 * f * x[1]^2
        storage[1, 2] = -f * x[1]
        storage[2, 1] = -f * x[1]
        storage[2, 2] = 2*c
    end

    for method in (NelderMead(), SimulatedAnnealing())
        optimize(rosenbrock, [0.0,0,.0], method)
        optimize(rosenbrock, Float32[0.0, 0.0], method)
    end

    for method in (BFGS(),
                   ConjugateGradient(),
                   GradientDescent(),
                   MomentumGradientDescent(),
                   AcceleratedGradientDescent(),
                   LBFGS())
        optimize(rosenbrock, rosenbrock_gradient!, [0.0,0,.0], method)
        optimize(rosenbrock, rosenbrock_gradient!, Float32[0.0, 0.0], method)
    end

    for method in (Newton(),)# NewtonTrustRegion())
        optimize(rosenbrock, rosenbrock_gradient!, rosenbrock_hessian!, [0.0,0.0], method)
        optimize(rosenbrock, rosenbrock_gradient!, rosenbrock_hessian!, Float32[0.0, 0.0], method)
    end
end
