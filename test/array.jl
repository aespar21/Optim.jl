@testset "input types" begin
    f(X) = (10 - X[1])^2 + (0 - X[2])^2 + (0 - X[3])^2 + (5 - X[4])^2

    function g!(X, S)
        S[1] = -20 + 2 * X[1]
        S[2] = 2 * X[2]
        S[3] = 2 * X[3]
        S[4] = -10 + 2 * X[4]
        return
    end
    @testset "vector" begin
        for m in (AcceleratedGradientDescent(), ConjugateGradient(), BFGS(), LBFGS(), NelderMead(), GradientDescent(), MomentumGradientDescent(), NelderMead(), SimulatedAnnealing(), ParticleSwarm())
            res = optimize(f, g!, [1., 0., 1., 0.], GradientDescent())

            @test typeof(Optim.minimizer(res)) <: Vector
            @test vecnorm(Optim.minimizer(res) - [10.0, 0.0, 0.0, 5.0]) < 10e-8
        end
        # TODO: Get finite differencing to work for generic arrays as well
        # optimize(f, eye(2), method = :gradient_descent)
    end

    @testset "matrix" begin
        for m in (AcceleratedGradientDescent(), ConjugateGradient(), BFGS(), LBFGS(), NelderMead(), GradientDescent(), MomentumGradientDescent(), NelderMead(), SimulatedAnnealing(), ParticleSwarm())
            res = optimize(f, g!, eye(2), GradientDescent())

            @test typeof(Optim.minimizer(res)) <: Matrix
            @test vecnorm(Optim.minimizer(res) - [10.0 0.0; 0.0 5.0]) < 10e-8
        end
        # TODO: Get finite differencing to work for generic arrays as well
        # optimize(f, eye(2), method = :gradient_descent)
    end

    @testset "tensor" begin
        eye3 = Array{Float64,3}(2,2,1)
        eye3[:,:,1] = eye(2)
        for m in (AcceleratedGradientDescent(), ConjugateGradient(), BFGS(), LBFGS(), NelderMead(), GradientDescent(), MomentumGradientDescent(), NelderMead(), SimulatedAnnealing(), ParticleSwarm())
            res = optimize(f, g!, eye3, GradientDescent())
            _minimizer = Optim.minimizer(res)
            @test typeof(_minimizer) <: Array
            @test size(_minimizer) == (2,2,1)
            @test vecnorm(_minimizer - [10.0 0.0; 0.0 5.0]) < 10e-8
        end
        # TODO: Get finite differencing to work for generic arrays as well
        # optimize(f, eye(2), method = :gradient_descent)
    end
end
