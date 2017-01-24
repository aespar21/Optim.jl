@testset "Momentum Gradient Descent" begin
    for diff in (:analytic, :finitediff, :forwarddiff)
        for (name, prob) in Optim.UnconstrainedProblems.examples
            if prob.isdifferentiable
                if diff == :analytic
                    res = Optim.optimize(prob.f, prob.initial_x, MomentumGradientDescent(), Optim.Options(allow_f_increases=true))
                    @test norm(Optim.minimizer(res) - prob.solutions) < 1e-2
                else
                    od = OnceDifferentiable(prob.f, prob.initial_x, method = diff)
                    res = Optim.optimize(od, prob.initial_x, MomentumGradientDescent(), Optim.Options(iterations = 2000, allow_f_increases=true))
                    @test norm(Optim.minimizer(res) - prob.solutions) < 1e-2
                end
            end
        end
    end
end
