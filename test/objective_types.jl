@testset "objective types" begin
    @testset "*Differentiable constructors" begin
        for (name, prob) in Optim.UnconstrainedProblems.examples
            if prob.isdifferentiable
                f_x = prob.f(prob.initial_x)
                g_stor = similar(prob.initial_x)
                prob.g!(prob.initial_x, g_stor)
                od = OnceDifferentiable(prob.f, prob.g!, prob.initial_x)
                @test f_x == Optim.value(od)
                @test g_stor == Optim.gradient(od)
            end
            if prob.istwicedifferentiable && name != "Large Polynomial"
                f_x = prob.f(prob.initial_x)
                g_stor = similar(prob.initial_x)
                n = length(g_stor)
                prob.g!(prob.initial_x, g_stor)
                h_stor = zeros(n, n)
                prob.h!(prob.initial_x, h_stor)
                td = TwiceDifferentiable(prob.f, prob.g!, prob.h!, prob.initial_x)
                @test f_x == Optim.value(td)
                @test Optim.value(od) == Optim.value(td)
                @test g_stor == Optim.gradient(td)
                @test Optim.gradient(od) == Optim.gradient(td)
                @test h_stor == Optim.hessian(td)
                ad_td = TwiceDifferentiable(prob.f, prob.g!, prob.initial_x; method=:forwarddiff)
                @test Optim.gradient(td) == Optim.gradient(ad_td)
                @test isapprox(Optim.hessian(td), Optim.hessian(ad_td); rtol = 1e-3)
                ad_td = TwiceDifferentiable(prob.f, prob.g!, prob.initial_x; method=:forwarddiff)
                @test Optim.gradient(td) == Optim.gradient(ad_td)
                @test isapprox(Optim.gradient(td), Optim.gradient(ad_td); rtol = 1e-3)
                @test isapprox(Optim.hessian(td), Optim.hessian(ad_td); rtol = 1e-3)
                fd_td = TwiceDifferentiable(prob.f, prob.initial_x; method=:finitediff)
                @test isapprox(Optim.gradient(td), Optim.gradient(fd_td))
                @test isapprox(Optim.hessian(td), Optim.hessian(fd_td); rtol = 1e-3)
                fd_td = TwiceDifferentiable(prob.f, prob.g!, prob.initial_x; method=:finitediff)
                @test Optim.gradient(td) == Optim.gradient(fd_td)
                @test isapprox(Optim.hessian(td), Optim.hessian(fd_td); rtol = 1e-3)
            end
        end
    end
end
