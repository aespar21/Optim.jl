@testset "Newton Trust Region" begin
@testset "Subproblems I" begin
    # verify that solve_tr_subproblem! finds the minimum
    n = 2
    gr = [-0.74637,0.52388]
    H = [0.945787 -3.07884; -3.07884 -1.27762]

    s = zeros(n)
    m, interior = Optim.solve_tr_subproblem!(gr, H, 1., s, max_iters=100)

    for j in 1:10
        bad_s = rand(n)
        bad_s ./= norm(bad_s)  # boundary
        model(s2) = (gr' * s2)[] + .5 * (s2' * H * s2)[]
        @test model(s) <= model(bad_s) + 1e-8
    end
end

@testset "Subproblems II" begin
    # random Hessians--verify that solve_tr_subproblem! finds the minimum
    for i in 1:10000
        n = rand(1:10)
        gr = randn(n)
        H = randn(n, n)
        H += H'

        s = zeros(n)
        m, interior = Optim.solve_tr_subproblem!(gr, H, 1., s, max_iters=100)

        model(s2) = (gr' * s2)[] + .5 * (s2' * H * s2)[]
        @test model(s) <= model(zeros(n)) + 1e-8  # origin

        for j in 1:10
            bad_s = rand(n)
            bad_s ./= norm(bad_s)  # boundary
            @test model(s) <= model(bad_s) + 1e-8
            bad_s .*= rand()  # interior
            @test model(s) <= model(bad_s) + 1e-8
        end
    end
end

@testset "Test problems" begin
    #######################################
    # First test the subproblem.
    srand(42)
    n = 5
    H = rand(n, n)
    H = H' * H + 4 * eye(n)
    H_eig = eigfact(H)
    U = H_eig[:vectors]

    gr = zeros(n)
    gr[1] = 1.
    s = zeros(Float64, n)

    true_s = -H \ gr
    s_norm2 = dot(true_s, true_s)
    true_m = dot(true_s, gr) + 0.5 * dot(true_s, H * true_s)

    # An interior solution
    delta = sqrt(s_norm2) + 1.0
    m, interior, lambda, hard_case, reached_solution =
        Optim.solve_tr_subproblem!(gr, H, delta, s)
    @test interior
    @test !hard_case
    @test reached_solution
    @test abs(m - true_m) < 1e-12
    @test norm(s - true_s) < 1e-12
    @test abs(lambda) < 1e-12

    # A boundary solution
    delta = 0.5 * sqrt(s_norm2)
    m, interior, lambda, hard_case, reached_solution =
        Optim.solve_tr_subproblem!(gr, H, delta, s)
    @test !interior
    @test !hard_case
    @test reached_solution
    @test m > true_m
    @test abs(norm(s) - delta) < 1e-12
    @test lambda > 0

    # A "hard case" where the gradient is orthogonal to the lowest eigenvector

    # Test the checking
    hard_case, lambda_1_multiplicity =
        Optim.check_hard_case_candidate([-1., 2., 3.], [0., 1., 1.])
    @test hard_case
    @test lambda_1_multiplicity == 1

    hard_case, lambda_1_multiplicity =
        Optim.check_hard_case_candidate([-1., -1., 3.], [0., 0., 1.])
    @test hard_case
    @test lambda_1_multiplicity == 2

    hard_case, lambda_1_multiplicity =
        Optim.check_hard_case_candidate([-1., -1., -1.], [0., 0., 0.])
    @test hard_case
    @test lambda_1_multiplicity == 3

    hard_case, lambda_1_multiplicity =
        Optim.check_hard_case_candidate([1., 2., 3.], [0., 1., 1.])
    @test !hard_case

    hard_case, lambda_1_multiplicity =
        Optim.check_hard_case_candidate([-1., -1., -1.], [0., 0., 1.])
    @test !hard_case

    hard_case, lambda_1_multiplicity =
        Optim.check_hard_case_candidate([-1., 2., 3.], [1., 1., 1.])
    @test !hard_case


    # Now check an actual had case problem
    L = zeros(Float64, n) + 0.1
    L[1] = -1.
    H = U * diagm(L) * U'
    H = 0.5 * (H' + H)
    @test issymmetric(H)
    gr = U[:,2][:]
    @test abs(dot(gr, U[:,1][:])) < 1e-12
    true_s = -H \ gr
    s_norm2 = dot(true_s, true_s)
    true_m = dot(true_s, gr) + 0.5 * dot(true_s, H * true_s)

    delta = 0.5 * sqrt(s_norm2)
    m, interior, lambda, hard_case, reached_solution =
        Optim.solve_tr_subproblem!(gr, H, delta, s)
    @test !interior
    @test hard_case
    @test reached_solution
    @test abs(lambda + L[1]) < 1e-4
    @test abs(norm(s) - delta) < 1e-12


    #######################################
    # Next, test on actual optimization problems.

    function f(x::Vector)
        (x[1] - 5.0)^4
    end

    function g!(x::Vector, storage::Vector)
        storage[1] = 4.0 * (x[1] - 5.0)^3
    end

    function h!(x::Vector, storage::Matrix)
        storage[1, 1] = 12.0 * (x[1] - 5.0)^2
    end

    d = TwiceDifferentiable(f, g!, h!)

    results = Optim.optimize(d, [0.0], NewtonTrustRegion())
    @test length(results.trace) == 0
    @test results.g_converged
    @test norm(Optim.minimizer(results) - [5.0]) < 0.01

    eta = 0.9

    function f_2(x::Vector)
        0.5 * (x[1]^2 + eta * x[2]^2)
    end

    function g!_2(x::Vector, storage::Vector)
        storage[1] = x[1]
        storage[2] = eta * x[2]
    end

    function h!_2(x::Vector, storage::Matrix)
        storage[1, 1] = 1.0
        storage[1, 2] = 0.0
        storage[2, 1] = 0.0
        storage[2, 2] = eta
    end

    d = TwiceDifferentiable(f_2, g!_2, h!_2)

    results = Optim.optimize(d, Float64[127, 921], NewtonTrustRegion())
    @test results.g_converged
    @test norm(Optim.minimizer(results) - [0.0, 0.0]) < 0.01

    # Test Optim.newton for all twice differentiable functions in
    # Optim.UnconstrainedProblems.examples
    for (name, prob) in Optim.UnconstrainedProblems.examples
    	if prob.istwicedifferentiable
    		ddf = OnceDifferentiable(prob.f, prob.g!)
    		res = Optim.optimize(ddf, prob.initial_x, NewtonTrustRegion(), Optim.Options(autodiff = true))
    		@test norm(Optim.minimizer(res) - prob.solutions) < 1e-2
    		res = Optim.optimize(ddf.f, prob.initial_x, NewtonTrustRegion(), Optim.Options(autodiff = true))
    		@test norm(Optim.minimizer(res) - prob.solutions) < 1e-2
            res = Optim.optimize(ddf.f, ddf.g!, prob.initial_x, NewtonTrustRegion(), Optim.Options(autodiff = true))
    		@test norm(Optim.minimizer(res) - prob.solutions) < 1e-2
    	end
    end
end


@testset "PR #341" begin
    # verify that no PosDef exception is thrown
    Optim.solve_tr_subproblem!([0, 1.], [-1000 0; 0. -999], 1e-2, ones(2))
end

end
