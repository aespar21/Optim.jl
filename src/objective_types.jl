type NonDifferentiable
    f
    f_x
    last_x
    f_calls
end
NonDifferentiable{T}(f, x_seed::Array{T}) = NonDifferentiable(f, f(x_seed), copy(x_seed), [1])

type Differentiable
    f
    g!
    fg!
    f_x
    g_x
    last_x
    f_calls
    g_calls
end
function Differentiable{T}(f, g!, fg!, x_seed::Array{T})
    g_x = similar(x_seed)
    g!(x_seed, g_x)
    Differentiable(f, g!, fg!, f(x_seed), g_x, copy(x_seed), [1], [1])
end
function Differentiable{T}(f, x_seed::Array{T}; method = :finitediff)
    n_x = length(x_seed)
    f_calls = [1]
    g_calls = [1]
    if method == :finitediff
        function g!(x::Array, storage::Array)
            Calculus.finite_difference!(x->(f_calls[1]+=1;f(x)), x, storage, :central)
            return
        end
        function fg!(x::Array, storage::Array)
            g!(x, storage)
            return f(x)
        end
    elseif method == :forwarddiff
        gcfg = ForwardDiff.GradientConfig(initial_x)
        g! = (x, out) -> ForwardDiff.gradient!(out, f, x, gcfg)

        fg! = (x, out) -> begin
            gr_res = DiffBase.DiffResult(zero(T), out)
            ForwardDiff.gradient!(gr_res, f, x, gcfg)
            DiffBase.value(gr_res)
        end
    end
    g_x = similar(x_seed)
    g!(x_seed, g_x)
    return Differentiable(f, g!, fg!, f(x_seed), g_x, copy(x_seed), f_calls, g_calls)
end

function Differentiable{T}(f, g!, x_seed::Array{T})
    function fg!(x::Array, storage::Array)
        g!(x, storage)
        return f(x)
    end
    g_x = similar(x_seed)
    g!(x_seed, g_x)
    return Differentiable(f, g!, fg!, f(x_seed), g_x, copy(x_seed), [1], [1])
end

type TwiceDifferentiable
    f
    g!
    fg!
    h!
    f_x
    g_x
    h_storage
    last_x
    f_calls
    g_calls
    h_calls
end
function TwiceDifferentiable{T}(f, g!, fg!, h!, x_seed::Array{T})
    n_x = length(x_seed)
    g_x = similar(x_seed)
    g!(x_seed, g_x)
    TwiceDifferentiable(f, g!, fg!, h!, f(x_seed),
                                g_x, Array{T}(n_x, n_x), copy(x_seed), [1], [1], [0])
end
function TwiceDifferentiable{T}(f, x_seed::Array{T}; method = :finitediff)
    n_x = length(x_seed)
    f_calls = [1]
    g_calls = [1]
    h_calls = [0]
    if method == :finitediff
        function g!(x::Vector, storage::Vector)
            Calculus.finite_difference!(x->(f_calls[1]+=1;f(x)), x, storage, :central)
            return
        end
        function fg!(x::Vector, storage::Vector)
            g!(x, storage)
            return f(x)
        end
        function h!(x::Vector, storage::Matrix)
            Calculus.finite_difference_hessian!(x->(f_calls[1]+=1;f(x)), x, storage)
            return
        end
    elseif method == :forwarddiff
        gcfg = ForwardDiff.GradientConfig(initial_x)
        g! = (x, out) -> ForwardDiff.gradient!(out, f, x, gcfg)

        fg! = (x, out) -> begin
            gr_res = DiffBase.DiffResult(zero(T), out)
            ForwardDiff.gradient!(gr_res, f, x, gcfg)
            DiffBase.value(gr_res)
        end

        hcfg = ForwardDiff.HessianConfig(initial_x)
        h! = (x, out) -> ForwardDiff.hessian!(out, f, x, hcfg)
    end
    g_x = similar(x_seed)
    g!(x_seed, g_x)
    return TwiceDifferentiable(f, g!, fg!, h!, f(x_seed),
                                       g_x, Array{T}(n_x, n_x), copy(x_seed), f_calls, g_calls, h_calls)
end


function TwiceDifferentiable{T}(f, g!, x_seed::Array{T}; method = :finitediff)
    n_x = length(x_seed)
    f_calls = [1]
    function fg!(x::Vector, storage::Vector)
        g!(x, storage)
        return f(x)
    end
    if method == :finitediff
        function h!(x::Vector, storage::Matrix)
            Calculus.finite_difference_hessian!(x->(f_calls[1]+=1;f(x)), x, storage)
            return
        end
    elseif method == :forwarddiff
        hcfg = ForwardDiff.HessianConfig(similar(x_seed))
        h! = (x, out) -> ForwardDiff.hessian!(out, f, x, hcfg)
    end
    g_x = similar(x_seed)
    g!(x_seed, g_x)
    return TwiceDifferentiable(f, g!, fg!, h!, f(x_seed),
                                       g_x, Array{T}(n_x, n_x), copy(x_seed), f_calls, [1], [0])
end
#=
function TwiceDifferentiable{T}(f, g!, fg!, x_seed::Array{T}; method = :finitediff)
    n_x = length(x_seed)
    f_calls = [1]
    if method == :finitediff
        function h!(x::Vector, storage::Matrix)
            Calculus.finite_difference_hessian!(x->(f_calls[1]+=1;f(x)), x, storage)
            return
        end
    elseif method == :forwarddiff
        hcfg = ForwardDiff.HessianConfig(similar(grad(d)))
        h! = (x, out) -> ForwardDiff.hessian!(out, f, x, hcfg)
    end
    g_x = similar(x_seed)
    g!(x_seed, g_x)
    return TwiceDifferentiable(f, g!, fg!, h!, f(x_seed),
                                       g_x, Array{T}(n_x, n_x), copy(x_seed), f_calls, [1], [0])
end
=#
function TwiceDifferentiable(d::Differentiable; method = :finitediff)
    n_x = length(d.last_x)
    T = eltype(d.last_x)
    if method == :finitediff
        function h!(x::Vector, storage::Matrix)
            Calculus.finite_difference_hessian!(x->(d.f_calls[1]+=1;d.f(x)), x, storage)
            return
        end
    elseif method == :forwarddiff
        hcfg = ForwardDiff.HessianConfig(similar(grad(d)))
        h! = (x, out) -> ForwardDiff.hessian!(out, d.f, x, hcfg)
    end
    return TwiceDifferentiable(d.f, d.g!, d.fg!, h!, d.f_x,
                                       d.g_x, Array{T}(n_x, n_x), d.last_x, d.f_calls, d.g_calls, [0])
end

function TwiceDifferentiable{T}(f,
                                     g!,
                                     h!,
                                     x_seed::Array{T})
    n_x = length(x_seed)
    function fg!(x::Vector, storage::Vector)
        g!(x, storage)
        return f(x)
    end
    g_x = similar(x_seed)
    g!(x_seed, g_x)
    return TwiceDifferentiable(f, g!, fg!, h!, f(x_seed),
                                       g_x, Array{T}(n_x, n_x), copy(x_seed), [1], [1], [0])
end

function value(obj, x)
    if x != obj.last_x
        obj.f_calls .+= 1
        return obj.f(x)
    else
        return obj.f_x
    end
end

function value!(obj, x)
    if x != obj.last_x
        obj.f_calls .+= 1
        copy!(obj.last_x, x)
        obj.f_x = obj.f(x)
    end
    obj.f_x
end

function value_grad!(obj, x)
    if x != obj.last_x
        obj.f_calls .+= 1
        obj.g_calls .+= 1
        copy!(obj.last_x, x)
        obj.f_x = obj.fg!(x, obj.g_x)
    end
    obj.f_x
end

#get_grad(obj) ?
grad(obj) = obj.g_x
grad(obj, i) = obj.g_x[i]
