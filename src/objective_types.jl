type NonDifferentiable
    f
    f_x
    f_calls
end
NonDifferentiable{T}(f, x_seed::Array{T}) = NonDifferentiable(f, zero(T), [0])

type Differentiable
    f
    g!
    fg!
    f_x
    g_x
    f_calls
    g_calls
end
Differentiable{T}(f, g!, fg!, x_seed::Array{T}) = Differentiable(f, g!, fg!, zero(T), similar(x_seed), [0], [0])
function Differentiable{T}(f::Function, x_seed::Array{T}; method = :finitediff)
    n_x = length(x_seed)
    f_calls = [0]
    g_calls = [0]
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
    return Differentiable(f, g!, fg!, zero(T), similar(x_seed), f_calls, g_calls)
end

function Differentiable{T}(f::Function, g!::Function, x_seed::Array{T})
    function fg!(x::Array, storage::Array)
        g!(x, storage)
        return f(x)
    end
    return Differentiable(f, g!, fg!, zero(T), similar(x_seed), [0], [0])
end

type TwiceDifferentiable
    f
    g!
    fg!
    h!
    f_x
    g_x
    h_storage
    f_calls
    g_calls
    h_calls
end
function TwiceDifferentiable{T}(f, g!, fg!, h!, x_seed::Array{T})
    n_x = length(x_seed)
    TwiceDifferentiable(f, g!, fg!, h!, zero(T),
                                similar(x_seed), Array{T}(n_x, n_x), [0], [0], [0])
end
function TwiceDifferentiable{T}(f::Function, x_seed::Array{T}; method = :finitediff)
    n_x = length(x_seed)
    f_calls = [0]
    g_calls = [0]
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
    return TwiceDifferentiable(f, g!, fg!, h!, zero(T),
                                       similar(x_seed), Array{T}(n_x, n_x), f_calls, g_calls, h_calls)
end


function TwiceDifferentiable{T}(f, g!, x_seed::Array{T}; method = :finitediff)
    n_x = length(x_seed)
    f_calls = [0]
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
    return TwiceDifferentiable(f, g!, fg!, h!, zero(T),
                                       similar(x_seed), Array{T}(n_x, n_x), f_calls, [0], [0])
end

function TwiceDifferentiable{T}(f, g!, fg!, x_seed::Array{T}; method = :finitediff)
    n_x = length(x_seed)
    f_calls = [0]
    if method == :finitediff
        function h!(x::Vector, storage::Matrix)
            Calculus.finite_difference_hessian!(x->(f_calls[1]+=1;f(x)), x, storage)
            return
        end
    elseif method == :forwarddiff
        hcfg = ForwardDiff.HessianConfig(similar(grad(d)))
        h! = (x, out) -> ForwardDiff.hessian!(out, f, x, hcfg)
    end
    return TwiceDifferentiable(f, g!, fg!, h!, zero(T),
                                       similar(x_seed), Array{T}(n_x, n_x), f_calls, [0], [0])
end

TwiceDifferentiable(d::Differentiable;
                    method = :finitediff) = TwiceDifferentiable(d.f,
                                                                d.g!,
                                                                d.fg!,
                                                                similar(grad(d));
                                                                method = method)


function TwiceDifferentiable{T}(f::Function,
                                     g!::Function,
                                     h!::Function,
                                     x_seed::Array{T})
    n_x = length(x_seed)
    function fg!(x::Vector, storage::Vector)
        g!(x, storage)
        return f(x)
    end
    return TwiceDifferentiable(f, g!, fg!, h!, zero(T),
                                       similar(x_seed), Array{T}(n_x, n_x), [0], [0], [0])
end

function value(obj, x)
    obj.f_calls .+= 1
    obj.f(x)
end

function value!(obj, x)
    obj.f_calls .+= 1
    obj.f(x)
end

function value_grad!(obj, x)
    obj.f_calls .+= 1
    obj.g_calls .+= 1
    obj.f_x = obj.fg!(x, obj.g_x)
end

#get_grad(obj) ?
grad(obj) = obj.g_x
grad(obj, i) = obj.g_x[i]
