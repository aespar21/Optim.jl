type NonDifferentiable
    f
    f_x
    f_calls
end
NonDifferentiable{T}(f, x_seed::Array{T}) = NonDifferentiable(f, zero(T), [0])

type FinDiff
    method
end
FinDiff(;g_method = :central) = FinDiff(g_method)

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
function Differentiable{T}(f::Function, x_seed::Array{T}; g_method = FinDiff())
    n_x = length(x_seed)
    f_calls = [0]
    g_calls = [0]
    function g!(x::Array, storage::Array)
        Calculus.finite_difference!(f, x, storage, g_method.method)
        f_calls[1] .+= g_method.method == :central ? 2*n_x : n_x
        return
    end
    function fg!(x::Array, storage::Array)
        g!(x, storage)
        return f(x)
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
function TwiceDifferentiable{T}(f::Function, x_seed::Array{T}; method = FinDiff())
    n_x = length(x_seed)
    f_calls = [0]
    g_calls = [0]
    h_calls = [0]
    function g!(x::Vector, storage::Vector)
        Calculus.finite_difference!(f, x, storage, method.method)
        f_calls[1] .+= method.method == :central ? 2*n_x : n_x
        return
    end
    function fg!(x::Vector, storage::Vector)
        g!(x, storage)
        return f(x)
    end
    function h!(x::Vector, storage::Matrix)
        Calculus.finite_difference_hessian!(f, x, storage)
        f_calls[1] .+= 2*n_x^2-2n_x # (n^2-n)/2 off-diagonal elements with 4 calls + n diagonals with 2 calls
        return
    end
    return TwiceDifferentiable(f, g!, fg!, h!, zero(T),
                                       similar(x_seed), Array{T}(n_x, n_x), f_calls, g_calls, h_calls)
end

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

function value_grad!(obj, x)
    obj.f_calls .+= 1
    obj.g_calls .+= 1
    obj.f_x = obj.fg!(x, obj.g_x)
end

grad(obj) = obj.g_x
grad(obj, i) = obj.g_x[i]
