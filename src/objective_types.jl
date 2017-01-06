type NonDifferentiableFunction
    f
    f_x
    f_calls::Integer
end
NonDifferentiableFunction{T}(f, x_seed::Array{T}) = NonDifferentiableFunction(f, zero(T), 0)

type DifferentiableFunction
    f
    g!
    fg!
    f_x
    g_x
    f_calls::Integer
    g_calls::Integer
end
DifferentiableFunction{T}(f, g!, fg!, x_seed::Array{T}) = DifferentiableFunction(f, g!, fg!, zero(T), similar(x_seed), 0, 0)
# TODO: Expose ability to do forward and backward differencing
function DifferentiableFunction{T}(f::Function, x_seed::Array{T})
    function g!(x::Array, storage::Array)
        Calculus.finite_difference!(f, x, storage, :central)
        return
    end
    function fg!(x::Array, storage::Array)
        g!(x, storage)
        return f(x)
    end
    return DifferentiableFunction(f, g!, fg!, zero(T), similar(x_seed), 0, 0)
end

function DifferentiableFunction{T}(f::Function, g!::Function, x_seed::Array{T})
    function fg!(x::Array, storage::Array)
        g!(x, storage)
        return f(x)
    end
    return DifferentiableFunction(f, g!, fg!, zero(T), similar(x_seed), 0, 0)
end

type TwiceDifferentiableFunction
    f
    g!
    fg!
    h!
    f_x
    g_x
    h_storage
    f_calls::Integer
    g_calls::Integer
    h_calls::Integer
end
function TwiceDifferentiableFunction{T}(f, g!, fg!, h!, x_seed::Array{T})
    n_x = length(x_seed)
    TwiceDifferentiableFunction(f, g!, fg!, h!, zero(T),
                                similar(x_seed), Array{T}(n_x, n_x), 0, 0, 0)
end
function TwiceDifferentiableFunction{T}(f::Function, x_seed::Array{T})
    n_x = length(x_seed)
    function g!(x::Vector, storage::Vector)
        Calculus.finite_difference!(f, x, storage, :central)
        return
    end
    function fg!(x::Vector, storage::Vector)
        g!(x, storage)
        return f(x)
    end
    function h!(x::Vector, storage::Matrix)
        Calculus.finite_difference_hessian!(f, x, storage)
        return
    end
    return TwiceDifferentiableFunction(f, g!, fg!, h!, zero(T),
                                       similar(x_seed), Array{T}(n_x, n_x), 0, 0, 0)
end

function TwiceDifferentiableFunction{T}(f::Function,
                                     g!::Function,
                                     h!::Function,
                                     x_seed::Array{T})
    n_x = length(x_seed)
    function fg!(x::Vector, storage::Vector)
        g!(x, storage)
        return f(x)
    end
    return TwiceDifferentiableFunction(f, g!, fg!, h!, zero(T),
                                       similar(x_seed), Array{T}(n_x, n_x), 0, 0, 0)
end

function value(obj, x)
    obj.f_calls += 1
    obj.f(x)
end

function value_grad!(obj, x)
    obj.f_calls += 1
    obj.g_calls += 1
    obj.f_x = obj.fg!(x, obj.g_x)
end

grad(obj) = obj.g_x
grad(obj, i) = obj.g_x[i]
