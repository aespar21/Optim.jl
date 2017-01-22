type NonDifferentiable
    f
    f_x
    last_x_f
    f_calls
end
NonDifferentiable{T}(f, x_seed::Array{T}) = NonDifferentiable(f, f(x_seed), copy(x_seed), [1])

type Differentiable{Tfg <: Union{Function, Void}}
    f
    g!
    fg!::Tfg
    f_x
    g
    last_x_f
    last_x_g
    f_calls
    g_calls
end
function Differentiable{T}(f, g!, fg!, x_seed::Array{T})
    g = similar(x_seed)
    g!(x_seed, g)
    Differentiable(f, g!, fg!, f(x_seed), g, copy(x_seed), copy(x_seed), [1], [1])
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
    g = similar(x_seed)
    g!(x_seed, g)
    return Differentiable(f, g!, fg!, f(x_seed), g, copy(x_seed), copy(x_seed), f_calls, g_calls)
end

function Differentiable{T}(f, g!, x_seed::Array{T})
    function fg!(x::Array, storage::Array)
        g!(x, storage)
        return f(x)
    end
    g = similar(x_seed)
    g!(x_seed, g)
    return Differentiable(f, g!, fg!, f(x_seed), g, copy(x_seed), copy(x_seed), [1], [1])
end

type TwiceDifferentiable{Tfg <: Union{Function, Void}}
    f
    g!
    fg!::Tfg
    h!
    f_x
    g
    H
    last_x_f
    last_x_g
    last_x_h
    f_calls
    g_calls
    h_calls
end
function TwiceDifferentiable{T}(f, g!, fg!, h!, x_seed::Array{T})
    n_x = length(x_seed)
    g = similar(x_seed)
    H = Array{T}(n_x, n_x)
    g!(x_seed, g)
    h!(x_seed, H)
    TwiceDifferentiable(f, g!, fg!, h!, f(x_seed),
                                g, H, copy(x_seed),
                                copy(x_seed), copy(x_seed), [1], [1], [1])
end
function TwiceDifferentiable{T}(f, x_seed::Array{T}; method = :finitediff)
    n_x = length(x_seed)
    f_calls = [1]
    g_calls = [1]
    h_calls = [1]
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
    g = similar(x_seed)
    H = Array{T}(n_x, n_x)
    g!(x_seed, g)
    h!(x_seed, H)
    return TwiceDifferentiable(f, g!, fg!, h!, f(x_seed),
                                       g, H, copy(x_seed),
                                       copy(x_seed), copy(x_seed), f_calls, g_calls, h_calls)
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
    g = similar(x_seed)
    H = Array{T}(n_x, n_x)
    g!(x_seed, g)
    h!(x_seed, H)
    return TwiceDifferentiable(f, g!, fg!, h!, f(x_seed),
                                       g, H, copy(x_seed),
                                       copy(x_seed), copy(x_seed), f_calls, [1], [1])
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
    g = similar(x_seed)
    g!(x_seed, g)
    return TwiceDifferentiable(f, g!, fg!, h!, f(x_seed),
                                       g, Array{T}(n_x, n_x), copy(x_seed), f_calls, [1], [0])
end
=#
function TwiceDifferentiable(d::Differentiable; method = :finitediff)
    n_x = length(d.last_x_f)
    T = eltype(d.last_x_f)
    if method == :finitediff
        function h!(x::Vector, storage::Matrix)
            Calculus.finite_difference_hessian!(x->(d.f_calls[1]+=1;d.f(x)), x, storage)
            return
        end
    elseif method == :forwarddiff
        hcfg = ForwardDiff.HessianConfig(similar(grad(d)))
        h! = (x, out) -> ForwardDiff.hessian!(out, d.f, x, hcfg)
    end
    H = Array{T}(n_x, n_x)
    h!(d.last_x_g, H)
    return TwiceDifferentiable(d.f, d.g!, d.fg!, h!, d.f_x,
                                       grad(d), Array{T}(n_x, n_x), d.last_x_f,
                                       d.last_x_g, similar(d.last_x_g), d.f_calls, d.g_calls, [1])
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
    g = similar(x_seed)
    H = Array{T}(n_x, n_x)
    g!(x_seed, g)
    h!(x_seed, H)
    return TwiceDifferentiable(f, g!, fg!, h!, f(x_seed),
                                       g, H, copy(x_seed),
                                       copy(x_seed), copy(x_seed), [1], [1], [0])
end

function value(obj, x)
    if x != obj.last_x_f
        obj.f_calls .+= 1
        return obj.f(x)
    else
        return obj.f_x
    end
end

function _unchecked_value!(obj, x)
    obj.f_calls .+= 1
    copy!(obj.last_x_f, x)
    obj.f_x = obj.f(x)
end
function value!(obj, x)
    if x != obj.last_x_f
        _unchecked_value!(obj, x)
    end
    obj.f_x
end


function _unchecked_grad!(obj, x)
    obj.g_calls .+= 1
    copy!(obj.last_x_g, x)
    obj.g!(x)
end
function gradient!(obj, x)
    if x != obj.last_x_g
        _unchecked_gradient!(obj, x)
    end
end

function value_grad!(obj, x)
    if x != obj.last_x_f && x != obj.last_x_g
        obj.f_calls .+= 1
        obj.g_calls .+= 1
        obj.last_x_f[:], obj.last_x_g[:] = copy(x), copy(x)
        obj.f_x = obj.fg!(x, obj.g)
    elseif x != obj.last_x_f
        _unchecked_value!(obj, x)
    elseif x != obj.last_x_g
        _unchecked_grad!(obj, x)
    end
    obj.f_x
end

function _unchecked_hessian!(obj, x)
    obj.h_calls .+= 1
    copy!(obj.last_x_h, x)
    obj.h!(x, obj.H)
end
function hessian!(obj, x)
    if x != obj.last_x_h
        _unchecked_hessian!(obj, x)
    end
end

#get_grad(obj) ?
grad(obj) = obj.g
grad(obj, i) = obj.g[i]
hessian(obj) = obj.H
#=
# This can be used when LineSearches switches to value_grad! and family
# Remember to change all fg!'s to "nothing" for finite differences
# and when f and g! are passed but no fg!. Can potentially avoid more calls
# than current setup.
function value_grad!(obj::Union{Differentiable{Void}, TwiceDifferentiable{Void}}, x)
    if x != obj.last_x_f
        _unchecked_value!(obj, x)
    end
    if x != obj.last_x_g
        _unchecked_grad!(obj, x)
    end
    obj.f_x
end
=#