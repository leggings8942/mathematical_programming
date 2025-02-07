module nonlinear_prog_prob
using LinearAlgebra

export penalty_method, barrier_method, Augmented_Lagrangian_Method, Internal_Point_Method

function effective_constrants(Q::Matrix{Float64}, c::Vector{Float64}, condition_a::Matrix{Float64}, condition_b::Vector{Float64}, x::Vector{Float64})
    # 開発断念
    # アルゴリズムは理解したが、実装が難しい
    Obj   = x' * Q * x + c' * x
    valid = condition_b - condition_a * x
    valid_idx = findall(isapprox.(valid, 0, atol=1e-8))
    x_new = inv(condition_a[valid_idx, :]) * condition_b[valid_idx]
    lagrange  = Q * x + c - condition_a[valid_idx, :]' * u
    return A*x - b
end

mutable struct Update_Rafael{T}
	alpha::T
	beta::T
	time::Int
	beta_t::T
	m::VecOrMat{T}
	v::VecOrMat{T}
	w::VecOrMat{T}
	σ_coef::T
end
Update_Rafael(alpha::T=0.1, beta::T=0.9) where T = Update_Rafael{T}(alpha, beta, 0, 1, T[], T[], T[], 0)

function soft_max(x::VecOrMat{T}, α::T) where T
	sign = typeof(x)(undef, size(x))
	sign[x .>= 0] .=  1
	sign[x .<  0] .= -1
	return sign .* max.(abs.(x), α)
end

function update(ur::Update_Rafael{T}, grads::VecOrMat{T}) where T
	if ur.time == 0
		ur.m = zeros(T, size(grads))
		ur.v = zeros(T, size(grads))
		ur.w = zeros(T, size(grads))
		ur.σ_coef = (1 + ur.beta) / 2
	end

	ur.time   += 1
	ur.beta_t *= ur.beta

	ur.m = ur.beta * ur.m + (1 - ur.beta) * grads
	m_hat = ur.m / (1 - ur.beta_t)

	ur.v = ur.beta * ur.v + (1 - ur.beta) * (grads .^ 2)
	ur.w = ur.beta * ur.w + (1 - ur.beta) * ((grads ./ soft_max(m_hat, 1e-32) .- 1) .^ 2)

	if ur.beta - ur.beta_t > 0.1
		v_hat    = ur.v * ur.σ_coef / (ur.beta - ur.beta_t)
		w_hat    = ur.w * ur.σ_coef / (ur.beta - ur.beta_t)

		output = ur.alpha * m_hat ./ max.(sqrt.((v_hat .+ w_hat) ./ 2), 1e-32)
	else
		output = ur.alpha * sign.(m_hat)
	end

	return output
end

function penalty_method(f::Function, df::Function, p::Function, dp::Function, x::VecOrMat, ρ::Float64, ϵ::Float64, max_iter::Int)
    for i in 1:max_iter
        for j in 1:max_iter
            x_new = x - 0.001 * (df(x) + ρ * dp(x))
            if sum(abs.(df(x_new) + ρ * dp(x_new))) < ϵ
                x = x_new
                break
            end
            x = x_new

            if j % 100 == 0
                println("iter I: ", i, " iter J: ", j, " x: ", x)
                println(" f(x): ", f(x), " df(x): ", df(x))
                println(" p(x): ", p(x), " dp(x): ", dp(x))
                println(" ρ: ", ρ)
            end
        end

        ρ = ρ * 2
        if sum(ρ * p(x)) < ϵ
            break
        end
    end
    return x
end

function barrier_method(f::Function, df::Function, p::Function, dp::Function, x::VecOrMat, ρ::Float64, ϵ::Float64, max_iter::Int)
    optim = Update_Rafael()
    count = 0
    for i in 1:max_iter
        x_new = x - update(optim, df(x) + ρ * dp(x))
        if sum(abs.((f(x_new) + ρ * p(x_new)) - (f(x) + ρ * p(x)))) < ϵ
            if count > 10
                x = x_new
                break
            end
            count += 1
        end
        x = x_new
        ρ = ρ / 1.01

        if i % 100 == 0
            println("iter: ", i, " x: ", x)
            println(" f(x): ", f(x), " df(x): ", df(x))
            println(" p(x): ", p(x), " dp(x):", dp(x))
            println(" ρ: ", ρ)
        end
    end
    return x
end

function Augmented_Lagrangian_Method(f::Function, df::Function, g::Vector{Function}, dg::Vector{Function}, x::VecOrMat, ρ::Float64, ϵ::Float64, max_iter::Int)
    m = length(g)
    u = zeros((m,))
    for i in 1:max_iter
        optim = Update_Rafael()
        for j in 1:max_iter
            ΔOBJ  = df(x)
            ΔCON  = sum([u[k] * dg[k](x) for k in 1:m])
            ΔAUG  = ρ * sum([g[k](x) * dg[k](x) for k in 1:m])
            ΔDIFF = ΔOBJ + ΔCON + ΔAUG
            x_new = x - update(optim, ΔDIFF)
            if sum(abs.(ΔDIFF)) < ϵ
                x = x_new
                break
            end
            x = x_new

            if j % 100 == 0
                println("iter I: ", i, " iter J: ", j, " x: ", x)
                println(" f(x): ", f(x), " df(x): ", df(x))
                for k in 1:m
                    println(" g(x): ", g[k](x), " dg(x): ", dg[k](x))
                end
                println(" ρ: ", ρ)
            end
        end
        for j in 1:m
            u[j] = u[j] + ρ * g[j](x)
        end
        ρ = ρ * 2

        s = sum(ρ * [g[j](x)^2 for j in 1:m])
        if s < ϵ
            break
        end
    end
    return x
end

function Internal_Point_Method(f::Function, df::Function, ddf::Function, ineq::Vector{Function}, dineq::Vector{Function}, ddineq::Vector{Function}, eq::Vector{Function}, deq::Vector{Function}, ddeq::Vector{Function}, x::Vector, ϵ::Float64, max_iter::Int)
    n = length(x)
    l = length(ineq)
    m = length(eq)
    s = ones(l)
    u = zeros(l + m)
    ρ = 1
    for i in 1:max_iter
        for j in 1:max_iter
            ΔOBJ   = df(x)
            ΔINEQ  = l ≠ 0 ? [dineq[o](x)[k] for k in 1:n, o in 1:l] : zeros(n, 1)
            ΔEQ    = m ≠ 0 ? [deq[o](x)[k]   for k in 1:n, o in 1:m] : zeros(n, 1)
            u_ineq = l ≠ 0 ? u[1:l]         : [0]
            u_eq   = m ≠ 0 ? u[l + 1:l + m] : [0]
            ΔDIFF  = ΔOBJ + ΔINEQ * u_ineq + ΔEQ * u_eq

            ΔΔOBJ  = ddf(x)
            ΔΔINEQ = l ≠ 0 ? [u[p]     * ddineq[p](x)[k, o] for k in 1:n, o in 1:n, p in 1:l] : zeros(n, n, 1)
            ΔΔEQ   = m ≠ 0 ? [u[l + p] * ddeq[p](x)[k, o]   for k in 1:n, o in 1:n, p in 1:m] : zeros(n, n, 1)
            ΔΔDIFF = ΔΔOBJ + sum(ΔΔINEQ, dims=3) + sum(ΔΔEQ, dims=3)

            ΔINEQ = l ≠ 0 ? ΔINEQ : []
            ΔEQ   = m ≠ 0 ? ΔEQ   : []

            hessian = zeros(n + 2l + m, n + 2l + m)
            hessian[1:n, 1:n] = ΔΔDIFF
            hessian[1:n, n + l + 1:n + 2l] = ΔINEQ
            hessian[1:n, n + 2l + 1:n + 2l + m] = ΔEQ
            hessian[n + 1:n + l, n + 1:n + l] = diagm(0 => u[1:l])
            hessian[n + 1:n + l, n + l + 1:n + 2l] = diagm(0 => s)
            hessian[n + l + 1:n + 2l, 1:n] = ΔINEQ'
            hessian[n + l + 1:n + 2l, n + 1:n + l] = diagm(0 => ones(l))
            hessian[n + 2l + 1:n + 2l + m, 1:n] = ΔEQ'

            jacobian = zeros(n + 2l + m)
            jacobian[1:n] = -ΔDIFF
            jacobian[n + 1:n + l] = ρ .- (u[1:l]) .* s
            jacobian[n + l + 1:n + 2l] = [-ineq[k](x) - s[k] for k in 1:l]
            jacobian[n + 2l + 1:n + 2l + m] = [-eq[k](x) for k in 1:m]

            ΔZ    = inv(hessian) * jacobian
            x_new = vcat(x, s, u) + 0.1 * ΔZ
            if sum(abs.(ΔDIFF)) < ϵ
                x = x_new[1:n]
                s = x_new[n + 1:n + l]
                u = x_new[n + l + 1:n + 2l + m]
                break
            end
            x = x_new[1:n]
            s = x_new[n + 1:n + l]
            u = x_new[n + l + 1:n + 2l + m]

            if j % 100 == 0
                println("iter I: ", i, " iter J: ", j, " x: ", x)
                println(" f(x): ", f(x), " df(x): ", df(x))
                for k in 1:l
                    println(" ine(x): ", ineq[k](x), " dine(x): ", dineq[k](x))
                end
                for k in 1:m
                    println(" eq(x): ", eq[k](x), " deq(x): ", deq[k](x))
                end
                println(" ρ: ", ρ)
            end
        end
        ρ = 0.1 * dot((u[1:l]), s) / l
        if ρ < ϵ
            break
        end
    end
    return x
end

function 凸_Quadratic_Internal_Point(Q::Matrix{Float64}, c::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64}, x::Vector{Float64}, ϵ::Float64, max_iter::Int)
    n = length(x)
    u = zeros(n)
    z = ones(n)
    ρ = 1
    for i in 1:max_iter
        for j in 1:max_iter
            ΔOBJ   = Q * x + c
            ΔCON   = A' * u
            ΔINVZ  = ρ * z
            ΔDIFF  = ΔOBJ - ΔCON - ΔINVZ

            ΔΔOBJ  = Q
            ΔΔCON  = A'
            ΔΔINVZ = ρ

            hessian = zeros(3n, 3n)
            hessian[1:n, 1:n] = ΔΔOBJ
            hessian[1:n, n + 1:2n] = -ΔΔCON
            hessian[1:n, 2n + 1:3n] = -ΔΔINVZ
            hessian[n + 1:2n, 1:n] = ΔΔCON'
            hessian[2n + 1:3n, 1:n] = diagm(0 => z)
            hessian[2n + 1:3n, 2n + 1:3n] = diagm(0 => x)

            jacobian = zeros(3n)
            jacobian[1:n] = -ΔDIFF
            jacobian[n + 1:2n] = 1 .- x .* z
            jacobian[2n + 1:3n] = b - A * x

            ΔZ    = inv(hessian) * jacobian
            x_new = vcat(x, u, z) + 0.1 * ΔZ
            if sum(abs.(ΔDIFF)) < ϵ
                x = x_new[1:n]
                u = x_new[n + 1:2n]
                z = x_new[2n + 1:3n]
                break
            end
            x = x_new[1:n]
            u = x_new[n + 1:2n]
            z = x_new[2n + 1:3n]

            if j % 100 == 0
                println("iter I: ", i, " iter J: ", j, " x: ", x)
                println(" f(x): ", f(x), " df(x): ", df(x))
                for k in 1:l
                    println(" ine(x): ", ineq[k](x), " dine(x): ", dineq[k](x))
                end
                for k in 1:m
                    println(" eq(x): ", eq[k](x), " deq(x): ", deq[k](x))
                end
                println(" ρ: ", ρ)
            end
        end
        ρ = 0.1 * dot((u[1:l]), s) / l
        if ρ < ϵ
            break
        end
    end
    return x
end
end # module nonlinear_prog_prob
