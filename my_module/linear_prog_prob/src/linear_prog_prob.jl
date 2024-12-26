module linear_prog_prob
export simplex, two_phase_simplex

function simplex(object::Matrix{T}, conditions_A::Matrix{T}, conditions_b::Vector{T}, non_Basic_Set::Union{Nothing, Set{Int}}=nothing) where T <: Union{Int, AbstractFloat}
    @assert size(object)[1] == 1 "the objective variable must be a horizontal vector :: size(object) = $(size(object))"
    @assert size(conditions_A)[1] == length(conditions_b) "the conditions variable's dimentions do not match :: size(conditions_A) = $(size(conditions_A)), length(conditions_b) = $(length(conditions_b))"
    @assert size(object)[2] == size(conditions_A)[2] "the conditions variable's dimentions do not match :: size(object) = $(size(object)), size(conditions_A) = $(size(conditions_A))"

    condits, expvars = size(conditions_A)
    @assert condits < expvars "The number of explanatory variables is greater than the number of conditions."

    if isnothing(non_Basic_Set)
        nonBasicSet = Set(1:(expvars - condits))
        basicSet    = Set((expvars - condits + 1):expvars)
    else
        nonBasicSet = non_Basic_Set
        basicSet    = setdiff(Set(1:expvars), nonBasicSet)
    end
    x           = zeros(Float64, expvars)
    z           = 0.0
    count       = 0
    while true
        nonBasicSpecifier = collect(nonBasicSet)
        basicSpecifier    = collect(basicSet)

        nonBasic = conditions_A[1:end, nonBasicSpecifier]
        basic    = conditions_A[1:end, basicSpecifier]
        coefNonBasic = object[1, nonBasicSpecifier]
        coefBasic    = object[1, basicSpecifier]

        inv_b  = inv(basic)
        B_dash = inv_b * conditions_b
        N_dash = inv_b * nonBasic

        xBasic = B_dash
        expect = (coefNonBasic' - coefBasic' * N_dash)'

        if count ≠ 0 && count % 1e5 == 0
            println(count, "回目の反復試行")
        end

        if any(xBasic .< 0)
            println("収束に失敗しました")
            println("実行可能解を得られませんでした")
            println("基底ベクトル：", basicSpecifier, " = ", xBasic)
            println("結果：", count, "回の反復試行を行いました")
            x[basicSpecifier] = xBasic
            z = coefBasic' * xBasic
            break
        end

        if maximum(expect) <= 0
            println("収束しました")
            println("結果：", count, "回の反復試行を行いました")
            x[basicSpecifier] = xBasic
            z = coefBasic' * xBasic
            break
        end

        bland_candid = findall(x -> x == maximum(expect), expect)
        idx_nonBasic = bland_candid[findfirst(x -> nonBasicSpecifier[x] == minimum(nonBasicSpecifier[bland_candid]), bland_candid)]
        xBasic2      = xBasic ./ N_dash[:, idx_nonBasic]
        xBasic2[xBasic2 .< 0] .= Inf64 # 非負制約
        bland_candid = findall(x -> x == minimum(xBasic2), xBasic2)
        idx_basic    = bland_candid[findfirst(x -> basicSpecifier[x]    == minimum(basicSpecifier[bland_candid]),    bland_candid)]

        if all(isinf.(xBasic2))
            println("収束に失敗しました")
            println("設定された問題は非有界です")
            println("結果：", count, "回の反復試行を行いました")
            x[basicSpecifier] = xBasic
            z = coefBasic' * xBasic
            break
        end

        push!(nonBasicSet, basicSpecifier[idx_basic])
        pop!( nonBasicSet, nonBasicSpecifier[idx_nonBasic])
        push!(basicSet,    nonBasicSpecifier[idx_nonBasic])
        pop!( basicSet,    basicSpecifier[idx_basic])

        # 反復回数のカウント
        count += 1
    end

    return x, z
end

function two_phase_simplex(object::Matrix{T}, conditions_A::Matrix{T}, conditions_b::Vector{T}) where T <: Union{Int, AbstractFloat}
    @assert size(object)[1] == 1 "the objective variable must be a horizontal vector :: size(object) = $(size(object))"
    @assert size(conditions_A)[1] == length(conditions_b) "the conditions variable's dimentions do not match :: size(conditions_A) = $(size(conditions_A)), length(conditions_b) = $(length(conditions_b))"
    @assert size(object)[2] == size(conditions_A)[2] "the conditions variable's dimentions do not match :: size(object) = $(size(object)), size(conditions_A) = $(size(conditions_A))"

    condits, expvars = size(conditions_A)
    @assert condits < expvars "The number of explanatory variables is greater than the number of conditions."

    auxiliar_object       = hcat(zeros(T, 1, expvars),  ones(1, 1))
    auxiliar_conditions_A = hcat(conditions_A,         -ones(condits, 1))
    condits, expvars      = size(auxiliar_conditions_A)

    nonBasicSet = Set(1:(expvars - 1 - condits))
    basicSet    = Set((expvars - condits):(expvars - 1))

    push!(nonBasicSet, expvars)
    
    nonBasicSpecifier = collect(nonBasicSet)
    basicSpecifier    = collect(basicSet)

    basic  = auxiliar_conditions_A[1:end, basicSpecifier]
    xBasic = inv(basic) * conditions_b

    if all(xBasic .>= 0)
        println("実行可能解を発見しました")
        println("単体法を実行します")
        x, z = simplex(object, conditions_A, conditions_b)
        return x, z
    end

    idx_nonBasic = findfirst(x -> x == expvars,         nonBasicSpecifier)
    idx_basic    = findfirst(x -> x == minimum(xBasic), xBasic)

    push!(nonBasicSet, basicSpecifier[idx_basic])
    pop!( nonBasicSet, nonBasicSpecifier[idx_nonBasic])
    push!(basicSet,    nonBasicSpecifier[idx_nonBasic])
    pop!( basicSet,    basicSpecifier[idx_basic])

    while true
        nonBasicSpecifier = collect(nonBasicSet)
        basicSpecifier    = collect(basicSet)

        nonBasic = auxiliar_conditions_A[1:end, nonBasicSpecifier]
        basic    = auxiliar_conditions_A[1:end, basicSpecifier]
        coefNonBasic = auxiliar_object[1, nonBasicSpecifier]
        coefBasic    = auxiliar_object[1, basicSpecifier]

        inv_b  = inv(basic)
        B_dash = inv_b * conditions_b
        N_dash = inv_b * nonBasic

        xBasic = copy(B_dash)
        expect = (coefNonBasic' - coefBasic' * N_dash)'

        if minimum(expect) >= 0
            println("収束しました")
            println("実行可能解を発見しました")
            println("非基底ベクトル：", nonBasicSpecifier, " = ", zeros(length(nonBasicSpecifier)))
            println("基底ベクトル：", basicSpecifier, " = ", xBasic)
            println("補助変数は ", expvars, " です")
            break
        end

        bland_candid = findall(x -> x == minimum(filter(x->x≠0, expect)), expect)
        idx_nonBasic = bland_candid[findfirst(x -> nonBasicSpecifier[x] == minimum(nonBasicSpecifier[bland_candid]), bland_candid)]
        xBasic2      = xBasic ./ N_dash[:, idx_nonBasic]
        bland_candid = findall(x -> x == minimum(xBasic2), xBasic2)
        idx_basic    = bland_candid[findfirst(x -> basicSpecifier[x] == minimum(basicSpecifier[bland_candid]), bland_candid)]

        push!(nonBasicSet, basicSpecifier[idx_basic])
        pop!( nonBasicSet, nonBasicSpecifier[idx_nonBasic])
        push!(basicSet,    nonBasicSpecifier[idx_nonBasic])
        pop!( basicSet,    basicSpecifier[idx_basic])
    end

    pop!(nonBasicSet, expvars)
    println("単体法を実行します")
    x, z = simplex(object, conditions_A, conditions_b, nonBasicSet)
    return x, z
end

end # module linear_prog_prob
