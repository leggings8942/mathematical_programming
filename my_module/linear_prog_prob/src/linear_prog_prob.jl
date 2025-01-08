module linear_prog_prob
using LinearAlgebra

export LP_Prob
export simplex, two_phase_simplex, convert_dual_problem

@enum LP_Prob begin
    Standard
    No_nonNegative
    Equality
end

function standard_simplex(object::Matrix{T}, conditions_A::Matrix{T}, conditions_b::Vector{T}, non_Basic_Set::Union{Nothing, Set{Int}}=nothing) where T <: Union{Int, AbstractFloat}
    @assert size(object)[1] == 1 "the objective variable must be a horizontal vector :: size(object) = $(size(object))"
    @assert size(conditions_A)[1] == length(conditions_b) "the conditions variable's dimentions do not match :: size(conditions_A) = $(size(conditions_A)), length(conditions_b) = $(length(conditions_b))"
    @assert size(object)[2] == size(conditions_A)[2] "the conditions variable's dimentions do not match :: size(object) = $(size(object)), size(conditions_A) = $(size(conditions_A))"

    condits, expvars = size(conditions_A)
    if isnothing(non_Basic_Set)
        nonBasicSet = Set(1:expvars)
        basicSet    = Set((expvars + 1):(expvars + condits))
    else
        nonBasicSet = non_Basic_Set
        basicSet    = setdiff(Set(1:(expvars + condits)), nonBasicSet)
    end
    Obj         = hcat(object, zeros(T, 1, condits))
    Cond_A      = hcat(conditions_A, Matrix{T}(I, condits, condits))
    x           = zeros(Float64, expvars + condits)
    z           = 0.0
    count       = 0
    while true
        nonBasicSpecifier = collect(nonBasicSet)
        basicSpecifier    = collect(basicSet)

        nonBasic = Cond_A[1:end, nonBasicSpecifier]
        basic    = Cond_A[1:end, basicSpecifier]
        coefNonBasic = Obj[1, nonBasicSpecifier]
        coefBasic    = Obj[1, basicSpecifier]

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
        xBasic2[xBasic2 .< 0]    .= Inf64 # 非負制約
        xBasic2[isnan.(xBasic2)] .= Inf64 # 非数値制約
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

function standard_two_phase_simplex(object::Matrix{T}, conditions_A::Matrix{T}, conditions_b::Vector{T}) where T <: Union{Int, AbstractFloat}
    @assert size(object)[1] == 1 "the objective variable must be a horizontal vector :: size(object) = $(size(object))"
    @assert size(conditions_A)[1] == length(conditions_b) "the conditions variable's dimentions do not match :: size(conditions_A) = $(size(conditions_A)), length(conditions_b) = $(length(conditions_b))"
    @assert size(object)[2] == size(conditions_A)[2] "the conditions variable's dimentions do not match :: size(object) = $(size(object)), size(conditions_A) = $(size(conditions_A))"

    condits, expvars = size(conditions_A)
    Cond_A           = hcat(conditions_A, Matrix{T}(I, condits, condits))

    auxiliar_object       = hcat(zeros(T, 1, expvars + condits),  ones(1, 1))
    auxiliar_conditions_A = hcat(Cond_A,                         -ones(condits, 1))
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

            if coefBasic' * xBasic == 0
                println("実行可能解を発見しました")
                println("非基底ベクトル：", nonBasicSpecifier, " = ", zeros(length(nonBasicSpecifier)))
                println("基底ベクトル：", basicSpecifier, " = ", xBasic)
                println("補助変数は ", expvars, " です")
                
                try
                    pop!(nonBasicSet, expvars)
                catch
                    println("補助変数が存在しません")
                    pop!(basicSet, expvars)

                    proxyVariable = rand(nonBasicSet)
                    while true
                        try
                            proxyVariable  = rand(nonBasicSet)
                            proxySpecifier = collect(basicSet ∪ Set(proxyVariable))
                            basic  = Cond_A[1:end, proxySpecifier]
                            inv(basic)
                            break
                        catch
                            continue
                        end
                    end

                    pop!(nonBasicSet, proxyVariable)
                end
                println("基数の初期値を設定しました：", nonBasicSet)
                println("単体法を実行します")
                x, z = standard_simplex(object, conditions_A, conditions_b, nonBasicSet)
                return x, z
            else
                println("実行可能解を得られませんでした")
                println("非基底ベクトル：", nonBasicSpecifier, " = ", zeros(length(nonBasicSpecifier)))
                println("基底ベクトル：", basicSpecifier, " = ", xBasic)
                println("補助変数は ", expvars, " です")
                return nothing
            end
        end

        bland_candid = findall(x -> x == minimum(filter(x->x≠0, expect)), expect)
        idx_nonBasic = bland_candid[findfirst(x -> nonBasicSpecifier[x] == minimum(nonBasicSpecifier[bland_candid]), bland_candid)]
        xBasic2      = xBasic ./ N_dash[:, idx_nonBasic]
        xBasic2[xBasic2 .< 0] .= Inf64 # 非負制約
        bland_candid = findall(x -> x == minimum(xBasic2), xBasic2)
        idx_basic    = bland_candid[findfirst(x -> basicSpecifier[x] == minimum(basicSpecifier[bland_candid]), bland_candid)]

        push!(nonBasicSet, basicSpecifier[idx_basic])
        pop!( nonBasicSet, nonBasicSpecifier[idx_nonBasic])
        push!(basicSet,    nonBasicSpecifier[idx_nonBasic])
        pop!( basicSet,    basicSpecifier[idx_basic])
    end

    return nothing
end

function simplex(object::Matrix{T}, conditions_A::Matrix{T}, conditions_b::Vector{T}; format::LP_Prob=Standard, non_Basic_Set::Union{Nothing, Set{Int}}=nothing) where T <: Union{Int, AbstractFloat}
    @assert size(object)[1] == 1 "the objective variable must be a horizontal vector :: size(object) = $(size(object))"
    @assert size(conditions_A)[1] == length(conditions_b) "the conditions variable's dimentions do not match :: size(conditions_A) = $(size(conditions_A)), length(conditions_b) = $(length(conditions_b))"
    @assert size(object)[2] == size(conditions_A)[2] "the conditions variable's dimentions do not match :: size(object) = $(size(object)), size(conditions_A) = $(size(conditions_A))"
    
    if format == Standard
        return standard_simplex(object, conditions_A, conditions_b, non_Basic_Set)
    elseif format == No_nonNegative
        convert_object       = hcat(object,       -object)
        convert_conditions_A = hcat(conditions_A, -conditions_A)
        convert_conditions_b = conditions_b
        if isnothing(non_Basic_Set)
            return standard_simplex(convert_object, convert_conditions_A, convert_conditions_b)
        else
            convert_non_Basic_Set = non_Basic_Set ∪ Set(collect(non_Basic_Set) .+ (expvars + condits))
            return standard_simplex(convert_object, convert_conditions_A, convert_conditions_b, convert_non_Basic_Set)
        end
    elseif format == Equality
        convert_object       = object
        convert_conditions_A = vcat(conditions_A, -conditions_A)
        convert_conditions_b = vcat(conditions_b, -conditions_b)
        return standard_simplex(convert_object, convert_conditions_A, convert_conditions_b, non_Basic_Set)
    end

    return nothing
end

function two_phase_simplex(object::Matrix{T}, conditions_A::Matrix{T}, conditions_b::Vector{T}; format::LP_Prob=Standard) where T <: Union{Int, AbstractFloat}
    @assert size(object)[1] == 1 "the objective variable must be a horizontal vector :: size(object) = $(size(object))"
    @assert size(conditions_A)[1] == length(conditions_b) "the conditions variable's dimentions do not match :: size(conditions_A) = $(size(conditions_A)), length(conditions_b) = $(length(conditions_b))"
    @assert size(object)[2] == size(conditions_A)[2] "the conditions variable's dimentions do not match :: size(object) = $(size(object)), size(conditions_A) = $(size(conditions_A))"

    if format == Standard
        res = standard_two_phase_simplex(object, conditions_A, conditions_b)
        if isnothing(res)
            return nothing
        else
            convert_x, convert_z = res
        end

        x = convert_x[1:length(object)]
        z = convert_z
        return x, z
    elseif format == No_nonNegative
        convert_object       = hcat(object,       -object)
        convert_conditions_A = hcat(conditions_A, -conditions_A)
        convert_conditions_b = conditions_b
        res = standard_two_phase_simplex(convert_object, convert_conditions_A, convert_conditions_b)
        if isnothing(res)
            return nothing
        else
            convert_x, convert_z = res
        end

        x = convert_x[1:length(object)] - convert_x[(length(object) + 1):2*length(object)]
        z = convert_z
        return x, z
    elseif format == Equality
        convert_object       = object
        convert_conditions_A = vcat(conditions_A, -conditions_A)
        convert_conditions_b = vcat(conditions_b, -conditions_b)
        res = standard_two_phase_simplex(convert_object, convert_conditions_A, convert_conditions_b)
        if isnothing(res)
            return nothing
        else
            convert_x, convert_z = res
        end
        

        x = convert_x[1:length(object)]
        z = convert_z
        return x, z
    end

    return nothing
end

function convert_dual_problem(object::Matrix{T}, conditions_A::Matrix{T}, conditions_b::Vector{T}; format::LP_Prob=Standard) where T <: Union{Int, AbstractFloat}
    @assert size(object)[1] == 1 "the objective variable must be a horizontal vector :: size(object) = $(size(object))"
    @assert size(conditions_A)[1] == length(conditions_b) "the conditions variable's dimentions do not match :: size(conditions_A) = $(size(conditions_A)), length(conditions_b) = $(length(conditions_b))"
    @assert size(object)[2] == size(conditions_A)[2] "the conditions variable's dimentions do not match :: size(object) = $(size(object)), size(conditions_A) = $(size(conditions_A))"

    dual_object       = -Matrix(conditions_b')
    dual_conditions_A = -Matrix(conditions_A')
    dual_conditions_b = -(object')[:]

    dual_format       = 
            if format == Standard
                Standard
            elseif format == No_nonNegative
                Equality
            elseif format == Equality
                No_nonNegative
            end

    return dual_object, dual_conditions_A, dual_conditions_b, dual_format
end

end # module linear_prog_prob
