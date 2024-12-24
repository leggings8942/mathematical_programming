module linear_prog_prob
export simplex

function simplex(object::Matrix{T}, conditions_A::Matrix{T}, conditions_b::Vector{T}) where T <: Union{Int, AbstractFloat}
    @assert size(object)[1] == 1 "the objective variable must be a horizontal vector :: size(object) = $(size(object))"
    @assert size(conditions_A)[1] == length(conditions_b) "the conditions variable's dimentions do not match :: size(conditions_A) = $(size(conditions_A)), length(conditions_b) = $(length(conditions_A))"
    @assert size(object)[2] == size(conditions_A)[2] "the conditions variable's dimentions do not match :: size(object) = $(size(object)), size(conditions_A) = $(size(conditions_A))"

    condits, expvars = size(conditions_A)
    @assert condits < expvars "The number of explanatory variables is greater than the number of conditions."

    N   = conditions_A[1:end, 1:(expvars - condits)]
    B   = conditions_A[1:end, (expvars - condits + 1):end]
    C_N = object[1, 1:(expvars - condits)]
    C_B = object[1, (expvars - condits + 1):end]
    
    X_B = B * conditions_b

    b_dash   = X_B
    C_N_dash = C_N' - C_B' * inv(B) * N
    N_dash   = inv(B) * N

    z = C_B' * b_dash

    display(Text("object ="))
    display(object)
    display(Text("conditions_A ="))
    display(conditions_A)
    display(Text("conditions_b ="))
    display(conditions_b)
    # @show N B C_N C_B
    # @show b_dash C_N_dash N_dash
    # @show z

    return X_B
end

end # module linear_prog_prob
