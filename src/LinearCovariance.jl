module LinearCovariance

export ml_degree_witness, MLDegreeWitness, MLModel,
    model, parameters, solutions, is_dual,
    # families
    generic_subspace, generic_diagonal, toeplitz, tree, trees,
    # solve specific instance
    mle, solve, covariance_matrix, logl, gradient_logl, hessian_logl, classify_point


using LinearAlgebra

import HomotopyContinuation
import DynamicPolynomials
import Distributions: Normal

const HC = HomotopyContinuation
const DP = DynamicPolynomials

include("tree_data.jl")

"""
    linear_system(f::Vector{<:DP.AbstractPolynomialLike})

Given a polynomial system which represents a linear system ``Ax=b`` return
`A` and `b`. If `f ` is not a linear system `nothing` is returned.
"""
function linear_system(f::Vector{<:DP.AbstractPolynomialLike}, vars = DP.variables(f))
    n = length(vars)
    A = zeros(DP.coefficienttype(f[1]), length(f), n)
    b = zeros(eltype(A), length(f))
    for (i, fᵢ) in enumerate(f)
        for t in DP.terms(fᵢ)
            constant = true
            for (j, v) in enumerate(vars)
                d = DP.degree(t, v)
                d ≤ 1 || return nothing

                if d == 1
                    A[i,j] = DP.coefficient(t)
                    constant = false
                    break
                end
            end
            if constant
                b[i] = -DP.coefficient(t)
            end
        end
    end
    A, b
end

outer(A) = A*A'

"""
    rand_pos_def(n)

Create a random positive definite `n × n` matrix. The matrix is generated
by first creating a `n × n` matrix `X` where each entry is independently drawn
from the `Normal(μ=0, σ²=0.5)` distribution. Then `X*X'` is returned.
"""
rand_pos_def(n) = outer(rand(Normal(0, 0.5), n, n))

n_vec_to_sym(k) = div(-1 + round(Int, sqrt(1+8k)), 2)
n_sym_to_vec(n) = binomial(n+1,2)

"""
    vec_to_sym(v)

Converts a vector `v` to a symmetrix matrix by filling the lower triangular
part columnwise.

### Example
```
julia> v = [1,2,3, 4, 5, 6];
julia> vec_to_sym(v)
3×3 Array{Int64,2}:
 1  2  3
 2  4  5
 3  5  6
 ```
"""
function vec_to_sym(s)
    n = n_vec_to_sym(length(s))
    S = Matrix{eltype(s)}(undef, n, n)
    l = 1
    for i in 1:n, j in i:n
        S[i,j] = S[j,i] = s[l]
        l += 1
    end
    S
end

"""
    sym_to_vec(S)

Converts a symmetric matrix `S` to a vector by filling the vector with lower triangular
part iterating columnwise.
"""
sym_to_vec(S) = (n = size(S, 1); [S[i,j] for i in 1:n for j in i:n])


"""
    mle_system(Σ::Matrix{<:AbstractPolynomialLike})

Generate the MLE system corresponding to the family of covariances matrices
parameterized by `Σ`.
Returns the named tuple `(system, variables, parameters)`.
"""
function mle_system(Σ::Matrix{<:DP.AbstractPolynomialLike})
    θ = DP.variables(vec(Σ))
    m = DP.nvariables(θ)
    n = size(Σ, 1)
    N = binomial(n+1,2)

    DP.@polyvar k[1:N] s[1:N]

    K, S = vec_to_sym(k), vec_to_sym(s)
    l = -tr(K * Σ) + tr(S * K * Σ * K)
    ∇l = DP.differentiate(l, θ)
    KΣ_I = vec(K * Σ - Matrix(I, n,n))
    (system=[∇l; KΣ_I], variables=[θ; k], parameters=s)
end

"""
    dual_mle_system(Σ::Matrix{<:AbstractPolynomialLike})

Generate the dual MLE system corresponding to the family of covariances matrices
parameterized by `Σ`.
Returns the named tuple `(system, variables, parameters)`.
"""
function dual_mle_system(Σ::Matrix{<:DP.AbstractPolynomialLike})
    θ = DP.variables(vec(Σ))
    m = DP.nvariables(θ)
    n = size(Σ, 1)
    N = binomial(n+1,2)

    DP.@polyvar k[1:N] s[1:N]

    K, S = vec_to_sym(k), vec_to_sym(s)
    l = -tr(K * Σ) + tr(S * Σ)
    ∇l = DP.differentiate(l, θ)
    KΣ_I = vec(K * Σ - Matrix(I, n,n))
    (system=[∇l; KΣ_I], variables=[θ; k], parameters=s)
end

"""
    mle_system_and_start_pair(Σ::Matrix{<:DP.AbstractPolynomialLike})

Generate the mle_system and a corresponding start pair `(x₀,p₀)`.
"""
function mle_system_and_start_pair(Σ::Matrix{<:DP.AbstractPolynomialLike})
    system, vars, params = mle_system(Σ)
    θ = DP.variables(vec(Σ))
    θ₀ = randn(ComplexF64, length(θ))
    Σ₀ = [p(θ => θ₀) for p in Σ]
    K₀ = inv(Σ₀)
    x₀ = [θ₀; sym_to_vec(K₀)]
    A, b = linear_system(DP.subs.(system[1:length(x₀)], Ref(vars => x₀)), params)
    p₀ = A \ b

    (system=system, x₀=x₀, p₀=p₀, variables=vars, parameters=params)
end

"""
    dual_mle_system_and_start_pair(Σ::Matrix{<:DP.AbstractPolynomialLike})

Generate the dual MLE system and a corresponding start pair `(x₀,p₀)`.
"""
function dual_mle_system_and_start_pair(Σ::Matrix{<:DP.AbstractPolynomialLike})
    system, vars, params = dual_mle_system(Σ)
    θ = DP.variables(vec(Σ))
    θ₀ = randn(ComplexF64, length(θ))
    Σ₀ = [p(θ => θ₀) for p in Σ]
    K₀ = inv(Σ₀)
    x₀ = [θ₀; sym_to_vec(K₀)]
    A, b = linear_system(DP.subs.(system[1:length(x₀)], Ref(vars => x₀)), params)
    p₀ = A \ b

    (system=system, x₀=x₀, p₀=p₀, variables=vars, parameters=params)
end

"""
    toeplitz(n::Integer)

Returns a symmetric `n×n` toeplitz matrix.
"""
function toeplitz(n::Integer)
    DP.@polyvar θ[1:n]
    sum(0:n-1) do i
        if i == 0
            θ[1] .* diagm(0 => ones(n))
        else
            θ[i+1] .* (diagm(i => ones(n-i)) + diagm(-i => ones(n-i)))
        end
    end
end

"""
    hankel_matrix(n::Integer)

Generate a `n×n` hankel matrix.
"""
function hankel_matrix(n::Integer)
    DP.@polyvar θ[1:2n-1]
    A = Matrix{DP.Polynomial{true,Int64}}(undef, n, n)
    for j in 1:n, i in 1:j
        A[i,j-i+1] = θ[j]
    end
    for k in 2:n
        for i in k:n
            A[i, n-i+k] = θ[k+n-1]
        end
    end
    A
end

"""
    tree(n, id::String)

Get the covariance matrix corresponding to the tree with the given `id` on `n` leaves.
Returns `nothing` if the tree was not found.

## Example
```
julia> tree(4, "{{1, 2}, {3, 4}}")
4×4 Array{PolyVar{true},2}:
 t₁  t₅  t₇  t₇
 t₅  t₂  t₇  t₇
 t₇  t₇  t₃  t₆
 t₇  t₇  t₆  t₄
 ```
"""
function tree(n::Integer, id::String)
    4 ≤ n ≤ 7 || throw(ArgumentError("Only trees with 4 to 7 leaves are supported."))
    for data in TREE_DATA
        if data.n == n && data.id == id
            return make_tree(data.tree)
        end
    end
    nothing
end

function make_tree(tree::Matrix{Symbol})
    var_names = sort(unique(vec(tree)))
    D = Dict(map(v -> (v, DP.PolyVar{true}(String(v))), var_names))
    map(v -> D[v], tree)
end

"""
    trees(n)

Return all trees with `n` leaves as a tuple (id, tree).
"""
function trees(n::Int)
    4 ≤ n ≤ 7 || throw(ArgumentError("Only trees with 4 to 7 leaves are supported."))
    map(d -> (id=d.id, tree=make_tree(d.tree)), filter(d -> d.n == n, TREE_DATA))
end


"""
    generic_subspace(n::Integer, m::Integer)

Generate a generic family of symmetric ``n×n`` matrices living in an ``m``-dimensional
subspace.
"""
function generic_subspace(n::Integer, m::Integer)
    m ≤ binomial(n+1,2) || throw(ArgumentError("`m=$m` is larger than the dimension of the space."))
    DP.@polyvar θ[1:m]
    return sum(θᵢ .* rand_pos_def(n) for θᵢ in θ)
end

"""
    generic_diagonal(n::Integer, m::Integer)

"""
function generic_diagonal(n::Integer, m::Integer)
    m ≤ n || throw(ArgumentError("`m=$m` is larger than the dimension of the space."))
    DP.@polyvar θ[1:m]
    sum(θᵢ .* diagm(0 => randn(n)) for θᵢ in θ)
end

struct MLModel{T1<:DP.AbstractPolynomialLike, T2<:Number}
    Σ::Matrix{T1}
    B::Vector{Matrix{T2}}
end
MLModel(Σ) = MLModel(Σ, get_basis(Σ))

function get_basis(Σ)
    vars = DP.variables(vec(Σ))
    map(1:length(vars)) do i
        [p(vars[i] => 1,
           vars[1:i-1]=>zeros(Int, max(i-1,0)),
           vars[i+1:end]=>zeros(Int, max(length(vars)-i,0))) for p in Σ]
    end
end

Base.size(M::MLModel) = (size(M.Σ, 1), length(M.B))
Base.size(M::MLModel, i::Int) = size(M)[i]
Base.show(io::IO, M::MLModel) = Base.print_matrix(io, M.Σ)
Base.broadcastable(M::MLModel) = Ref(M)


"""
    MLDegreeWitness

Data structure holding an MLE model. This also holds a set of solutions for a generic instance,
which we call a witness.
"""
struct MLDegreeWitness{T1, T2, V<:AbstractVector}
    model::MLModel{T1,T2}
    solutions::Vector{V}
    p::Vector{ComplexF64}
    dual::Bool
end


function MLDegreeWitness(Σ::AbstractMatrix, solutions, p, dual)
    MLDegreeWitness(MLModel(Σ), solutions, p, dual)
end

function Base.show(io::IO, R::MLDegreeWitness)
    println("MLDegreeWitness:")
    println(" • ML degree → $(length(R.solutions))")
    println(" • model dimension → $(size(R.model, 2))")
    println(" • dual → $(R.dual)")
end

"""
    model(W::MLDegreeWitness)

Obtain the model corresponding to the `MLDegreeWitness` `W`.
"""
model(R::MLDegreeWitness) = R.model

"""
    solutions(W::MLDegreeWitness)

Obtain the witness solutions corresponding to the `MLDegreeWitness` `W`
with given parameters.
"""
solutions(W::MLDegreeWitness) = W.solutions

"""
    parameters(W::MLDegreeWitness)

Obtain the parameters of the `MLDegreeWitness` `W`.
"""
parameters(W::MLDegreeWitness) = W.p

"""
    is_dual(W::MLDegreeWitness)

Indicates whether `W` is a witness for the dual MLE.
"""
is_dual(W::MLDegreeWitness) = W.dual

"""
    ml_degree_witness(Σ::AbstractMatrix; ml_degree=nothing, max_tries=5, dual=false)

Compute a [`MLDegreeWitness`](@ref) for a given model Σ. If the ML degree is already
known it can be provided to stop the computations early. The stopping criterion is based
on a heuristic, `max_tries` indicates how many different parameters are tried a most until
an agreement is found.
"""
function ml_degree_witness(Σ; ml_degree=nothing, max_tries = 5, dual=false)
    if dual
        F, x₀, p₀, x, p = dual_mle_system_and_start_pair(Σ)
    else
        F, x₀, p₀, x, p = mle_system_and_start_pair(Σ)
    end
    result = HC.monodromy_solve(F, x₀, p₀; target_solutions_count=ml_degree,
                                            parameters=p, max_loops_no_progress=5)
    if HC.nsolutions(result) == ml_degree
        return MLDegreeWitness(Σ, HC.solutions(result), result.parameters, dual)
    end

    best_result = result
    result_agreed = false
    for i in 1:max_tries
        q₀ = randn(ComplexF64, length(p₀))
        S_q₀ = HC.solutions(HC.solve(F, HC.solutions(result); parameters=p, start_parameters=p₀, target_parameters=q₀))
        new_result = HC.monodromy_solve(F, S_q₀, q₀; parameters=p, max_loops_no_progress=3)
        if HC.nsolutions(new_result) == HC.nsolutions(best_result)
            result_agreed = true
            break
        elseif HC.nsolutions(new_result) > HC.nsolutions(best_result)
            best_result = new_result
        end
    end
    MLDegreeWitness(Σ, HC.solutions(best_result), best_result.parameters, dual)
end

function verify(W::MLDegreeWitness)
    if W.dual
        F, var, params = dual_mle_system(model(W).Σ)
    else
        F, var, params = mle_system(model(W).Σ)
    end
    HC.verify_solution_completeness(F, solutions(W), parameters(W); parameters=params)
end

"""
    solve(W::MLDegreeWitness, S::AbstractMatrix; kwargs...)

Compute all solutions to the MLE problem of `W` for the given sample covariance matrix
`S`.
"""
function solve(W::MLDegreeWitness, S::AbstractMatrix; kwargs...)
    issymmetric(S) || throw("Sample covariance matrix `S` is not symmetric. Consider wrapping it in `Symmetric(S)` to enforce symmetry.")
    if W.dual
        F, var, params = dual_mle_system(model(W).Σ)
    else
        F, var, params = mle_system(model(W).Σ)
    end
    result = HC.solve(F, solutions(W); parameters=params,
                         start_parameters=W.p,
                         target_parameters=sym_to_vec(S),
                         kwargs...)

    m = size(model(W), 2)
    map(s -> s[1:m], HC.real_solutions(result))
end

"""
    covariance_matrix(M::MLModel, θ)

Compute the covariance matrix corresponding to the value of `θ` and the given model
`M`.
"""
covariance_matrix(W::MLDegreeWitness, θ) = covariance_matrix(model(W), θ)
covariance_matrix(M::MLModel, θ) = sum(θ[i] * M.B[i] for i in 1:size(M,2))


"""
    logl(M::MLModel, θ, S::AbstractMatrix)

Evaluate the log-likelihood ``log(det(Σ⁻¹)) - tr(SΣ⁻¹)`` of the MLE problem.
"""
function logl(M::MLModel, θ, S::AbstractMatrix)
    logl(covariance_matrix(M, θ), S)
end
logl(Σ::AbstractMatrix, S::AbstractMatrix) = -logdet(Σ) - tr(S*inv(Σ))

"""
    gradient_logl(M::MLModel, θ, S::AbstractMatrix)

Evaluate the gradient of the log-likelihood ``log(det(Σ⁻¹)) - tr(SΣ⁻¹)`` of the MLE problem.
"""
gradient_logl(M::MLModel, θ, S::AbstractMatrix) = gradient_logl(M.B, θ, S)
function gradient_logl(B::Vector{<:Matrix}, θ, S::AbstractMatrix)
    Σ = sum(θ[i] * B[i] for i in 1:length(B))
    Σ⁻¹ = inv(Σ)
    map(1:length(B)) do i
        -tr(Σ⁻¹ * B[i]) + tr(S *  Σ⁻¹ * B[i] * Σ⁻¹)
    end
end

"""
    hessian_logl(M::MLModel, θ, S::AbstractMatrix)

Evaluate the hessian of the log-likelihood ``log(det(Σ⁻¹)) - tr(SΣ⁻¹)`` of the MLE problem.
"""
hessian_logl(M::MLModel, θ, S::AbstractMatrix) = hessian_logl(M.B, θ, S)
function hessian_logl(B::Vector{<:Matrix}, θ, S::AbstractMatrix)
    m = length(B)
    Σ = sum(θ[i] * B[i] for i in 1:m)
    Σ⁻¹ = inv(Σ)
    H = zeros(eltype(Σ), m, m)
    for i in 1:m, j in i:m
        kernel = Σ⁻¹ * B[i] * Σ⁻¹ * B[j]
        H[i,j] = H[j,i] = tr(kernel) - 2tr(S * kernel * Σ⁻¹)
    end
    Symmetric(H)
end

"""
    classify_point(M::MLModel, θ, S::AbstractMatrix)

Classify the critical point `θ` of the log-likelihood function.
"""
function classify_point(M::MLModel, θ, S::AbstractMatrix)
    H = hessian_logl(M, θ, S)
    emin, emax = extrema(eigvals(H))
    if emin < 0 && emax < 0
        :local_maximum
    elseif emin > 0 && emax > 0
        :local_minimum
    else
        :saddle_point
    end
end


"""
    mle(W::MLDegreeWitness, S::AbstractMatrix; only_positive_definite=true, only_positive=false)

Compute the MLE for the matrix `S` using the MLDegreeWitness `W`.
Returns the parameters for the MLE covariance matrix or `nothing` if no solution was found
satisfying the constraints (see options below).

## Options

* `only_positive_definite`: controls whether only positive definite
covariance matrices should be considered.
* `only_positive`: controls whether only (entrywise) positive covariance matrices
should be considered.
"""
function mle(W::MLDegreeWitness, S::AbstractMatrix; only_positive_definite=true, only_positive=false, kwargs...)
    is_dual(W) && throw(ArgumentError("`mle` is currently only supported for MLE not dual MLE."))
    θs = solve(W, S; kwargs...)
    if only_positive_definite
        filter!(θs) do θ
            isposdef(covariance_matrix(model(W), θ))
        end
    end

    if only_positive
        filter!(θs) do θ
            all(covariance_matrix(model(W), θ) .> 0)
        end
    end

    filter!(θs) do θ
        classify_point(model(W), θ, S) == :local_maximum
    end
    sort!(θs; by=θ -> logl(model(W), θ, S), rev=true)

    if isempty(θs)
        return nothing
    end
    return first(θs)
end

end # module
