module LinearCovariance

export mle_system, vec_to_sym, sym_to_vec, hankel_matrix,
    mle_system_and_start_pair, tree, trees, generic_subspace,
    dual_mle_system, dual_mle_system_and_start_pair, toeplitz, mle_degree

using LinearAlgebra

import HomotopyContinuation
import DynamicPolynomials

const HC = HomotopyContinuation
const DP = DynamicPolynomials

include("tree_data.jl")

"""
    linear_system(f::Vector{<:DP.AbstractPolynomialLike})

Given a polynomial system which represents a linear system ``Ax=b`` return
`A` and `b`. If `f ` is not a linear system `nothing` is returned.
"""
function linear_system(f::Vector{<:DP.AbstractPolynomialLike})
    n = DP.nvariables(f)
    vars = DP.variables(f)
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
rand_pos_def(n) = outer(randn(n, n))

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
    A, b = linear_system(DP.subs.(system[1:length(x₀)], Ref(vars => x₀)))
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
    A, b = linear_system(DP.subs.(system[1:length(x₀)], Ref(vars => x₀)))
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

Get the covariance matrix corresponding to the tree with the given `id`.
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
    for data in TREE_DATA
        if data.n == n && data.id == id
            return data.tree
        end
    end
    nothing
end

"""
    trees(n)

Return all trees with `n` leaves as a tuple (id, tree).
"""
trees(n::Int) = map(d -> (id=d.id, tree=d.tree), filter(d -> d.n == n, TREE_DATA))


"""
    generic_subspace(n::Integer, m::Integer)

Generate a generic family of symmetric ``n×n`` matrices living in an ``m``-dimensional
subspace.
"""
function generic_subspace(n::Integer, m::Integer)
    m ≤ binomial(n+1,2) || throw(ArgumentError("`m=$m` is larger than the dimension of the space."))
    DP.@polyvar θ[1:m]
    sum(θᵢ .* rand_pos_def(n) for θᵢ in θ)
end


function mle_degree(Σ; max_tries = 5)
    F, x₀, p₀, x, p = mle_system_and_start_pair(Σ)
    result = HC.monodromy_solve(F, x₀, p₀; parameters=p, max_loops_no_progress=5)
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

    if result_agreed
        println("\nMLE degree: ", HC.nsolutions(best_result))
        return HC.nsolutions(best_result), best_result
    else
        return nothing, best_result
    end
end



function get_basis(Σ)
    vars = variables(vec(Σ))
    map(1:length(vars)) do i
        [p(vars[i] => 1,
           vars[1:i-1]=>zeros(Int, max(i-1,0)),
           vars[i+1:end]=>zeros(Int, max(length(vars)-i,0))) for p in Σ]
    end
end


function logl(B, θ, S::AbstractMatrix)
    Σ = sum(θ[i] * B[i] for i in 1:length(B))
    -log(det(Σ)) - tr(S*inv(Σ))
end

function gradient_logl(B, θ, S::AbstractMatrix)
    Σ = sum(θ[i] * B[i]' for i in 1:length(B))
    Σ⁻¹ = inv(Σ)
    map(1:length(B)) do i
        -tr(Σ⁻¹ * B[i]) + tr(S *  Σ⁻¹ * B[i] * Σ⁻¹)
    end
end

function hessian_logl(B, θ, S::AbstractMatrix)
    m = length(B)
    Σ = sum(θ[i] * B[i] for i in 1:m)
    Σ⁻¹ = inv(Σ)
    H = zeros(eltype(Σ), m,m)
    for i in 1:m, j in i:m
        kernel = Σ⁻¹ * B[i] * Σ⁻¹ * B[j]
        H[i,j] = H[j,i] = tr(kernel) - 2tr(S * kernel * Σ⁻¹)
    end
    Symmetric(H)
end


end # module
