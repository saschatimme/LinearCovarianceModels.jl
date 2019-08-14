using LinearCovarianceModels
using DynamicPolynomials, LinearAlgebra, Test

const LC = LinearCovarianceModels

@testset "LinearCovariance.jl" begin
    # Write your own tests here.
    @testset "linear system" begin
        @polyvar x y z
        f = [2x+3y+z+5, -1x+2y+4z-2]
        A, b = LC.linear_system(f)
        @test A == [2 3 1; -1 2 4]
        @test b == [-5, 2]
        @test A * [x,y,z] - b == f
    end

    @testset "vec to sym and back" begin
        v = [1,2,3, 4, 5, 6]
        @test vec_to_sym(v) == [1 2 3; 2 4 5; 3 5 6]
        @test sym_to_vec(vec_to_sym(v)) == v
    end
    @testset "mle system" begin
        @polyvar x y z
        Σ = [x y; y z]
        F, vars, params = mle_system(Σ)
        Σ₀ = [2. 3; 3 -5]
        K₀ = inv(Σ₀)
        F₀ = [subs(f, vars=>[[2, 3, -5]; sym_to_vec(K₀)]) for f in F]
        @test norm(F₀[end-3:end]) ≈ 0.0 atol=1e-14
        A₀, b₀ = LC.linear_system(F₀[1:3])
        @test norm(A₀ \ b₀ - [2,3,-5]) ≈ 0.0 atol=1e-12
    end

    @testset "toeplitz" begin
        A = toeplitz(3)
        x = variables(vec(A))
        @test A == [x[1] x[2] x[3]
                    x[2] x[1] x[2]
                    x[3] x[2] x[1]]
    end

    @testset "mle_system_and_start_pair" begin
        F, x₀, p₀, x, p = mle_system_and_start_pair(hankel_matrix(4))
        @test norm([f(x=>x₀, p=>p₀) for f in F]) < 1e-14
    end

    @testset "dual_mle_system_and_start_pair" begin
        F, x₀, p₀, x, p = dual_mle_system_and_start_pair(hankel_matrix(4))
        @test norm([f(x=>x₀, p=>p₀) for f in F]) < 1e-14
    end
end
