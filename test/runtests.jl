using LinearCovarianceModels
using LinearAlgebra, Test
using HomotopyContinuation

const HC = HomotopyContinuation
const LC = LinearCovarianceModels

@testset "LinearCovariance.jl" begin
    @testset "vec to sym and back" begin
        v = [1,2,3, 4, 5, 6]
        @test vec_to_sym(v) == [1 2 3; 2 4 5; 3 5 6]
        @test sym_to_vec(vec_to_sym(v)) == v
    end

    @testset "LCModels" begin
        A = toeplitz(3)
        @test A isa LCModel
        x = variables(vec(A.Σ))
        @test A.Σ == [x[1] x[2] x[3]
                    x[2] x[1] x[2]
                    x[3] x[2] x[1]]
        @test dim(A) == 3
        T = tree(4, "{1,2},{1,2,3}")
        @test dim(T) == 7

        Ts = trees(5)
        @test all(isa.(last.(Ts), LCModel))

        D = generic_diagonal(6, 3)
        @test dim(D) == 3
        @test_throws ArgumentError generic_diagonal(6, 0)
        @test_throws ArgumentError generic_diagonal(6, 7)

        D = generic_subspace(6, 4)
        @test dim(D) == 4
        @test_throws ArgumentError generic_subspace(6, binomial(6+1,2)+1)
        @test_throws ArgumentError generic_subspace(6, 0)

        # @test_throws ArgumentError LCModel(toeplitz(3).Σ .^2)

        # throw for non-symmetric input
        @test_throws ArgumentError LCModel([x[1] x[1]; x[2] x[1]])

        # handle special matrix types
        @var θ[1:7]
        M = LCModel(SymTridiagonal(θ[1:4],θ[5:7]))
        @test dim(M) == 7
    end

    @testset "mle system" begin
        @var x y z
        Σ = [x y; y z]
        F = mle_system(LCModel(Σ))
        @test F isa HC.System
    end

    @testset "ml_degree_witness" begin
        Σ = toeplitz(3)
        W = ml_degree_witness(Σ)
        @test model(W) isa LCModel
        @test is_dual(W) == false
        @test ml_degree(W) == 3
        @test length(solutions(W)) == 3
        @test dim(model(W)) == 3
        @test parameters(W) isa AbstractVector
        @test verify(W)
        @test verify(W; trace_tol=1e-6)
    end

    @testset "solve" begin
        S = [4/5 -9/5 -1/25
            -9/5 79/16 25/24
            -1/25 25/24 17/16]
        Σ = toeplitz(3)
        W = ml_degree_witness(Σ)
        crits = critical_points(W, S)
        @test length(crits) == 3
        sort!(crits; by=s -> s[2], rev=true)
        p1,p2,p3 = last.(crits)
        @test p1 == :global_maximum
        @test p2 == :local_maximum
        @test p3 == :saddle_point
        @test crits[1][1] ≈ mle(W, S) atol=1e-8
    end
end
