using LinearCovariance


for n=4:5, m = 1:binomial(n+1,2)
    println("\n\nn: ", n, " m: ", m)
    Σ = generic_subspace(n, m)
    res = mle_degree(Σ)
end
