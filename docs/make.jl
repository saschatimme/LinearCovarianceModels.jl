using Documenter, LinearCovariance

makedocs(
    sitename = "LinearCovarianceModels",
    pages = [
        "Introduction" => "index.md",
        ],
    strict=true)

deploydocs(
    repo = "github.com/saschatimme/LinearCovarianceModels.jl.git"
)
