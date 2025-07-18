initime = time()
using DelimitedFiles, Random, Statistics, LinearAlgebra
using Distributions, Plots, LaTeXStrings # using DataFrames 
# Threads.nthreads()
BLAS.set_num_threads(1)
# ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())

if "output" ∉ readdir() mkdir("output") end
if "csv" ∉ readdir("./output/") mkdir("./output/csv") end
include("gendat.jl")
include("estimators.jl")

case = haskey(ENV, "case") ? parse(Int, ENV["case"]) : 3
Random.seed!(1);
N = 10^6 ÷ 2
beta0 = [NaN; -ones(6)] #c1 c3 -7
if case == 1
    beta0[1] = -7.65
    # beta0[1] = -6
elseif case == 2
    beta0[1] = -0.5
    # beta0[1] = 1
elseif case == 3
    beta0[1] = -7
    # beta0[1] = -5
elseif case == 4
    beta0[1] = -1.8
    # beta0[1] = -0.14
elseif case == 5
    beta0[1] = -3.2
end
d = length(beta0)
(X, Y) = gendat(N, case, beta0);
print((mean(Y), sum(Y)))
# print((1-mean(Y))/ mean(Y))

rpt = 1000
Betas = fill(NaN, d, rpt);

@time for rr in 1:rpt
    local X, Y
    (X, Y) = gendat(N, case, beta0)
    Betas[:,rr] = getEst(X,Y)[1]
end

rec = sum(nanmean((Betas .- beta0).^2, 2))
rec_var = sum(nanvar(Betas, 2))
rec_bias = sum((nanmean(Betas, 2) .- beta0).^2)
writedlm("output/csv/Full-case$(case)N$(N).csv", Betas)
writedlm("output/full-case$(case).csv", rec)

print("\n Total time used: $(round(time() - initime)) seconds")
