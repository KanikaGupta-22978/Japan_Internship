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
print(mean(Y))
print(sum(Y))
# print((1-mean(Y))/ mean(Y))

nmd = 5
rpt = 1000
n0 = 200
nss = [1000, 2000, 3000, 5000, 10000]
lns = length(nss)
Betas = fill(NaN, d, lns, rpt, nmd);
nss_star = Array{Int64}(undef, lns, rpt, 3)

@time for rr in 1:rpt
    local X, Y
    (X, Y) = gendat(N, case, beta0)
    fitbeta = estBetas(X, Y, nss, n0)
    m = 0
    Betas[:,:,rr,m+=1] = UniW(X, Y, nss)
    Betas[:,:,rr,m+=1] = Uni(X, Y, nss)[1]
    nss_star[:,rr,1] = Uni(X, Y, nss)[2]
    Betas[:,:,rr,m+=1] = Beta_poi = fitbeta[1]
    Betas[:,:,rr,m+=1] = Beta_slik = fitbeta[2]
    Betas[:,:,rr,m+=1] = Beta_naiv = fitbeta[3]
    nss_star[:,rr,2:3] = fitbeta[end]
end

print("case:$(case)\n N:$(N)\n rpt:$(rpt) \n nss:$(nss)\n")
rec = fill(NaN, nmd, length(nss))
rec_var = fill(NaN, nmd, length(nss))
rec_bias = fill(NaN, nmd, length(nss))

for m in 1:nmd, (idn, n) in enumerate(nss)
    rec[m,idn] = sum(nanmean((Betas[:,idn,:,m] .- beta0).^2, 2))
    rec_var[m,idn] = sum(nanvar(Betas[:,idn,:,m], 2))
    rec_bias[m,idn] = sum((nanmean(Betas[:,idn,:,m], 2) .- beta0).^2)
    fname = "output/csv/case$(case)N$(N)method$(m)n$(n).csv"
    writedlm(fname, Betas[:,idn,:,m])
end

show(stdout, "text/plain", rec)
println("\n")
show(stdout, "text/plain", reshape(mean(nss_star, dims=2), lns, 3))
print("\n", sum(rec))
label = ["uniW" "uniLik" "optW" "optLik" "LCC" "Full"]
pl = plot(nss, log.(rec[1:end,:]'), label=label, lw=2, m=(7,:auto))

fullmse = readdlm("output/full-case$(case).csv")
hline!(log(fullmse), label="Full")


savefig(pl, "output/0case$(case).pdf")
writedlm("output/case$(case).csv", [nss rec'])

print("\n Total time used: $(round(time() - initime)) seconds")
