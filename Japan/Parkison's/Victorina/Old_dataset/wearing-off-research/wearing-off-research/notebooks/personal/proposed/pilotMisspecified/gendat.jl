function gendat(n, case, beta0)
    d = length(beta0)
    ds = d - 1
    corr  = 0.5
    sigmax = corr.^abs.((1:ds) .- (1:ds)')
    sigmax = sigmax ./ 4
    if case == 1 # Normal
        Z = rand(MvNormal(zeros(ds), sigmax), n)'
    elseif case == 2 # lognormal, inbalanced
        Z = exp.(rand(MvNormal(zeros(ds), sigmax), n)')
    elseif case == 3 # T3
        Z = rand(MvNormal(zeros(ds), sigmax), n);
        df = 3
        Z = collect(Z') ./ sqrt.(rand(Chisq(df), n)./df) ./ 3
    elseif case == 4 # exponential
        Z = randexp(n, ds)
    elseif case == 5 # uniform
        Z = rand(n, ds)
    end
    # X = [ones(n) Z]
    P = 1 .- 1 ./ (1 .+ exp.(beta0[1] .+ vec(Z * beta0[2:end])))
    Y = [rand(Bernoulli(P[i])) for i in 1:n]
    return [ones(n) Z], Y
end
