function getMSLE(x, y, π, pilot)
    (n, d) = size(x)
    beta = pilot
    S = Array{Float64}(undef, n, d)
    H = Array{Float64}(undef, d, d)
    loop = 1;
    Loop = 100;
    msg = "NA";
    while loop <= Loop
        p = 1 ./ (1 .+ exp.(-vec(x * beta)).*π)
        S = (y .- p) .* x
        phi = p .* (1 .- p)
        H = x' * (phi .* x)
        ss = sum(S, dims=1)
        shs = try
            H \ ss'
        catch
            msg = "H is singular"; println(msg)
            beta = fill(NaN, d)
            break
        end
        beta_new = beta .+ 0.5shs
        tlr  = sum(shs.^2)
        beta = beta_new
        if tlr < 0.00001
            msg = "Successful convergence"
            break
        end
        if loop == Loop
            msg = "Maximum iteration reached"; println(msg)
            beta = fill(NaN, d)
            break
        end
        loop += 1
    end
    return beta, msg, loop, H, S'S
end

# getEst
function getEst(x, y; w=ones(Float64, length(y)))
    (n, d) = size(x)
    beta = zeros(d)
    S = Array{Float64}(undef, n, d)
    H = Array{Float64}(undef, d, d)
    loop = 1;
    Loop = 100;
    msg = "NA";
    wx = w .* x
    while loop <= Loop
        p = 1 ./ (1 .+ exp.(-vec(x * beta)))
        S = (y .- p) .* wx
        H = x' * (p .* (1 .- p) .* wx)
        ss = sum(S, dims=1)
        shs = try
            (H\ss')
        catch
            msg = "H is singular"; println(msg)
            beta = fill(NaN, d)
            break
        end
        beta_new = beta .+ 0.5shs
        tlr  = sum(shs.^2)
        beta = beta_new
        if tlr < 0.000001
            msg = "Successful convergence"
            break
        end
        if loop == Loop
            msg = "Maximum iteration reached"; println(msg)
            beta = fill(NaN, d)
            break
        end
        loop += 1
    end
    return vec(beta), msg, loop, H, S'S
end

function calPI(X, Y, n0)
    (N,d) = size(X)
    N1 = sum(Y)
    loc0 = Y.==0
    N0 = sum(loc0)
    cc = fill(n0 ./ 2N1, N)
    cc[Y.==0] .= n0 ./ 2N0
    idx_plt = rand(N) .<= cc
    x_plt = X[idx_plt, :]
    y_plt = Y[idx_plt]
    cc_plt = cc[idx_plt]
    beta_plt, msg, loop, ddm_plt, F = getEst(x_plt, y_plt)
    beta_plt[1] -= log(N0/N1)
    beta_plt .+= 1.5rand(d)
    P_plt = 1 ./ (1 .+ exp.(-vec(X * beta_plt)))
    dm = P_plt .* sqrt.(1 .- P_plt) .* sqrt.(vec(sum((X/ddm_plt).^2, dims=2)))
    pi_P = dm ./ (N0 * sum(dm[idx_plt]./cc_plt)/N) # / (1-d/n0)

    dm_lcc = abs.(Y .- P_plt)
    pi_Plcc = dm_lcc ./ sum(dm_lcc[idx_plt]./cc_plt)

    return beta_plt, ddm_plt, pi_P, pi_Plcc
end
# calPI(X, Y, 100)[3]

# different estimation methods
function estBetas(X, Y, nss, n0)
    (N,d) = size(X)
    N1 = sum(Y)
    lns = length(nss)
    Betas_popt = fill(NaN, d, lns);
    Betas_slik = fill(NaN, d, lns);
    Betas_naiv = fill(NaN, d, lns);
    n_star = Array{Int64}(undef, lns, 2)
    beta_plt, ddm_plt, pi_P, pi_Plcc= calPI(X, Y, n0)
    for (idn, n) in enumerate(nss)
        idx = rand(N) .<= Y .+ (1 .- Y) .* n .* pi_P
        n_star[idn,1] = sum(idx[Y.==0])
        x = X[idx, :]
        y = Y[idx]
        π = min.(n .* pi_P[idx], 1)
        πy = y .+ (1 .- y) .* π
        Betas_popt[:,idn] = getEst(x, y, w=1 ./ πy)[1]
        Betas_slik[:,idn] = getMSLE(x, y, π, beta_plt)[1]
        # LCC
        PLCC = (n+N1).*pi_Plcc
        idx_lcc = rand(N) .<= PLCC
        n_star[idn,2] = sum(idx_lcc[Y.==0])
        x_lcc = X[idx_lcc, :]
        y_lcc = Y[idx_lcc]
        w_lcc = max.(1, PLCC)[idx_lcc]
        Betas_naiv[:,idn] = getEst(x_lcc, y_lcc, w = w_lcc)[1] .+ beta_plt
    end
    return Betas_popt, Betas_slik, Betas_naiv, n_star
end

# Uniform sampling
function Uni(X, Y, nss)
    (N,d) = size(X)
    loc0 = Y.==0
    N0 = sum(loc0)
    lns = length(nss)
    Betas = fill(NaN, d, lns)
    n_star = Array{Int64,1}(undef, lns)
    for (idn, n) in enumerate(nss)
        u = rand(N)
        pi_uni = ones(N)
        pi_uni[loc0] .= n/N0
        idx_uni = u .<= pi_uni
        x_uni = X[idx_uni, :]
        y_uni = Y[idx_uni]
        Betas[:,idn] = getEst(x_uni, y_uni)[1]
        n_star[idn] = sum(idx_uni[loc0])
    end
    Betas[1,:] .+= log.(nss ./ N0)
    return Betas, n_star
end

# Uniform sampling
function UniW(X, Y, nss)
    (N,d) = size(X)
    loc0 = Y.==0
    N0 = sum(loc0)
    lns = length(nss)
    Betas = fill(NaN, d, lns)
    for (idn, n) in enumerate(nss)
        u = rand(N)
        pi_uni = ones(N)
        pi_uni[loc0] .= n/N0
        idx_uni = u .<= pi_uni
        x_uni = X[idx_uni, :]
        y_uni = Y[idx_uni]
        w_uni = 1 ./ pi_uni[idx_uni]
        Betas[:,idn] = getEst(x_uni, y_uni, w=w_uni)[1]
    end
    return Betas
end

nanmean(x) = mean(filter(!isnan, x))
nanmean(x,y) = mapslices(nanmean, x, dims=y)
nanvar(x) = var(filter(!isnan, x))
nanvar(x,y) = mapslices(nanvar, x, dims=y)
