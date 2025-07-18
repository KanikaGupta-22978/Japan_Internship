using Plots, DelimitedFiles, LaTeXStrings
# Plots.scalefontsizes(1.8)

N = 10^6 ÷ 2
ncase = 1:4
label = ["uniW" "uniLik" "optW" "optLik" "LCC" "Full"]
tp = 1:5

for (i, case) in enumerate(ncase)
    rs = readdlm("output/case$(case).csv")# [1:end-1,:]
    rs[:,2:end] = log.(rs[:,2:end])
    plc = plot(rs[:,1]./N, rs[:,tp.+1], # size = 1.2 .*(400, 600),
              label=label[:,tp], lw=3, m=(9,:auto),
              tickfontsize=16, xguidefontsize=18, yguidefontsize=18,
              legendfontsize=14, grid=false, thickness_scaling=1,
              xlabel="sampling rate", ylabel="log(MSE)"# , legend=:topright
              ,legend= case == Inf ? :topright : false
               )
    annotate!(0.01, sum(extrema(rs[:,tp.+1]) .* [0.05, 0.95]),
              text("Misspecified Pilot", :center, 16))
    fullmse = readdlm("output/full-case$(case).csv")
    hline!(log(fullmse), label="Full")
    savefig(plc, "output/0case$(case).pdf")
end

# pdfs = "output/0case" .* map(string, ncase) .* ".pdf"
# onefile = "00mseMisPilot.pdf"
# run(`pdftk $pdfs output output/$(onefile)`)
