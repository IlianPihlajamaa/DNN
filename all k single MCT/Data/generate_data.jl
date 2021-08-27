using Plots, Dierckx, DelimitedFiles, Printf

include("PercusYevick.jl")
include("ModeCouplingTimeScheme.jl")
cd(@__DIR__)
function main()
    Nk = 100
    k_array = collect(LinRange(0.2, 39.8, Nk))
    open("k_array.txt", "w") do io
        writedlm(io, k_array)
    end

    η_crit = 0.5159121289383616

    Nη = 1000
    σ_η = 0.04
    η_array = σ_η * randn(Nη) .+ η_crit

    println("Generated $Nη values of the volume fraction between η = $(minimum(η_array)) and η = $(maximum(η_array))")


    Nt = 1000
    i = 0
    k_array = collect(LinRange(0.2, 39.8, 100))
    for η in η_array
        i += 1
        log_time_sample_array = rand(Nt).*16 .- 6
        k_sample_array = rand(Nt)*39.6 .+ 0.2
        Sk = find_analytical_S_k.(k_array, η)
        t_array, ϕ = find_intermediate_scattering_function(η)
        log_time_array = log10.(t_array)
        interpolation_spline = Spline2D(k_array, log_time_array, ϕ)
        ϕ_sample_array = interpolation_spline.(k_sample_array, log_time_sample_array)



        open(@sprintf("Sk_data\\Sk_eta_%.10f.txt",η), "w") do io
            writedlm(io, Sk)
        end

        open(@sprintf("phi_data\\phi_eta_%.10f.txt",η), "w") do io
            writedlm(io, [k_sample_array log_time_sample_array ϕ_sample_array])
        end
    end
end
main()
    ########################################
