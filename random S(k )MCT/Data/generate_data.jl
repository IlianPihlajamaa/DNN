using Plots, Dierckx, DelimitedFiles, Printf, Base.Threads

include("PercusYevick.jl")
include("ModeCouplingTimeScheme.jl")
cd(@__DIR__)
function main()
    Nk = 100
    k_array = collect(LinRange(0.2, 39.8, Nk))
    open("k_array.txt", "w") do io
        writedlm(io, k_array)
    end


    Nρ = 1
    ρ_array = 1.5#rand(Nρ) .+ 1 

    Nt = 200
    i = 0
    begin_time = time()
    for ρ in ρ_array
        i += 1
        log_time_sample_array = collect(LinRange(-6,10,Nt))#rand(Nt).*16 .- 6
        k_sample_array = [7.0 for i = 1:Nt]#rand(Nt)*39.6 .+ 0.2
        Cₖ = find_random_C_k(k_array, ρ)
        Sₖ = find_analytical_S_k(k_array, ρ, Cₖ)
        t_array, ϕ = find_intermediate_scattering_function(ρ, k_array, Cₖ)
        display(plot(log10.(t_array), ϕ[16, :], ylims=(0,1)))
        if any(isnan, ϕ)
            println("It contained NaNs, throwing away....")
            continue
        end
        log_time_array = log10.(t_array)
        interpolation_spline = Spline2D(k_array, log_time_array, ϕ)
        ϕ_sample_array = interpolation_spline.(k_sample_array, log_time_sample_array)

        open(@sprintf("Sk_data\\exampleSk_rho_%.10f.txt",ρ), "w") do io
            writedlm(io, Sₖ)
        end

        open(@sprintf("phi_data\\examplephi_rho_%.10f.txt",ρ), "w") do io
            writedlm(io, [k_sample_array log_time_sample_array ϕ_sample_array])
        end
        println("Generated the $(i)th state point at ρ = $(round(ρ,digits=2)), estimated time = $( round( (time()-begin_time) / i * (Nρ-i)) ) seconds")

    end
end
main()
    ########################################
