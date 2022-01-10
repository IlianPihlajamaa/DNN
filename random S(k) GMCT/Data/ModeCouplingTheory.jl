using LoopVectorization, Dierckx, Polyester, BenchmarkTools, Tullio, Base.Threads, IfElse, TupleTools, DelimitedFiles, Printf
cd(@__DIR__)

for file in ["MemoryKernel.jl", "Closures.jl", "PercusYevick.jl", "SymmetricTensors.jl",  "NonErgodicityParameter.jl", "TimeStepping.jl"]
    include(file)
end
using Roots

function find_nonergodicity_parameter(Cₖ, ρ, D₀, k_array, order)
    return find_non_ergodicity_parameter(Cₖ, k_array, ρ, D₀, order; tolerance=10^-6, max_iterations=10^6)
end


function main(order)
    println("\n\n order = $order \n\n")
    println("Number of threads = ", nthreads())
    Nₖ = 64
    kmax = 40
    Δk = kmax/Nₖ
    k_array = Δk*(collect(1:Nₖ) .- 0.5)
    Cₖ = find_random_C_k(k_array)
    D₀ = 1.0
    xatol = 10^-4

    f(ρ) = sum(find_nonergodicity_parameter(Cₖ, ρ, D₀, k_array, order)) - 0.3
    function findbrackets()
        samples = LinRange(0.1, 10, 50)
        res = zeros(length(samples))
        for (i,ρ) in enumerate(samples)
            res[i] = f(ρ)
            if i > 1
                if res[i]*res[i-1] < 0 
                    return samples[i-1], samples[i]
                end
            end
        end

        throw(ArgumentError("No brackets to be found"))
    end
    lo, hi = findbrackets()
    ρc = @time find_zero(f, (lo, hi), Bisection(), xatol=xatol)
    fc = @time find_nonergodicity_parameter(Cₖ, ρc+2xatol, D₀, k_array, order)

    randdiff = 0.0
    if rand() > 0.5
        randdiff += randn()*ρc/5 #far from critical point
    else
        randdiff += randn()*ρc/100 #close to critical point
    end
    ρ = ρc + randdiff #change to rand
    t_array, ϕ_out = solve_GMCT(Cₖ, k_array, ρ, D₀, order, N=2, tolerance=10^-12)


    open(@sprintf("Full Data\\fc(k)\\fc_rho_%.10f_rhoc_%.10f_order_%d.txt", ρ, ρc, order), "w") do io
        writedlm(io, fc)
    end

    open(@sprintf("Full Data\\F(k,t)\\Fkt_rho_%.10f_order_%d.txt", ρ, order), "w") do io
        writedlm(io, ϕ_out)
    end

    Sk = @. 1 + ρ*Cₖ / (1 - ρ*Cₖ)
    open(@sprintf("Full Data\\S(k)\\Sk_rho_%.10f.txt", ρ), "w") do io
        writedlm(io, Sk)
    end
    println("Distance from critical point ", randdiff, " at density $ρ")

    index = findmax(Sk)[2]

    Nt = 200
    log_time_sample_array = rand(Nt).*16 .- 6
    k_sample_array = rand(Nt)*(kmax-Δk) .+ Δk/2
    log_time_array = log10.(t_array)
    interpolation_spline = Spline2D(k_array, log_time_array, ϕ_out)
    ϕ_sample_array = interpolation_spline.(k_sample_array, log_time_sample_array)
    open(@sprintf("Sampled Data\\F(k,t)\\Fkt_rho_%.10f_order_%d.txt", ρ, order), "w") do io
        writedlm(io, [k_sample_array log_time_sample_array ϕ_sample_array])
    end


    p1 = plot(fc)
    p2 = plot(k_array, Sk)
    p3 = plot()
    plot!(p3, log10.(t_array), ϕ_out[index, :], ylims=(0,1), label="order = $order")
    display(p1)
    display(p2)
    display(p3)
end

# for order = randperm!(collect(1:5))
# main(order)
# end
η = 0.515
Nₖ = 64
kmax = 40
Δk = kmax/Nₖ
k_array = Δk*(collect(1:Nₖ) .- 0.5)
Cₖ = find_analytical_C_k(k_array, η)
D₀ = 1.0
xatol = 10^-4
ρ = η*6/π
order = 1
t_array, ϕ_out = solve_GMCT(Cₖ, k_array, ρ, D₀, order, N=2, tolerance=10^-12)
display(plot(log10.(t_array), ϕ_out[12, :]))
Sk = @. 1 + ρ*Cₖ / (1 - ρ*Cₖ)

open("Validation\\Hard Spheres\\Sk_HS_rho_$η.txt", "w") do io
    writedlm(io, Sk)
end

open("Validation\\Hard Spheres\\F_HS_rho_$(η)_k_$(k_array[12]).txt", "w") do io
    writedlm(io, (t_array, ϕ_out[12, :]))
end
# open("k_array.txt", "w") do io
#     writedlm(io, k_array)
# end

# open("logt_array.txt", "w") do io
#     writedlm(io, log10.(t_array))
# end