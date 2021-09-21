using LoopVectorization, Plots, Tullio, BenchmarkTools, LaTeXStrings
include("PercusYevick.jl")
cd(@__DIR__)

plot_font = "Computer Modern"
default(fontfamily=plot_font,
        linewidth=2, framestyle=:box, label=nothing, grid=false)
scalefontsizes(1.5)

# vertices
@inline @fastmath function vertex(Cₖ, k_array, ip, iq, ik)
    Cq = Cₖ[iq]
    Cp = Cₖ[ip]
    p = k_array[ip]
    q = k_array[iq]
    k = k_array[ik] 
    k_dot_q = (k^2+q^2-p^2)/2
    V = k_dot_q*Cq/k + (k^2 - k_dot_q)*Cp/k
end

function find_kernel!(kernel, ϕ, V2, it,  D₀, ρ, ν)
    factor = -D₀ * ρ /(16 * π^3 * ν)
    Nk = length(ϕ[:, it])
    @tturbo for ik = 1:Nk
        kernelk = 0.0
         for iq = 1:Nk
            for ip = 1:Nk
                kernelk += ϕ[iq, it]*ϕ[ip, it]* V2[ip, iq, ik]
            end
        end
        kernel[ik, it] = kernelk
    end
    @tturbo for ik = 1:Nk
        kernel[ik, it] *= factor
    end
end

function time_integrate(f::Array, δt, it)
    result = 0.0
    for it2 = 1:it-1
        result += δt * (f[it2]+f[it2+1])/2
    end
    return result
end

function time_integrate(f::Float64, δt, it)
    return f*δt
end

function do_time_steps!(I_ϕ, I_Kernel, kernel, ϕ, Δt, N, Nk, δt, Ω², ν, V2, Sₖ, Cₖ, D₀, ρ, phi_plot, t_array)
    # println("Time = ", Δt)
    C1 = zeros(Nk)
    C2 = zeros(Nk)
    C3 = zeros(Nk)
    i2 = 2N
    max_iterations = 10000
    for i = 2N+1:4N
        error = 1
        ϕold = zeros(Nk)
        iterations = 0
        while error > 10^-6
            if iterations > max_iterations
                throw("Error: Iteration did not converge.")
            end
            iterations += 1
            @simd for ik = 1:Nk
                C1[ik] = 3/(2δt) - I_Kernel[ik, 1] + Ω²[ik]/ν
                C2[ik] = I_ϕ[ik, 1] - 1.0
                C3[ik] = 0
                C3[ik] += 2/δt*ϕ[ik, i-1] - ϕ[ik, i-2]/(2δt)
                C3[ik] += kernel[ik, i-i2]*ϕ[ik, i2] - kernel[ik, i-1]*I_ϕ[ik, 1] - ϕ[ik, i-1]*I_Kernel[ik, 1]
                for j=2:i2
                    C3[ik] -= (kernel[ik, i-j] - kernel[ik, i-j+1])*I_ϕ[ik, j]
                end
                for j =2:i-i2
                    C3[ik] -= (ϕ[ik, i-j] - ϕ[ik, i-j+1])*I_Kernel[ik, j]
                end
                ϕ[ik, i] = C2[ik]/C1[ik]*kernel[ik, i] + C3[ik]/C1[ik]
            end
            find_kernel!(kernel, ϕ, V2, i, D₀, ρ, ν)
            error = maximum(abs.(ϕ[:, i] - ϕold))
            ϕold .= ϕ[:, i]
        end
        for ik = 1:Nk
            I_ϕ[ik, i] = (ϕ[ik, i]+ϕ[ik, i-1])/2
            I_Kernel[ik, i] = (kernel[ik, i]+kernel[ik, i-1])/2
        end
        push!(phi_plot, ϕ[:, i])
        push!(t_array, i*δt)
    end
    return
end

function new_time_mapping(I_ϕ, I_Kernel, ϕ, kernel, N, Nk, Δt)
    I_ϕ_new = zeros(Nk, N*4)
    I_Kernel_new = zeros(Nk, N*4)
    Kernel_new = zeros(Nk, N*4)
    ϕ_new = zeros(Nk, N*4)
    @inbounds  for ik = 1:Nk
        @simd for j = 1:N
            I_ϕ_new[ik, j] = (I_ϕ[ik, 2j] + I_ϕ[ik, 2j - 1])/2
            I_Kernel_new[ik, j] = (I_Kernel[ik, 2j] + I_Kernel[ik, 2j - 1])/2
            ϕ_new[ik, j] = ϕ[ik, 2j]
            Kernel_new[ik, j] = kernel[ik, 2j]
        end
        @simd for j = N + 1:2*N
            I_ϕ_new[ik, j] = (I_ϕ[ik, 2j] + 4I_ϕ[ik, 2j - 1] + I_ϕ[ik, 2j-2])/6
            I_Kernel_new[ik, j] = (I_Kernel[ik, 2j] + 4I_Kernel[ik, 2j - 1] + I_Kernel[ik, 2j-2])/6
            ϕ_new[ik, j] = ϕ[ik, 2j]
            Kernel_new[ik, j] = kernel[ik, 2j]
        end
    end
    Δt *= 2
    δt = Δt/(4N)
    return I_ϕ_new, I_Kernel_new, Kernel_new, ϕ_new, Δt, δt
end

function find_intermediate_scattering_function(ρ, k_array, Cₖ)
    # parameters
    D₀ = 1.0
    ν = 1.0

    # spatial grid
    Nk = length(k_array)
    Δk = k_array[2] - k_array[1]
    k_weights = zeros(Nk)
    k_weights[1] = (k_array[2] - k_array[1])
    k_weights[Nk] = (Nk*Δk - (k_array[Nk]+k_array[Nk-1])/2)
    for ik = 2:Nk-1
        k_weights[ik] = (k_array[ik+1]/2 - k_array[ik-1]/2)
    end

    # structure
    Sₖ = find_analytical_S_k(k_array, ρ, Cₖ)

    V2 = zeros(Nk, Nk, Nk)
    @inbounds @simd for ik = 1:Nk
        k = k_array[ik]
        for iq = 1:Nk
            q = k_array[iq]
            for ip = abs(ik-iq)+1:min(ik+iq-1,Nk)
                p = k_array[ip]
                V2[ip, iq, ik] = vertex(Cₖ, k_array, ip, iq, ik)^2
                p_weight = k_weights[ip]
                q_weight = k_weights[ip]
                V2[ip, iq, ik] *=  q * p / k * q_weight*p_weight*2*π*Sₖ[iq]*Sₖ[ip]
            end
        end
    end

    N = 16
    Nt = 4N
    Δt = 10^-10
    δt = Δt/Nt
    t = 0
    t_array = Float64[]
    t_weights = Float64[]

    # initial conditions
    ϕ = zeros(Nk, Nt)
    ∂ₜϕ = zeros(Nk, Nt)
    ϕ₀ = ones(Nk)
    @tullio kernel₀[ik] :=  V2[ip, iq, ik]*ϕ₀[iq]*ϕ₀[ip] threads=false
    kernel₀[:] *= -D₀ * ρ /(16 * π^3 * ν)

    # preallocation & initialization
    kernel = zeros(Nk, Nt)
    time_integral = zeros(Nk)

    Ω² = D₀ * k_array.^2 ./ Sₖ

    @inbounds @simd  for ik = 1:Nk
        ∂ₜϕ[ik, 1] = (-D₀ * k_array[ik]^2*ϕ₀[ik]/(ν*Sₖ[ik]))
        ϕ[ik, 1] = ϕ₀[ik] + δt * ∂ₜϕ[ik, 1] 
    end

    find_kernel!(kernel, ϕ, V2, 1, D₀, ρ, ν)

    # a few euler steps
    for it = 1:4N-1
        t += Δt 
        for ik = 1:Nk
            time_integral[ik] =  time_integrate(kernel[ik, 1:it].*∂ₜϕ[ik, it:-1:1], δt, it)
        end
        @inbounds @simd for ik = 1:Nk
            ∂ₜϕ[ik, it]  = -Ω²[ik] * ϕ[ik, it]/ν - time_integral[ik]
            ϕ[ik, it+1] = ϕ[ik, it] + δt * ∂ₜϕ[ik, it]
        end
        find_kernel!(kernel, ϕ, V2, it+1, D₀, ρ, ν)
    end

    # construct integrals

    I_ϕ = zeros(Nk, Nt)
    I_Kernel = zeros(Nk, Nt)

     for it = 1:Nt
        if it == 1
            @inbounds @simd for ik = 1:Nk
                I_ϕ[ik, it] = (ϕ[ik, it] + 1.0)/2
                I_Kernel[ik, it] = (3kernel[ik, it] - kernel[ik, it+1])/2
            end
        else
            @inbounds @simd for ik = 1:Nk
                I_ϕ[ik, it] = (ϕ[ik, it] + ϕ[ik, it-1])/2
                I_Kernel[ik, it] = (kernel[ik, it] + kernel[ik, it-1])/2
            end
        end
    end

    phi_plot = Array{Array{Float64, 1},1}()
    t_array = Array{Float64, 1}()

    while Δt < 10^10
        I_ϕ, I_Kernel, kernel, ϕ, Δt, δt = new_time_mapping(I_ϕ, I_Kernel, ϕ, kernel, N, Nk, Δt)
        do_time_steps!(I_ϕ, I_Kernel, kernel, ϕ, Δt, N, Nk, δt, Ω², ν, V2, Sₖ, Cₖ, D₀, ρ, phi_plot, t_array)
    end

    ϕ = zeros(Nk, length(phi_plot))
    @inbounds  for ik = 1:Nk
        for it = 1:length(phi_plot)
            ϕ[ik, it] = phi_plot[it][ik]
        end
    end
    return t_array, ϕ
end

function find_random_C_k(k_array, ρ)
    a = rand(4) * -3                # between -3 and 0
    b = rand(4)*0.5 .+ 0.1          # between 0.1 and 0.6
    c = 1 ./ (rand(4)*20 .+ 0.1)    # between 0.05 and 1
    d = rand(4) * 2π                # between 0 and 2π

    Cₖ = zeros(length(k_array))
    for i = 1:4
        @. Cₖ += a[i] * exp(-b[i]*k_array) * cos(c[i]*k_array - d[i])
    end
    if maximum([abs(Cₖ[i+1]- Cₖ[i]) for i = 1:length(Cₖ)-1]) > 15.0
        Cₖ = find_random_C_k(k_array, ρ)
    end
    N_turning_points = 0
    for i = 1:length(Cₖ)-1
        if Cₖ[i+1]*Cₖ[i] < 0.0
            N_turning_points += 1
        end
    end
    if N_turning_points < 4 || N_turning_points > 15
        Cₖ = find_random_C_k(k_array, ρ)
    end
    if minimum(find_analytical_S_k(k_array, ρ, Cₖ)) < 0.0
        Cₖ = find_random_C_k(k_array, ρ)
    end
    # println("a = ", a)
    # println("b = ", b)
    # println("c = ", c)
    # println("d = ", d)
    return Cₖ
end

# p1 = plot()
# p = plot(xlabel=L"\log_{10}(t)", ylabel=L"F(k,t)/S(k)", ylims=(0,1), xlims=(-5,8))

# for sample = 1:6
#     ρ = 0.9
#     k_array = collect(LinRange(0.2, 39.8, 100))
#     # Cₖ = find_analytical_C_k(k_array, ρ)
#     Cₖ = find_random_C_k(k_array, ρ)
#     Sₖ = find_analytical_S_k(k_array, ρ, Cₖ)
#     plot!(p1, k_array, Sₖ, xlabel=L"k", ylabel=L"S(k)", label=nothing, framestyle = :box)

#     @time t_array, ϕ = find_intermediate_scattering_function(ρ, k_array, Cₖ)

#     for i in [16]
#         plot!(p, log10.(t_array), ϕ[i, :], label=nothing)#L"k=%$(round(k_array[i], digits=2))")
#     end
# end

scalefontsizes(1/1.5)
# plot!(p1, ylims=(0,3))
# display(p1)
# display(p)
# savefig(p1, "StructureFactorSample.png")
# savefig(p, "F(kt)Sample.png")

