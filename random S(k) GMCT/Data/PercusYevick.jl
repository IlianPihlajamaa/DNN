using LoopVectorization, Plots

function find_analytical_C_r(r, η)
    """
        Finds the direct correlation function given by the 
        analytical percus yevick solution of the Ornstein Zernike 
        equation for hard spheres for a given volume fraction η on the coordinates r
        in units of one over the diameter of the particles
    """ 
    C = -(1 - η)^-4 * ((1 + 2η)^2 .- 6η * (1 + η/2)^2*r .+ 1/2 * η*(1 + 2η)^2 * r.^3)
    C[r.>1] .= 0
    return C
end

function find_analytical_C_k(k, η)
    """
        Finds the fourier transform of the direct correlation function given by the 
        analytical percus yevick solution of the Ornstein Zernike 
        equation for hard spheres for a given volume fraction η on the coordinates r
        in units of one over the diameter of the particles
    """ 
    A = -(1 - η)^-4 *(1 + 2η)^2
    B = (1 - η)^-4*  6η*(1 + η/2)^2
    D = -(1 - η)^-4 * 1/2 * η*(1 + 2η)^2
    Cₖ = @. 4π/k^6 * (24*D - 2*B * k^2 - (24*D - 2 * (B + 6*D) * k^2 + (A + B + D) * k^4) * cos(k) + k * (-24*D + (A + 2*B + 4*D) * k^2) * sin(k))
    return Cₖ
end


function find_analytical_S_k(k, η)
    """
        Finds the static structure factor given by the 
        analytical percus yevick solution of the Ornstein Zernike 
        equation for hard spheres for a given volume fraction η on the coordinates r
        in units of one over the diameter of the particles
    """ 
        Cₖ = find_analytical_C_k(k, η)
        ρ = 6/π * η
        Sₖ = @. 1 + ρ*Cₖ / (1 - ρ*Cₖ)
    return Sₖ
end

function find_analytical_S_k(ρ::Float64, Cₖ)
    """
        Finds the static structure factor given by the 
        analytical percus yevick solution of the Ornstein Zernike 
        equation for hard spheres for a given volume fraction η on the coordinates r
        in units of one over the diameter of the particles
    """ 
        Sₖ = @. 1 + ρ*Cₖ / (1 - ρ*Cₖ)
    return Sₖ
end

function plotCk(ρ)
    k = collect(LinRange(0, 60, 1000))
    η = π/6 * ρ
    Ck = find_analytical_C_k(k, η)
    p = plot(k, find_analytical_C_k(k, η))
    return p
end

function plotSk(ρ)
    k = collect(LinRange(0, 60, 1000))
    η = π/6 * ρ
    p = plot(k, find_analytical_S_k(k, η))
    return p
end

function find_random_C_k(k_array)
    ρ = 1.0

    a = rand(4) * -5                # between -3 and 0
    b = rand(4)*0.5 .+ 0.1          # between 0.1 and 0.6
    c = 1 ./ (rand(4)*20 .+ 0.5)    # between 0.05 and 2
    d = rand(4) * 2π                # between 0 and 2π

    Cₖ = zeros(length(k_array))

    for i = 1:4
        @. Cₖ += a[i] * exp(-b[i]*k_array) * cos(c[i]*k_array - d[i])
    end

    if maximum([abs(Cₖ[i+1]- Cₖ[i]) for i = 1:length(Cₖ)-1]) > 15.0 # differentiable
        Cₖ = find_random_C_k(k_array)
    end
    N_turning_points = 0
    for i = 1:length(Cₖ)-1
        if Cₖ[i+1]*Cₖ[i] < 0.0
            N_turning_points += 1
        end
    end
    if N_turning_points < 4 || N_turning_points > 15 # must oscillate but not too much
        Cₖ = find_random_C_k(k_array)
    end
    if minimum(find_analytical_S_k(ρ, Cₖ)) < 0.0 # Sk must be strictly positive
        Cₖ = find_random_C_k(k_array)
    end
    if maximum(find_analytical_S_k(ρ, Cₖ)) > 9 # peak of Sk must be less than 6
        Cₖ = find_random_C_k(k_array)
    end

    return Cₖ
end

# k_array = LinRange(0.2, 39.8, 640)
# ρ = 1.2
# p = plot()
# for i = 1:5
#     Cₖ = find_random_C_k(k_array, ρ)
#     Sₖ = find_analytical_S_k(ρ, Cₖ)
#     plot!(p, k_array, Sₖ)
# end
# p