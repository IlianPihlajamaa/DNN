function vertex(Cₖ, k_array, ip, iq, ik)
    Cq = Cₖ[iq]
    Cp = Cₖ[ip]
    p = k_array[ip]
    q = k_array[iq]
    k = k_array[ik] 
    k_dot_q = (k^2+q^2-p^2)/2
    V = k_dot_q*Cq/k + (k^2 - k_dot_q)*Cp/k
    return V
end


function find_vertex_functions!(cache_arrays, Cₖ, Sₖ, k_array, D₀, ρ)
    Δk = k_array[2] - k_array[1]
    Nk = length(k_array)

    @turbo for iq = 1:Nk
        for ip = 1:Nk
            p = k_array[ip]
            q = k_array[iq]
            Sp = Sₖ[ip]
            Sq = Sₖ[iq]
            cp = Cₖ[ip]
            cq = Cₖ[iq]         
            cache_arrays.V1[iq, ip] = p*q*Sp*Sq*(cp + cq)^2 / 4*D₀ * ρ /(8 * π^2)*Δk*Δk
            cache_arrays.V2[iq, ip] = p*q*Sp*Sq*(q^2-p^2)^2*(cq - cp)^2 / 4*D₀ * ρ /(8 * π^2)*Δk*Δk
            cache_arrays.V3[iq, ip] = p*q*Sp*Sq*(q^2-p^2)*(cq^2 - cp^2) / 2*D₀ * ρ /(8 * π^2)*Δk*Δk
        end
    end
end

function find_Ω²(k_array, Sₖ, D₀)
    return @. D₀ * k_array^2 / Sₖ 
end

function find_fₖ1!(fₖ1, kernel1, Ω²)
    fₖ1 .= kernel1 ./ (Ω² .+ kernel1)
end

function find_error(f1, f2)
    error = 0.0
    for i = 1:length(f1)
        @inbounds absdif = abs(f1[i] - f2[i])
        if absdif > error
            error = absdif
        end
    end
    return error
end
    
function find_non_ergodicity_parameter1(Cₖ, k_array, ρ, D₀; tolerance=10^-8, max_iterations=10^6)
    Nk = length(k_array)
    error = 1.0
    iteration = 0 
    Sₖ = @. 1 + ρ*Cₖ / (1 - ρ*Cₖ)
    Ω² = find_Ω²(k_array, Sₖ, D₀)
    fₖ1_old = ones(Nk)
    fₖ1 = zeros(Nk)
    kernel1 = zeros(Nk)
    cache_arrays = (
                    Σ1 = zeros(Nk),
                    Σ2 = zeros(Nk),
                    Σ3 = zeros(Nk),
                    V1 = zeros(Nk, Nk),
                    V2 = zeros(Nk, Nk),
                    V3 = zeros(Nk, Nk),
                    A1 = zeros(Nk, Nk),
                    A2 = zeros(Nk, Nk),
                    A3 = zeros(Nk, Nk),
                    q_integral = zeros(Nk)
    )

    find_vertex_functions!(cache_arrays, Cₖ, Sₖ, k_array, D₀, ρ)
    println("Finding non-ergodicity parameter at density ", ρ)
    while error > tolerance
        iteration += 1
        if iteration > max_iterations
            println("Iteration did not converge. The error is ", error)
            throw("Iteration aborted")
        end
        find_closure1!(kernel1, fₖ1_old, k_array, cache_arrays)
        find_fₖ1!(fₖ1, kernel1, Ω²)
        error = find_error(fₖ1, fₖ1_old)
        fₖ1_old .= fₖ1
    end
    println("Converged after ", iteration, " iterations.")
    return fₖ1
end

function find_fₖ2!(fₖ2, kernel2, Ω²)
    Nk = length(Ω²)
    ik = 1
    for ik2 = 1:Nk
        for ik1 = ik2:Nk
            fₖ2[ik] = kernel2[ik] / (Ω²[ik1] + Ω²[ik2] + kernel2[ik])
            ik += 1
        end
    end
end

function find_non_ergodicity_parameter2(Cₖ, k_array, ρ, D₀; tolerance=10^-8, max_iterations=10^6)
    Nk = length(k_array)
    error = 1.0
    iteration = 0 
    Sₖ = @. 1.0 + ρ*Cₖ / (1.0 - ρ*Cₖ)
    Ω² = find_Ω²(k_array, Sₖ, D₀)
    fₖ1_old = ones(Nk)
    fₖ1 = zeros(Nk)
    fₖ2 = zeros(Nk*(Nk + 1) ÷ 2)
    kernel1 = zeros(Nk)
    kernel2 = zeros(Nk*(Nk + 1) ÷ 2)
    cache_arrays = (
                    Σ1 = zeros(Nk),
                    Σ2 = zeros(Nk),
                    Σ3 = zeros(Nk),
                    V1 = zeros(Nk, Nk),
                    V2 = zeros(Nk, Nk),
                    V3 = zeros(Nk, Nk),
                    A1 = zeros(Nk, Nk),
                    A2 = zeros(Nk, Nk),
                    A3 = zeros(Nk, Nk),
                    q_integral = zeros(Nk),
                    linear_indices = find_linear_indices(Nk, 2)
    )
    find_vertex_functions!(cache_arrays, Cₖ, Sₖ, k_array, D₀, ρ)
    println("Finding non-ergodicity parameter at density ", ρ)
    while error > tolerance
        iteration += 1
        if iteration > max_iterations
            println("Iteration did not converge. The error is ", error)
            throw("Iteration aborted")
        end
        find_closure2_fⁿ!(kernel2, fₖ1_old, k_array, Ω², cache_arrays)
        find_fₖ2!(fₖ2, kernel2, Ω²)
        find_kernel1!(kernel1, fₖ2, k_array, cache_arrays)
        find_fₖ1!(fₖ1, kernel1, Ω²)
        error = find_error(fₖ1, fₖ1_old)
        fₖ1_old .= fₖ1
    end
    println("Converged after ", iteration, " iterations.")
    return fₖ1
end

function find_fₖ3!(fₖ3, kernel3, Ω²)
    Nk = length(Ω²)
    ik = 1
    @inbounds for ik3 = 1:Nk
        Ω²3 = Ω²[ik3]
        for ik2 = ik3:Nk
            Ω²2 = Ω²[ik2]
            @simd for ik1 = ik2:Nk
                Ω²1 = Ω²[ik1]
                k = kernel3[ik]
                fₖ3[ik] = k / (Ω²1 + Ω²2 + Ω²3 + k)
                ik += 1
            end
        end
    end
end

function find_non_ergodicity_parameter3(Cₖ, k_array, ρ, D₀; tolerance=10^-8, max_iterations=10^6)
    Nk = length(k_array)
    error = 1.0
    iteration = 0 
    Sₖ = @. 1.0 + ρ*Cₖ / (1.0 - ρ*Cₖ)
    Ω² = find_Ω²(k_array, Sₖ, D₀)
    fₖ1_old = ones(Nk)
    fₖ1 = zeros(Nk)
    fₖ2 = zeros(Nk*(Nk + 1) ÷ 2)
    fₖ3 = zeros(Nk*(Nk + 1)*(Nk + 2) ÷ 6)
    kernel1 = zeros(Nk)
    kernel2 = zeros(Nk*(Nk + 1) ÷ 2)
    kernel3 = zeros(Nk*(Nk + 1)*(Nk + 2) ÷ 6)
    q_integral = zeros(Nk)
    cache_arrays = (
        Σ1 = zeros(Nk),
        Σ2 = zeros(Nk),
        Σ3 = zeros(Nk),
        V1 = zeros(Nk, Nk),
        V2 = zeros(Nk, Nk),
        V3 = zeros(Nk, Nk),
        A1 = zeros(Nk, Nk),
        A2 = zeros(Nk, Nk),
        A3 = zeros(Nk, Nk),
        q_integral = zeros(Nk),
        kernel2_temp = zeros(Nk, Nk),
        linear_indices = find_linear_indices(Nk, 3)
    )
    cache_arrays_threaded = (
        Σ1 = zeros(Nk, nthreads()),
        Σ2 = zeros(Nk, nthreads()),
        Σ3 = zeros(Nk, nthreads()),
        V1 = zeros(Nk, Nk),
        V2 = zeros(Nk, Nk),
        V3 = zeros(Nk, Nk),
        A1 = zeros(Nk, Nk, nthreads()),
        A2 = zeros(Nk, Nk, nthreads()),
        A3 = zeros(Nk, Nk, nthreads()),
        q_integral = zeros(Nk, nthreads()),
        kernel2_temp =  zeros(Nk, Nk),
        linear_indices = find_linear_indices(Nk, 3)
    )
    find_vertex_functions!(cache_arrays, Cₖ, Sₖ, k_array, D₀, ρ)
    find_vertex_functions!(cache_arrays_threaded, Cₖ, Sₖ, k_array, D₀, ρ)
    println("Finding non-ergodicity parameter at density ", ρ)

    while error > tolerance 
        iteration += 1
        if iteration > max_iterations
            println("Iteration did not converge. The error is ", error)
            throw("Iteration aborted")
        end
        find_closure3_fⁿ!(kernel3, cache_arrays, fₖ1_old, Ω², k_array)
        find_fₖ3!(fₖ3, kernel3, Ω²)
        find_kernel2!(kernel2, cache_arrays_threaded, fₖ3, Ω², k_array)
        find_fₖ2!(fₖ2, kernel2, Ω²)
        find_kernel1!(kernel1, fₖ2, k_array, cache_arrays)
        find_fₖ1!(fₖ1, kernel1, Ω²)
        error = find_error(fₖ1, fₖ1_old)
        fₖ1_old .= fₖ1
    end
    println("Converged after ", iteration, " iterations.")
    return fₖ1
end

function find_fₖ4!(fₖ4, kernel4, Ω²)
    Nk = length(Ω²)
    ik = 1
    @inbounds for ik4 = 1:Nk
        Ω²4 = Ω²[ik4]
        for ik3 = ik4:Nk
            Ω²3 = Ω²[ik3]
            for ik2 = ik3:Nk
                Ω²2 = Ω²[ik2]
                @simd for ik1 = ik2:Nk
                    Ω²1 = Ω²[ik1]
                    k = kernel4[ik]
                    fₖ4[ik] = k / (Ω²1 + Ω²2 + Ω²3 + Ω²4 + k)
                    ik += 1
                end
            end
        end
    end
end

function find_non_ergodicity_parameter4(Cₖ, k_array, ρ, D₀; tolerance=10^-8, max_iterations=10^6)
    Nk = length(k_array)
    error = 1.0
    iteration = 0 
    Sₖ = @. 1.0 + ρ*Cₖ / (1.0 - ρ*Cₖ)
    Ω² = find_Ω²(k_array, Sₖ, D₀)
    fₖ1_old = ones(Nk)
    fₖ1 = zeros(Nk)
    Nkk = Nk*(Nk + 1)÷2
    Nkkk = Nk*(Nk + 1)*(Nk + 2) ÷ 6
    Nkkkk = Nk*(Nk + 1)*(Nk + 2)*(Nk + 3) ÷ 24
    fₖ1 = zeros(Nk)
    fₖ2 = zeros(Nkk)
    fₖ3 = zeros(Nkkk)
    fₖ4 = zeros(Nkkkk)
    kernel1 = zeros(Nk)
    kernel2 = zeros(Nkk)
    kernel3 = zeros(Nkkk)
    kernel4 = zeros(Nkkkk)
    q_integral = zeros(Nk)
    cache_arrays = (
        Σ1 = zeros(Nk),
        Σ2 = zeros(Nk),
        Σ3 = zeros(Nk),
        V1 = zeros(Nk, Nk),
        V2 = zeros(Nk, Nk),
        V3 = zeros(Nk, Nk),
        A1 = zeros(Nk, Nk),
        A2 = zeros(Nk, Nk),
        A3 = zeros(Nk, Nk),
        q_integral = zeros(Nk),
        kernel2_temp = zeros(Nk, Nk),
        linear_indices = find_linear_indices(Nk, 4)
    )
    cache_arrays_threaded = (
        Σ1 = zeros(Nk, nthreads()),
        Σ2 = zeros(Nk, nthreads()),
        Σ3 = zeros(Nk, nthreads()),
        V1 = zeros(Nk, Nk),
        V2 = zeros(Nk, Nk),
        V3 = zeros(Nk, Nk),
        A1 = zeros(Nk, Nk, nthreads()),
        A2 = zeros(Nk, Nk, nthreads()),
        A3 = zeros(Nk, Nk, nthreads()),
        q_integral = zeros(Nk, nthreads()),
        kernel2_temp =  zeros(Nk, Nk),
        kernel3_temp = zeros(Nk, Nkk),
        linear_indices = find_linear_indices(Nk, 4)
    )
    find_vertex_functions!(cache_arrays, Cₖ, Sₖ, k_array, D₀, ρ)
    find_vertex_functions!(cache_arrays_threaded, Cₖ, Sₖ, k_array, D₀, ρ)
    println("Finding non-ergodicity parameter at density ", ρ)
    @time while error > tolerance
        iteration += 1
        if iteration > max_iterations
            println("Iteration did not converge. The error is ", error)
            throw("Iteration aborted")
        end
        find_closure4_fⁿ!(kernel4, cache_arrays, fₖ1_old, Ω², k_array)
        find_fₖ4!(fₖ4, kernel4, Ω²)
        find_kernel3!(kernel3, cache_arrays_threaded, fₖ4, Ω², k_array)
        find_fₖ3!(fₖ3, kernel3, Ω²)
        find_kernel2!(kernel2, cache_arrays_threaded, fₖ3, Ω², k_array)
        find_fₖ2!(fₖ2, kernel2, Ω²)
        find_kernel1!(kernel1, fₖ2, k_array, cache_arrays)
        find_fₖ1!(fₖ1, kernel1, Ω²)
        error = find_error(fₖ1, fₖ1_old)
        fₖ1_old .= fₖ1
    end
    println("Converged after ", iteration, " iterations.")
    return fₖ1
end

function find_fₖ5!(fₖ5, kernel5, Ω²)
    Nk = length(Ω²)
    ik = 1
    @inbounds for ik5 = 1:Nk
        Ω²5 = Ω²[ik5]
        for ik4 = ik5:Nk
            Ω²4 = Ω²[ik4]
            for ik3 = ik4:Nk
                Ω²3 = Ω²[ik3]
                for ik2 = ik3:Nk
                    Ω²2 = Ω²[ik2]
                    @simd for ik1 = ik2:Nk
                        Ω²1 = Ω²[ik1]
                        k = kernel5[ik]
                        fₖ5[ik] = k / (Ω²1 + Ω²2 + Ω²3 + Ω²4 + Ω²5 + k)
                        ik += 1
                    end
                end
            end
        end
    end
end

function find_non_ergodicity_parameter5(Cₖ, k_array, ρ, D₀; tolerance=10^-8, max_iterations=10^6)
    Nk = length(k_array)
    error = 1.0
    iteration = 0 
    Sₖ = @. 1.0 + ρ*Cₖ / (1.0 - ρ*Cₖ)
    Ω² = find_Ω²(k_array, Sₖ, D₀)
    fₖ1_old = ones(Nk)
    Nkk = Nk*(Nk + 1)÷2
    Nkkk = Nk*(Nk + 1)*(Nk + 2) ÷ 6
    Nkkkk = Nk*(Nk + 1)*(Nk + 2)*(Nk + 3) ÷ 24
    Nkkkkk = Nk*(Nk + 1)*(Nk + 2)*(Nk + 3)*(Nk + 4) ÷ 120
    fₖ1 = zeros(Nk)
    fₖ2 = zeros(Nkk)
    fₖ3 = zeros(Nkkk)
    fₖ4 = zeros(Nkkkk)
    fₖ5 = zeros(Nkkkkk)
    kernel1 = zeros(Nk)
    kernel2 = zeros(Nkk)
    kernel3 = zeros(Nkkk)
    kernel4 = zeros(Nkkkk)
    kernel5 = zeros(Nkkkkk)
    q_integral = zeros(Nk)
    cache_arrays = (
        Σ1 = zeros(Nk),
        Σ2 = zeros(Nk),
        Σ3 = zeros(Nk),
        V1 = zeros(Nk, Nk),
        V2 = zeros(Nk, Nk),
        V3 = zeros(Nk, Nk),
        A1 = zeros(Nk, Nk),
        A2 = zeros(Nk, Nk),
        A3 = zeros(Nk, Nk),
        q_integral = zeros(Nk),
        kernel2_temp = zeros(Nk, Nk),
        linear_indices = find_linear_indices(Nk, 5)
    )
    cache_arrays_threaded = (
        Σ1 = zeros(Nk, nthreads()),
        Σ2 = zeros(Nk, nthreads()),
        Σ3 = zeros(Nk, nthreads()),
        V1 = zeros(Nk, Nk),
        V2 = zeros(Nk, Nk),
        V3 = zeros(Nk, Nk),
        A1 = zeros(Nk, Nk, nthreads()),
        A2 = zeros(Nk, Nk, nthreads()),
        A3 = zeros(Nk, Nk, nthreads()),
        q_integral = zeros(Nk, nthreads()),
        kernel2_temp =  zeros(Nk, Nk),
        kernel3_temp = zeros(Nk, Nkk),
        kernel4_temp = zeros(Nk, Nkkk),
        linear_indices = find_linear_indices(Nk, 5)
    )
    find_vertex_functions!(cache_arrays, Cₖ, Sₖ, k_array, D₀, ρ)
    find_vertex_functions!(cache_arrays_threaded, Cₖ, Sₖ, k_array, D₀, ρ)
    println("Finding non-ergodicity parameter at density ", ρ)
    @time while error > tolerance
        iteration += 1
        if iteration > max_iterations
            println("Iteration did not converge. The error is ", error)
            throw("Iteration aborted")
        end
        find_closure5_fⁿ!(kernel5, cache_arrays, fₖ1_old, Ω², k_array)
        find_fₖ5!(fₖ5, kernel5, Ω²)
        find_kernel4!(kernel4, cache_arrays_threaded, fₖ5, Ω², k_array)
        find_fₖ4!(fₖ4, kernel4, Ω²)
        find_kernel3!(kernel3, cache_arrays_threaded, fₖ4, Ω², k_array)
        find_fₖ3!(fₖ3, kernel3, Ω²)
        find_kernel2!(kernel2, cache_arrays_threaded, fₖ3, Ω², k_array)
        find_fₖ2!(fₖ2, kernel2, Ω²)
        find_kernel1!(kernel1, fₖ2, k_array, cache_arrays)
        find_fₖ1!(fₖ1, kernel1, Ω²)
        error = find_error(fₖ1, fₖ1_old)
        fₖ1_old .= fₖ1
    end
    println("Converged after ", iteration, " iterations.")
    return fₖ1
end



function find_non_ergodicity_parameter(Cₖ, k_array, ρ, D₀, order; tolerance=10^-8, max_iterations=10^6)
    if order == 1 #MCT
        return find_non_ergodicity_parameter1(Cₖ, k_array, ρ, D₀; tolerance=tolerance, max_iterations=max_iterations)
    elseif order == 2
        return find_non_ergodicity_parameter2(Cₖ, k_array, ρ, D₀; tolerance=tolerance, max_iterations=max_iterations)
    elseif order == 3
        return find_non_ergodicity_parameter3(Cₖ, k_array, ρ, D₀; tolerance=tolerance, max_iterations=max_iterations)
    elseif order == 4
        return find_non_ergodicity_parameter4(Cₖ, k_array, ρ, D₀; tolerance=tolerance, max_iterations=max_iterations)
    elseif order == 5
        return find_non_ergodicity_parameter5(Cₖ, k_array, ρ, D₀; tolerance=tolerance, max_iterations=max_iterations)
    end
end

