
using LoopVectorization, Plots, Tullio, BenchmarkTools
include("PercusYevick.jl")

# vertices
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

function find_vertex_functions!(params)
    k_array = params.k_array
    Δk = k_array[2] - k_array[1]
    Nk = params.Nk
    D₀ = params.D₀
    ρ = params.ρ
    @turbo for iq = 1:Nk
        for ip = 1:Nk
            p = k_array[ip]
            q = k_array[iq]
            Sp = params.Sₖ[ip]
            Sq = params.Sₖ[iq]
            cp = params.Cₖ[ip]
            cq = params.Cₖ[iq]         
            params.V1[iq, ip] = p*q*Sp*Sq*(cp + cq)^2 / 4*D₀ * ρ /(8 * π^2)*Δk*Δk
            params.V2[iq, ip] = p*q*Sp*Sq*(q^2-p^2)^2*(cq - cp)^2 / 4*D₀ * ρ /(8 * π^2)*Δk*Δk
            params.V3[iq, ip] = p*q*Sp*Sq*(q^2-p^2)*(cq^2 - cp^2) / 2*D₀ * ρ /(8 * π^2)*Δk*Δk
        end
    end
end

function initialize_ϕ1!(ϕ, parameters)
    for it = 1:parameters.Nt
        for ik = 1:parameters.Nk 
            ϕ[1][ik, it] = exp(-parameters.Ω²[ik] * parameters.δt * parameters.Nt)
        end
    end
end

function initialize_ϕ2!(ϕ, parameters)
    for it = 1:parameters.Nt
        ik = 0
        for ik2 = 1:parameters.Nk 
            for ik1 = ik2:parameters.Nk
                ik += 1
                ϕ[2][ik, it] = exp(-(parameters.Ω²[ik1] + parameters.Ω²[ik2]) * parameters.δt * parameters.Nt)
            end
        end
    end
end

function initialize_ϕ3!(ϕ, parameters)
    for it = 1:parameters.Nt
        ik = 0
        for ik3 = 1:parameters.Nk 
            for ik2 = ik3:parameters.Nk 
                for ik1 = ik2:parameters.Nk
                    ik += 1
                    ϕ[3][ik, it] = exp(-(parameters.Ω²[ik1] + parameters.Ω²[ik2] + parameters.Ω²[ik3]) * parameters.δt * parameters.Nt)
                end
            end
        end
    end
end

function initialize_ϕ4!(ϕ, parameters)
    for it = 1:parameters.Nt
        ik = 0
        for ik4 = 1:parameters.Nk
            for ik3 = ik4:parameters.Nk 
                for ik2 = ik3:parameters.Nk 
                    for ik1 = ik2:parameters.Nk
                        ik += 1
                        ϕ[4][ik, it] = exp(-(parameters.Ω²[ik1] + parameters.Ω²[ik2] + parameters.Ω²[ik3] + parameters.Ω²[ik4]) * parameters.δt * parameters.Nt)
                    end
                end
            end
        end
    end
end

function initialize_ϕ5!(ϕ, parameters)
    for it = 1:parameters.Nt
        ik = 0
        for ik5 = 1:parameters.Nk
            for ik4 = ik5:parameters.Nk
                for ik3 = ik4:parameters.Nk 
                    for ik2 = ik3:parameters.Nk 
                        for ik1 = ik2:parameters.Nk
                            ik += 1
                            ϕ[4][ik, it] = exp(-(parameters.Ω²[ik1] + parameters.Ω²[ik2] + parameters.Ω²[ik3] + parameters.Ω²[ik4] + parameters.Ω²[ik5]) * parameters.δt * parameters.Nt)
                        end
                    end
                end
            end
        end
    end
end


function new_time_mapping!(parameters, temp_arrays, results)
    N = parameters.N
    δt = parameters.δt
    for i = 2N+1:4N
        push!(results.ϕ_out, results.ϕ[1][:, i])
        push!(results.t_array, i*δt)
    end
    for order = 1:parameters.order
        kernel = results.kernel[order]
        ϕ = results.ϕ[order]
        I_ϕ = temp_arrays.I_ϕ[order]
        I_Kernel = temp_arrays.I_kernel[order]
        Nk, Nt = size(kernel)
        N = Nt ÷ 4
        @inbounds for ik = 1:Nk
            for j = 1:N
                I_ϕ[ik, j] = (I_ϕ[ik, 2j] + I_ϕ[ik, 2j - 1])/2
                I_Kernel[ik, j] = (I_Kernel[ik, 2j] + I_Kernel[ik, 2j - 1])/2
                ϕ[ik, j] = ϕ[ik, 2j]
                kernel[ik, j] = kernel[ik, 2j]
            end
            for j = (N + 1):2*N
                I_ϕ[ik, j] = (I_ϕ[ik, 2j] + 4I_ϕ[ik, 2j - 1] + I_ϕ[ik, 2j-2])/6
                I_Kernel[ik, j] = (I_Kernel[ik, 2j] + 4I_Kernel[ik, 2j - 1] + I_Kernel[ik, 2j-2])/6
                ϕ[ik, j] = ϕ[ik, 2j]
                kernel[ik, j] = kernel[ik, 2j]
            end
            for j = 2N+1:4N
                I_ϕ[ik, j] = 0.0
                I_Kernel[ik, j] = 0.0
                ϕ[ik, j] = 0.0
                kernel[ik, j] = 0.0
            end
        end
    end
    parameters.Δt *= 2
    parameters.δt = parameters.Δt/(parameters.Nt)
end


function initialize_integrals!(parameters, temp_arrays, results)
    for order = 1:parameters.order
        I_ϕ = temp_arrays.I_ϕ[order]
        I_Kernel = temp_arrays.I_kernel[order]
        ϕ = results.ϕ[order]
        kernel = results.kernel[order]
        Nk, Nt = size(ϕ)
        for it = 1:Nt
            if it == 1
                for ik = 1:Nk
                    I_ϕ[ik, it] = (ϕ[ik, it] + 1.0)/2
                    I_Kernel[ik, it] = (3kernel[ik, it] - kernel[ik, it+1])/2
                end
            else
                for ik = 1:Nk
                    I_ϕ[ik, it] = (ϕ[ik, it] + ϕ[ik, it-1])/2
                    I_Kernel[ik, it] = (kernel[ik, it] + kernel[ik, it-1])/2
                end
            end
        end
    end
end

function find_Ωik(parameters, ik, current_order)
    idx = parameters.full_indices[ik]
    Ω² = parameters.Ω²
    result = 0.0
    for i = 1:current_order
        result += Ω²[idx[i]]
    end
    return result
end

function fuchs!(parameters, temp_arrays, results, i)
    N = parameters.N
    i2 = 2N
    δt = parameters.δt
    for order = 1:parameters.order
        C1 = temp_arrays.C1[order]
        C2 = temp_arrays.C2[order]
        C3 = temp_arrays.C3[order]
        I_Kernel = temp_arrays.I_kernel[order]
        I_ϕ = temp_arrays.I_ϕ[order]
        ϕ = results.ϕ[order]
        kernel = results.kernel[order]
        Nk = size(ϕ)[1]
        @inbounds for ik = 1:Nk
            Ω²ik = find_Ωik(parameters, ik, order)
            c1ik = 3/(2δt) + I_Kernel[ik, 1] + Ω²ik
            c2ik = I_ϕ[ik, 1] - 1.0
            c3ik = 0.0
            c3ik = 0
            c3ik += 2/δt*ϕ[ik, i-1] - ϕ[ik, i-2]/(2δt)
            c3ik -= kernel[ik, i-i2]*ϕ[ik, i2] - kernel[ik, i-1]*I_ϕ[ik, 1] - ϕ[ik, i-1]*I_Kernel[ik, 1]
            for j = 2:i2
                c3ik += (kernel[ik, i-j] - kernel[ik, i-j+1])*I_ϕ[ik, j]
            end
            for j = 2:i-i2
                c3ik += (ϕ[ik, i-j] - ϕ[ik, i-j+1])*I_Kernel[ik, j]
            end
            C1[ik] = c1ik
            C2[ik] = c2ik
            C3[ik] = c3ik
            ϕ[ik, i] = -c2ik/c1ik*kernel[ik, i] + c3ik/c1ik
        end
    end
end




function update_integrals(parameters, temp_arrays, results, i)
    for order = 1:parameters.order
        I_Kernel = temp_arrays.I_kernel[order]
        I_ϕ = temp_arrays.I_ϕ[order]
        ϕ = results.ϕ[order]
        kernel = results.kernel[order]
        Nk = size(kernel)[1]
        @inbounds for ik = 1:Nk
            I_ϕ[ik, i] = (ϕ[ik, i]+ϕ[ik, i-1])/2
            I_Kernel[ik, i] = (kernel[ik, i]+kernel[ik, i-1])/2
        end
    end
end

function find_error(ϕ, ϕ_old, i)
    error = 0.0
    for ik = 1:size(ϕ)[1]
        newerror = abs(ϕ[ik, i] - ϕ_old[ik])
        if newerror > error
            error = newerror
        end
    end
    return error
end 

function update_ϕ!(parameters, temp_arrays, results, i, order)
    C1 = temp_arrays.C1[order]
    C2 = temp_arrays.C2[order]
    C3 = temp_arrays.C3[order]
    ϕ = results.ϕ[order]
    kernel = results.kernel[order]
    Nk = size(ϕ)[1]
    @inbounds for ik = 1:Nk
        ϕ[ik, i] = -C2[ik]/C1[ik]*kernel[ik, i] + C3[ik]/C1[ik]
    end
end


function update_kernels!(parameters, temp_arrays, results, it)
    order = parameters.order
    results.kernel_evals += 1
    if order == 1
        find_closure1!(parameters, temp_arrays, results, it)
    elseif order == 2
        find_closure2_fⁿ!(parameters, temp_arrays, results, it)
        find_kernel1!(parameters, temp_arrays, results, it)
    elseif order == 3
        find_closure3_fⁿ!(parameters, temp_arrays, results, it)
        find_kernel2!(parameters, temp_arrays, results, it)
        find_kernel1!(parameters, temp_arrays, results, it)
    elseif order == 4
        find_closure4_fⁿ!(parameters, temp_arrays, results, it)
        find_kernel3!(parameters, temp_arrays, results, it)
        find_kernel2!(parameters, temp_arrays, results, it)
        find_kernel1!(parameters, temp_arrays, results, it)
    elseif order == 5
        find_closure5_fⁿ!(parameters, temp_arrays, results, it)
        find_kernel4!(parameters, temp_arrays, results, it)
        find_kernel3!(parameters, temp_arrays, results, it)
        find_kernel2!(parameters, temp_arrays, results, it)
        find_kernel1!(parameters, temp_arrays, results, it)
    end
end

function update_kernels_and_ϕ!(parameters, temp_arrays, results, it)
    order = parameters.order
    results.kernel_evals += 1
    if order == 1
        find_closure1!(parameters, temp_arrays, results, it)
        update_ϕ!(parameters, temp_arrays, results, it, 1)
    elseif order == 2
        find_closure2_fⁿ!(parameters, temp_arrays, results, it)
        update_ϕ!(parameters, temp_arrays, results, it, 2)
        find_kernel1!(parameters, temp_arrays, results, it)
        update_ϕ!(parameters, temp_arrays, results, it, 1)
    elseif order == 3
        find_closure3_fⁿ!(parameters, temp_arrays, results, it)
        update_ϕ!(parameters, temp_arrays, results, it, 3)
        find_kernel2!(parameters, temp_arrays, results, it)
        update_ϕ!(parameters, temp_arrays, results, it, 2)
        find_kernel1!(parameters, temp_arrays, results, it)
        update_ϕ!(parameters, temp_arrays, results, it, 1)
    elseif order == 4
        find_closure4_fⁿ!(parameters, temp_arrays, results, it)
        update_ϕ!(parameters, temp_arrays, results, it, 4)
        find_kernel3!(parameters, temp_arrays, results, it)
        update_ϕ!(parameters, temp_arrays, results, it, 3)
        find_kernel2!(parameters, temp_arrays, results, it)
        update_ϕ!(parameters, temp_arrays, results, it, 2)
        find_kernel1!(parameters, temp_arrays, results, it)
        update_ϕ!(parameters, temp_arrays, results, it, 1)
    elseif order == 5
        find_closure5_fⁿ!(parameters, temp_arrays, results, it)
        update_ϕ!(parameters, temp_arrays, results, it, 5)
        find_kernel4!(parameters, temp_arrays, results, it)
        update_ϕ!(parameters, temp_arrays, results, it, 4)
        find_kernel3!(parameters, temp_arrays, results, it)
        update_ϕ!(parameters, temp_arrays, results, it, 3)
        find_kernel2!(parameters, temp_arrays, results, it)
        update_ϕ!(parameters, temp_arrays, results, it, 2)
        find_kernel1!(parameters, temp_arrays, results, it)
        update_ϕ!(parameters, temp_arrays, results, it, 1)
    end
end


function update_ϕ!(parameters, temp_arrays, results, i)
    for order = 1:parameters.order
        C1 = temp_arrays.C1[order]
        C2 = temp_arrays.C2[order]
        C3 = temp_arrays.C3[order]
        ϕ = results.ϕ[order]
        kernel = results.kernel[order]
        Nk = size(ϕ)[1]
        @inbounds for ik = 1:Nk
            ϕ[ik, i] = -C2[ik]/C1[ik]*kernel[ik, i] + C3[ik]/C1[ik]
        end
    end
end

function update_kernels!(parameters, temp_arrays, results)
    for i = 1:parameters.Nt
        update_kernels!(parameters, temp_arrays, results, i)
    end
end



function do_time_steps!(parameters, temp_arrays, results)
    N = parameters.N
    ϕ_old = temp_arrays.ϕ_old
    ϕ = results.ϕ[1]
    tolerance = parameters.tolerance
    for i = 2N+1:4N
        error = 1
        iterations = 0
        ϕ_old .= 0
        fuchs!(parameters, temp_arrays, results, i)
        while error > tolerance
            iterations += 1
            if iterations > parameters.max_iterations
                throw(DomainError("Iteration did not converge"))
            end
            update_kernels_and_ϕ!(parameters, temp_arrays, results, i)
            # update_ϕ!(parameters, temp_arrays, results, i)
            error = find_error(ϕ, ϕ_old, i)
            @views ϕ_old .= ϕ[:, i]
            update_integrals(parameters, temp_arrays, results, i)
            # println(error)
        end
    end
    return
end


function create_parameter_set(order, D₀, N, Nk, ρ, tolerance, max_iterations, tmax, Δt, Cₖ, k_array)
    Nt = 4N
    Sₖ = @. 1.0 + ρ*Cₖ / (1.0 - ρ*Cₖ)
    δt = Δt/Nt
    η = π*ρ / 6
    Ω² = D₀ * k_array.^2 ./ Sₖ
    linear_indices = find_linear_indices(Nk, order)
    full_indices = find_full_indices(Nk, order)
    params = Parameters(order, D₀, N, Nt, Nk, ρ, η, δt, Δt, tmax, tolerance, max_iterations,
             Cₖ, Sₖ, zeros(Nk, Nk), zeros(Nk, Nk), zeros(Nk, Nk), k_array, Ω², linear_indices, full_indices)
    find_vertex_functions!(params)
    return params
end

function create_temp_arr_set(parameters)
    Nt = parameters.Nt
    Nk = parameters.Nk
    I_ϕ = Array{Array{Float64, 2}, 1}()
    I_kernel = Array{Array{Float64, 2}, 1}()
    C1 = Array{Array{Float64, 1}, 1}()
    C2 = Array{Array{Float64, 1}, 1}()
    C3 = Array{Array{Float64, 1}, 1}()
    kernel_temp = Array{Array{Float64, 2}, 1}()
    for i = 1:parameters.order
        push!(I_ϕ, zeros(symmetric_tensor_size(Nk, i), Nt))
        push!(I_kernel, zeros(symmetric_tensor_size(Nk, i), Nt))
        push!(C1, zeros(symmetric_tensor_size(Nk, i)))
        push!(C2, zeros(symmetric_tensor_size(Nk, i)))
        push!(C3, zeros(symmetric_tensor_size(Nk, i)))
        if 1 <= i < parameters.order
            push!(kernel_temp, zeros(Nk, symmetric_tensor_size(Nk, i-1)))
        end
    end
    temp_arrays = TemporaryArrays(zeros(Nk), zeros(Nk), zeros(Nk), zeros(Nk, Nk), 
                   zeros(Nk, Nk), zeros(Nk, Nk), zeros(Nk), I_ϕ, I_kernel, C1, C2, C3, zeros(Nk), kernel_temp)
    return temp_arrays
end

function allocate_results(parameters)
    ϕ = Array{Float64, 2}[]
    order = parameters.order
    kernel = Array{Float64, 2}[]
    for order in 1:parameters.order
        Nki = symmetric_tensor_size(parameters.Nk, order)
        push!(ϕ, zeros(Nki, parameters.Nt))
        push!(kernel, zeros(Nki, parameters.Nt))
    end
    if order >= 1
        initialize_ϕ1!(ϕ, parameters)
    end
    if order >= 2 
        initialize_ϕ2!(ϕ, parameters)
    end
    if order >= 3 
        initialize_ϕ3!(ϕ, parameters)
    end
    if order >= 4 
        initialize_ϕ3!(ϕ, parameters)
    end
    if order >= 5 
        initialize_ϕ3!(ϕ, parameters)
    end
    if order >= 6 
        initialize_ϕ3!(ϕ, parameters)
    end
    #extend later

    ϕ_out = Array{Float64, 1}[]
    t_array = Float64[]

    results = Results(ϕ, kernel, t_array, ϕ_out, 0)
    return results
end



function solve_GMCT(Cₖ, k_array, ρ, D₀, order; tolerance=10^-8, max_iterations=10^6, N=32, tmax=10^10, Δt=10^-10)
    begintime = time()
    Nk = length(k_array)
    parameters = create_parameter_set(order, D₀, N, Nk, ρ, tolerance, max_iterations, tmax, Δt, Cₖ, k_array)
    temp_arrays = create_temp_arr_set(parameters)
    results = allocate_results(parameters)
    update_kernels!(parameters, temp_arrays, results)
    initialize_integrals!(parameters, temp_arrays, results)
    while parameters.Δt < parameters.tmax
        println("Time = ", parameters.Δt)
        if sum(results.ϕ[1]) > 10^-9 # if fully relaxed don't calculate memory kernels
            do_time_steps!(parameters, temp_arrays, results)
        end
        new_time_mapping!(parameters, temp_arrays, results)
    end
    ϕ_out = results.ϕ_out
    @tullio ϕ_out2[j, i] := ϕ_out[i][j]
    endtime = time()
    println("Full solution found after $(round(endtime-begintime, digits=2)) seconds with $(results.kernel_evals) kernel evaluations.")
    return results.t_array, ϕ_out2
end