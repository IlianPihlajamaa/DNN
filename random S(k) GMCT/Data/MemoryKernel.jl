mutable struct Parameters{T}
    order::Int64
    D₀::Float64
    N::Int64
    Nt::Int64
    Nk::Int64
    ρ::Float64
    η::Float64
    δt::Float64
    Δt::Float64
    tmax::Int64
    tolerance::Float64
    max_iterations::Int64
    Cₖ::Array{Float64, 1}
    Sₖ::Array{Float64, 1}
    V1::Array{Float64, 2}
    V2::Array{Float64, 2}
    V3::Array{Float64, 2}
    k_array::Array{Float64, 1}
    Ω²::Array{Float64, 1}
    linear_indices::Array{Array{Int64, 1}, 1}
    full_indices::T
end

struct TemporaryArrays
    Σ1::Array{Float64, 1}
    Σ2::Array{Float64, 1}
    Σ3::Array{Float64, 1}
    A1::Array{Float64, 2}
    A2::Array{Float64, 2}
    A3::Array{Float64, 2}
    q_integral::Array{Float64, 1}
    I_ϕ::Array{Array{Float64, 2}, 1}
    I_kernel::Array{Array{Float64, 2}, 1}
    C1::Array{Array{Float64, 1}, 1}
    C2::Array{Array{Float64, 1}, 1}
    C3::Array{Array{Float64, 1}, 1}
    ϕ_old::Array{Float64, 1}
    kernel_temp::Array{Array{Float64, 2}, 1}
end

mutable struct Results
    ϕ::Array{Array{Float64, 2}, 1}
    kernel::Array{Array{Float64, 2}, 1}
    t_array::Array{Float64, 1}
    ϕ_out::Array{Array{Float64, 1}, 1}
    kernel_evals::Int64
end

function bengtzelius!1(k_array, cache_arrays)
    Nk = length(k_array)
    A1 = cache_arrays.A1
    A2 = cache_arrays.A2
    A3 = cache_arrays.A3
    Σ1 = cache_arrays.Σ1
    Σ2 = cache_arrays.Σ2
    Σ3 = cache_arrays.Σ3
    Σ1 .= 0.0
    Σ2 .= 0.0
    Σ3 .= 0.0
    @inbounds for iq = 1:Nk
        Σ1[1] += A1[iq, iq]
        Σ2[1] += A2[iq, iq]
        Σ3[1] += A3[iq, iq]
    end
end

function bengtzelius!2(k_array, cache_arrays)
    Nk = length(k_array)
    A1 = cache_arrays.A1
    A2 = cache_arrays.A2
    A3 = cache_arrays.A3
    Σ1 = cache_arrays.Σ1
    Σ2 = cache_arrays.Σ2
    Σ3 = cache_arrays.Σ3
    @inbounds for ik = 2:Nk
        s1A1 = 0.0
        s2A1 = 0.0
        s1A2 = 0.0
        s2A2 = 0.0
        s1A3 = 0.0
        s2A3 = 0.0
        qmax = Nk-ik+1
        for iq = 1:qmax
            ip = iq+ik-1
            s1A1 += A1[iq, ip] + A1[ip, iq]
            s1A2 += A2[iq, ip] + A2[ip, iq]
            s1A3 += A3[iq, ip] + A3[ip, iq]
        end
        for iq = 1:ik-1
            ip = ik-iq
            s2A1 += A1[iq, ip]
            s2A2 += A2[iq, ip]
            s2A3 += A3[iq, ip]
        end
        Σ1[ik] = Σ1[ik-1] + s1A1 - s2A1
        Σ2[ik] = Σ2[ik-1] + s1A2 - s2A2
        Σ3[ik] = Σ3[ik-1] + s1A3 - s2A3
    end 
end

function bengtzelius!3(k_array, cache_arrays)
    Nk = length(k_array)
    Σ1 = cache_arrays.Σ1
    Σ2 = cache_arrays.Σ2
    Σ3 = cache_arrays.Σ3
    @inbounds for ik = 1:Nk
        k = k_array[ik]
        invk = 1.0/k
        Σ1[ik] *= k
        Σ2[ik] *= invk*invk*invk
        Σ3[ik] *= invk
    end
end

function bengtzelius!(kernel, k_array, cache_arrays)
    Nk = length(k_array)
    Σ1 = cache_arrays.Σ1
    Σ2 = cache_arrays.Σ2
    Σ3 = cache_arrays.Σ3
    bengtzelius!1(k_array, cache_arrays)
    bengtzelius!2(k_array, cache_arrays)
    bengtzelius!3(k_array, cache_arrays)
    for ik = 1:Nk
        kernel[ik] = (Σ1[ik] + Σ2[ik] + Σ3[ik])
    end
    return 
end

function bengtzelius!_threaded(kernel, k_array, cache_arrays)
    Nk = length(k_array)
    A1 = cache_arrays.A1
    A2 = cache_arrays.A2
    A3 = cache_arrays.A3
    Σ1 = cache_arrays.Σ1
    Σ2 = cache_arrays.Σ2
    Σ3 = cache_arrays.Σ3
    for ik = 1:Nk
        Σ1[ik, threadid()] = 0.0
        Σ2[ik, threadid()] = 0.0
        Σ3[ik, threadid()] = 0.0
    end
    @inbounds for iq = 1:Nk
        Σ1[1, threadid()] += A1[iq, iq, threadid()]
        Σ2[1, threadid()] += A2[iq, iq, threadid()]
        Σ3[1, threadid()] += A3[iq, iq, threadid()]
    end
    @inbounds for ik = 2:Nk
        s1A1 = 0.0
        s2A1 = 0.0
        s1A2 = 0.0
        s2A2 = 0.0
        s1A3 = 0.0
        s2A3 = 0.0
        qmax = Nk-ik+1
        for iq = 1:qmax
            ip = iq+ik-1
            s1A1 += A1[iq, ip, threadid()] + A1[ip, iq, threadid()]
            s1A2 += A2[iq, ip, threadid()] + A2[ip, iq, threadid()]
            s1A3 += A3[iq, ip, threadid()] + A3[ip, iq, threadid()]
        end
        for iq = 1:ik-1
            ip = ik-iq
            s2A1 += A1[iq, ip, threadid()]
            s2A2 += A2[iq, ip, threadid()]
            s2A3 += A3[iq, ip, threadid()]
        end
        Σ1[ik, threadid()] = Σ1[ik-1, threadid()] + s1A1 - s2A1
        Σ2[ik, threadid()] = Σ2[ik-1, threadid()] + s1A2 - s2A2
        Σ3[ik, threadid()] = Σ3[ik-1, threadid()] + s1A3 - s2A3
    end 
    @inbounds for ik = 1:Nk
        Σ1[ik, threadid()] *= k_array[ik]^1
        Σ2[ik, threadid()] *= k_array[ik]^-3
        Σ3[ik, threadid()] *= k_array[ik]^-1
    end
    for ik = 1:Nk
        kernel[ik, threadid()] = Σ1[ik, threadid()] + Σ2[ik, threadid()] + Σ3[ik, threadid()]
    end
    return 
end

function find_kernel1!(kernel, f2, k_array, cache_arrays)
    A1 = cache_arrays.A1
    A2 = cache_arrays.A2
    A3 = cache_arrays.A3
    V1 = cache_arrays.V1
    V2 = cache_arrays.V2
    V3 = cache_arrays.V3
    linear_indices = cache_arrays.linear_indices
    Nk = length(k_array)
    @turbo for ip = 1:Nk
        for iq = 1:Nk
            first_index = ifelse(iq >= ip, iq, ip)
            second_index = ifelse(iq >= ip, ip, iq)
            index = linear_indices[1][first_index] + linear_indices[2][second_index]
            fqp = f2[index]
            A1[iq, ip] = V1[iq, ip] * fqp
            A2[iq, ip] = V2[iq, ip] * fqp
            A3[iq, ip] = V3[iq, ip] * fqp
        end
    end
    bengtzelius!(kernel, k_array, cache_arrays)
end


function find_kernel2!(kernel, cache_arrays, f3, Ω², k_array)
    q_integral = cache_arrays.q_integral
    A1 = cache_arrays.A1
    A2 = cache_arrays.A2
    A3 = cache_arrays.A3
    V1 = cache_arrays.V1
    V2 = cache_arrays.V2
    V3 = cache_arrays.V3
    kernel_temp = cache_arrays.kernel2_temp
    linear_indices = cache_arrays.linear_indices
    Nk = length(k_array)
    for ik2 = 1:Nk 
        thread_id = threadid()
        @inbounds for ip = 1:Nk
            for iq = 1:Nk
                indices = TupleTools.sort((ik2, ip, iq), rev=true)
                index = linear_indices[1][indices[1]] + linear_indices[2][indices[2]] + linear_indices[3][indices[3]]
                fqp = f3[index]
                A1[iq, ip, thread_id] = V1[iq, ip] * fqp
                A2[iq, ip, thread_id] = V2[iq, ip] * fqp
                A3[iq, ip, thread_id] = V3[iq, ip] * fqp
            end
        end
        bengtzelius!(q_integral, k_array, cache_arrays)
        @turbo for ik1 = 1:Nk
            kernel_temp[ik1, ik2] = q_integral[ik1, thread_id] * Ω²[ik1]
        end
    end

    ikk = 1
    @inbounds for ik2 = 1:Nk
        @simd for ik1 = ik2:Nk
            kernel[ikk] = (kernel_temp[ik1, ik2] + kernel_temp[ik2, ik1])/(Ω²[ik1] + Ω²[ik2])
            ikk += 1
        end
    end

end

function find_kernel3!(kernel, cache_arrays, f4, Ω², k_array)
    q_integral = cache_arrays.q_integral
    A1 = cache_arrays.A1
    A2 = cache_arrays.A2
    A3 = cache_arrays.A3
    V1 = cache_arrays.V1
    V2 = cache_arrays.V2
    V3 = cache_arrays.V3
    linear_indices = cache_arrays.linear_indices
    kernel_temp = cache_arrays.kernel3_temp
    Nk = length(k_array)
    for ik3 = 1:Nk
        thread_id = threadid()
        for ik2 = ik3:Nk
            ikk = linear_indices[1][ik2] + linear_indices[2][ik3]
            @inbounds for ip = 1:Nk
                for iq = 1:Nk
                    indices = TupleTools.sort((ik3, ik2, ip, iq), rev=true)
                    index = linear_indices[1][indices[1]] + linear_indices[2][indices[2]] + linear_indices[3][indices[3]] + linear_indices[4][indices[4]]
                    fqp = f4[index]
                    A1[iq, ip, thread_id] = V1[iq, ip] * fqp
                    A2[iq, ip, thread_id] = V2[iq, ip] * fqp
                    A3[iq, ip, thread_id] = V3[iq, ip] * fqp
                end
            end
            bengtzelius!(q_integral, k_array, cache_arrays)
            @turbo for ik1 = 1:Nk
                kernel_temp[ik1, ikk] = q_integral[ik1, thread_id] * Ω²[ik1]
            end
        end
    end
    ikkk = 1
    @inbounds for ik3 = 1:Nk
        for ik2 = ik3:Nk
            for ik1 = ik2:Nk
                index12 = linear_indices[1][ik1] + linear_indices[2][ik2]
                index13 = linear_indices[1][ik1] + linear_indices[2][ik3]
                index23 = linear_indices[1][ik2] + linear_indices[2][ik3]
                kernel[ikkk] = (kernel_temp[ik1, index23] + kernel_temp[ik2, index13] + kernel_temp[ik3, index12])/(Ω²[ik1] + Ω²[ik2] + Ω²[ik3])
                ikkk += 1
            end
        end
    end
end


function find_kernel4!(kernel, cache_arrays, f5, Ω², k_array)
    q_integral = cache_arrays.q_integral
    A1 = cache_arrays.A1
    A2 = cache_arrays.A2
    A3 = cache_arrays.A3
    V1 = cache_arrays.V1
    V2 = cache_arrays.V2
    V3 = cache_arrays.V3
    linear_indices = cache_arrays.linear_indices
    kernel_temp = cache_arrays.kernel4_temp
    Nk = length(k_array)
    for ik4 = 1:Nk
        for ik3 = ik4:Nk
            thread_id = threadid()
            for ik2 = ik3:Nk
                ikkk = linear_indices[1][ik2] + linear_indices[2][ik3] + linear_indices[3][ik4]
                @inbounds for ip = 1:Nk
                    for iq = 1:Nk
                        indices = TupleTools.sort((ik4, ik3, ik2, ip, iq), rev=true)
                        index = linear_indices[1][indices[1]] + linear_indices[2][indices[2]] + linear_indices[3][indices[3]] + linear_indices[4][indices[4]] + linear_indices[5][indices[5]]
                        fqp = f5[index]
                        A1[iq, ip, thread_id] = V1[iq, ip] * fqp
                        A2[iq, ip, thread_id] = V2[iq, ip] * fqp
                        A3[iq, ip, thread_id] = V3[iq, ip] * fqp
                    end
                end
                bengtzelius!(q_integral, k_array, cache_arrays)
                @turbo for ik1 = 1:Nk
                    kernel_temp[ik1, ikkk] = q_integral[ik1, thread_id] * Ω²[ik1]
                end
            end
        end
    end
    ikkkk = 1
    @inbounds for ik4 = 1:Nk
        for ik3 = ik4:Nk
            for ik2 = ik3:Nk
                for ik1 = ik2:Nk
                    index234 = linear_indices[1][ik2] + linear_indices[2][ik3] + linear_indices[3][ik4]
                    index134 = linear_indices[1][ik1] + linear_indices[2][ik3] + linear_indices[3][ik4]
                    index124 = linear_indices[1][ik1] + linear_indices[2][ik2] + linear_indices[3][ik4]
                    index123 = linear_indices[1][ik1] + linear_indices[2][ik2] + linear_indices[3][ik3]
                    kernel[ikkkk] = (kernel_temp[ik1, index234] + kernel_temp[ik2, index134] + kernel_temp[ik3, index124] + kernel_temp[ik4, index123])/(Ω²[ik1] + Ω²[ik2] + Ω²[ik3] + Ω²[ik4])
                    ikkkk += 1
                end
            end
        end
    end
end


function find_kernel1!(parameters, temp_arrays, results::Results, it)
    A1 = temp_arrays.A1
    A2 = temp_arrays.A2
    A3 = temp_arrays.A3
    V1 = parameters.V1
    V2 = parameters.V2
    V3 = parameters.V3
    linear_indices = parameters.linear_indices
    q_integral = temp_arrays.q_integral
    f2 = results.ϕ[2]
    kernel = results.kernel[1]
    k_array = parameters.k_array
    Nk = length(k_array)
    @turbo for ip = 1:Nk
        for iq = 1:Nk
            first_index = ifelse(iq >= ip, iq, ip)
            second_index = ifelse(iq >= ip, ip, iq)
            index = linear_indices[1][first_index] + linear_indices[2][second_index]
            fqp = f2[index, it]
            A1[iq, ip] = V1[iq, ip] * fqp
            A2[iq, ip] = V2[iq, ip] * fqp
            A3[iq, ip] = V3[iq, ip] * fqp
        end
    end
    bengtzelius!(q_integral, parameters.k_array, temp_arrays)
    for ik = 1:Nk
        kernel[ik, it] = q_integral[ik]
    end
end


function find_kernel2!(parameters, temp_arrays, results::Results, it)
    A1 = temp_arrays.A1
    A2 = temp_arrays.A2
    A3 = temp_arrays.A3
    V1 = parameters.V1
    V2 = parameters.V2
    V3 = parameters.V3
    linear_indices = parameters.linear_indices
    q_integral = temp_arrays.q_integral
    kernel_temp = temp_arrays.kernel_temp[2]
    f3 = results.ϕ[3]
    kernel = results.kernel[2]
    Ω² = parameters.Ω²
    k_array = parameters.k_array
    Nk = length(k_array)
    for ik2 = 1:Nk 
        @inbounds for ip = 1:Nk
            for iq = 1:Nk
                indices = TupleTools.sort((ik2, ip, iq), rev=true)
                index = linear_indices[1][indices[1]] + linear_indices[2][indices[2]] + linear_indices[3][indices[3]]
                fqp = f3[index, it]
                A1[iq, ip] = V1[iq, ip] * fqp
                A2[iq, ip] = V2[iq, ip] * fqp
                A3[iq, ip] = V3[iq, ip] * fqp
            end
        end
        bengtzelius!(q_integral, k_array, temp_arrays)
        @inbounds for ik1 = 1:Nk
            kernel_temp[ik1, ik2] = q_integral[ik1] * Ω²[ik1]
        end
    end
    ikk = 1
    @inbounds for ik2 = 1:Nk
         for ik1 = ik2:Nk
            kernel[ikk, it] = (kernel_temp[ik1, ik2] + kernel_temp[ik2, ik1])/(Ω²[ik1] + Ω²[ik2])
            ikk += 1
        end
    end
end

function find_kernel3!(parameters, temp_arrays, results::Results, it)
    A1 = temp_arrays.A1
    A2 = temp_arrays.A2
    A3 = temp_arrays.A3
    V1 = parameters.V1
    V2 = parameters.V2
    V3 = parameters.V3
    linear_indices = parameters.linear_indices
    q_integral = temp_arrays.q_integral
    kernel_temp = temp_arrays.kernel_temp[3]
    f4 = results.ϕ[4]
    kernel = results.kernel[3]
    Ω² = parameters.Ω²
    k_array = parameters.k_array
    Nk = length(k_array)

    for ik3 = 1:Nk
        for ik2 = ik3:Nk
            ikk = linear_indices[1][ik2] + linear_indices[2][ik3]
            @inbounds for ip = 1:Nk
                for iq = 1:Nk
                    indices = TupleTools.sort((ik3, ik2, ip, iq), rev=true)
                    index = linear_indices[1][indices[1]] + linear_indices[2][indices[2]] + linear_indices[3][indices[3]] + linear_indices[4][indices[4]]
                    fqp = f4[index, it]
                    A1[iq, ip] = V1[iq, ip] * fqp
                    A2[iq, ip] = V2[iq, ip] * fqp
                    A3[iq, ip] = V3[iq, ip] * fqp
                end
            end
            bengtzelius!_threaded(q_integral, k_array, temp_arrays)
            @turbo for ik1 = 1:Nk
                kernel_temp[ik1, ikk] = q_integral[ik1] * Ω²[ik1]
            end
        end
    end
    ikkk = 1
    @inbounds for ik3 = 1:Nk
        for ik2 = ik3:Nk
            for ik1 = ik2:Nk
                index12 = linear_indices[1][ik1] + linear_indices[2][ik2]
                index13 = linear_indices[1][ik1] + linear_indices[2][ik3]
                index23 = linear_indices[1][ik2] + linear_indices[2][ik3]
                kernel[ikkk, it] = (kernel_temp[ik1, index23] + kernel_temp[ik2, index13] + kernel_temp[ik3, index12])/(Ω²[ik1] + Ω²[ik2] + Ω²[ik3])
                ikkk += 1
            end
        end
    end
end


function find_kernel4!(parameters, temp_arrays, results::Results, it)
    A1 = temp_arrays.A1
    A2 = temp_arrays.A2
    A3 = temp_arrays.A3
    V1 = parameters.V1
    V2 = parameters.V2
    V3 = parameters.V3
    linear_indices = parameters.linear_indices
    q_integral = temp_arrays.q_integral
    kernel_temp = temp_arrays.kernel_temp[4]
    f5 = results.ϕ[5]
    kernel = results.kernel[4]
    Ω² = parameters.Ω²
    k_array = parameters.k_array
    Nk = length(k_array)

    for ik4 = 1:Nk
        for ik3 = ik4:Nk
            for ik2 = ik3:Nk
                ikkk = linear_indices[1][ik2] + linear_indices[2][ik3] + linear_indices[3][ik4]
                @inbounds for ip = 1:Nk
                    for iq = 1:Nk
                        indices = TupleTools.sort((ik4, ik3, ik2, ip, iq), rev=true)
                        index = linear_indices[1][indices[1]] + linear_indices[2][indices[2]] + linear_indices[3][indices[3]] + linear_indices[4][indices[4]] + linear_indices[5][indices[5]]
                        fqp = f5[index, it]
                        A1[iq, ip] = V1[iq, ip] * fqp
                        A2[iq, ip] = V2[iq, ip] * fqp
                        A3[iq, ip] = V3[iq, ip] * fqp
                    end
                end
                bengtzelius!_threaded(q_integral, k_array, temp_arrays)
                @turbo for ik1 = 1:Nk
                    kernel_temp[ik1, ikkk] = q_integral[ik1] * Ω²[ik1]
                end
            end
        end
    end
    ikkkk = 1
    @inbounds for ik4 = 1:Nk
        for ik3 = ik4:Nk
            for ik2 = ik3:Nk
                for ik1 = ik2:Nk
                    index234 = linear_indices[1][ik2] + linear_indices[2][ik3] + linear_indices[3][ik4]
                    index134 = linear_indices[1][ik1] + linear_indices[2][ik3] + linear_indices[3][ik4]
                    index124 = linear_indices[1][ik1] + linear_indices[2][ik2] + linear_indices[3][ik4]
                    index123 = linear_indices[1][ik1] + linear_indices[2][ik2] + linear_indices[3][ik3]
                    kernel[ikkkk, it] = (kernel_temp[ik1, index234] + kernel_temp[ik2, index134] + kernel_temp[ik3, index124] + kernel_temp[ik4, index123])/(Ω²[ik1] + Ω²[ik2] + Ω²[ik3] + Ω²[ik4])
                    ikkkk += 1
                end
            end
        end
    end
end

# function find_kernel4!(kernel, cache_arrays, f5, Ω², k_array)
#     q_integral = cache_arrays.q_integral
#     A1 = cache_arrays.A1
#     A2 = cache_arrays.A2
#     A3 = cache_arrays.A3
#     V1 = cache_arrays.V1
#     V2 = cache_arrays.V2
#     V3 = cache_arrays.V3
#     linear_indices = cache_arrays.linear_indices
#     kernel_temp = cache_arrays.kernel4_temp
#     Nk = length(k_array)
#     @threads for ik4 = 1:Nk
#         for ik3 = ik4:Nk
#             thread_id = threadid()
#             for ik2 = ik3:Nk
#                 ikkk = linear_indices[1][ik2] + linear_indices[2][ik3] + linear_indices[3][ik4]
#                 @inbounds for ip = 1:Nk
#                     for iq = 1:Nk
#                         indices = TupleTools.sort((ik4, ik3, ik2, ip, iq), rev=true)
#                         index = linear_indices[1][indices[1]] + linear_indices[2][indices[2]] + linear_indices[3][indices[3]] + linear_indices[4][indices[4]] + linear_indices[5][indices[5]]
#                         fqp = f5[index]
#                         A1[iq, ip, thread_id] = V1[iq, ip] * fqp
#                         A2[iq, ip, thread_id] = V2[iq, ip] * fqp
#                         A3[iq, ip, thread_id] = V3[iq, ip] * fqp
#                     end
#                 end
#                 bengtzelius!_threaded(q_integral, k_array, cache_arrays)
#                 @turbo for ik1 = 1:Nk
#                     kernel_temp[ik1, ikkk] = q_integral[ik1, thread_id] * Ω²[ik1]
#                 end
#             end
#         end
#     end
#     ikkkk = 1
#     @inbounds for ik4 = 1:Nk
#         for ik3 = ik4:Nk
#             for ik2 = ik3:Nk
#                 for ik1 = ik2:Nk
#                     index234 = linear_indices[1][ik2] + linear_indices[2][ik3] + linear_indices[3][ik4]
#                     index134 = linear_indices[1][ik1] + linear_indices[2][ik3] + linear_indices[3][ik4]
#                     index124 = linear_indices[1][ik1] + linear_indices[2][ik2] + linear_indices[3][ik4]
#                     index123 = linear_indices[1][ik1] + linear_indices[2][ik2] + linear_indices[3][ik3]
#                     kernel[ikkkk] = (kernel_temp[ik1, index234] + kernel_temp[ik2, index134] + kernel_temp[ik3, index124] + kernel_temp[ik4, index123])/(Ω²[ik1] + Ω²[ik2] + Ω²[ik3] + Ω²[ik4])
#                     ikkkk += 1
#                 end
#             end
#         end
#     end
# end

