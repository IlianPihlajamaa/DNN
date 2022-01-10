function find_closure1!(kernel, fₖ, k_array, cache_arrays)
    A1 = cache_arrays.A1
    A2 = cache_arrays.A2
    A3 = cache_arrays.A3
    V1 = cache_arrays.V1
    V2 = cache_arrays.V2
    V3 = cache_arrays.V3
    q_integral = cache_arrays.q_integral
    Nk = length(fₖ)
    @turbo for iq = 1:Nk
        for ip = 1:Nk
            fq = fₖ[iq]            
            fp = fₖ[ip]
            A1[iq, ip] = V1[iq, ip]* fp*fq
            A2[iq, ip] = V2[iq, ip]* fp*fq
            A3[iq, ip] = V3[iq, ip]* fp*fq
        end
    end
    bengtzelius!(q_integral, k_array, cache_arrays)
    kernel .= q_integral
end

function find_closure1!(parameters, temp_arrays, results::Results, it)
    A1 = temp_arrays.A1
    A2 = temp_arrays.A2
    A3 = temp_arrays.A3
    V1 = parameters.V1
    V2 = parameters.V2
    V3 = parameters.V3
    q_integral = temp_arrays.q_integral
    Nk = parameters.Nk
    ϕ1 = results.ϕ[1]
    kernel = results.kernel[1]
    @turbo for iq = 1:Nk
        for ip = 1:Nk
            fq = ϕ1[iq, it]            
            fp = ϕ1[ip, it]
            A1[iq, ip] = V1[iq, ip]* fp*fq
            A2[iq, ip] = V2[iq, ip]* fp*fq
            A3[iq, ip] = V3[iq, ip]* fp*fq
        end
    end
    bengtzelius!(q_integral, parameters.k_array, temp_arrays)
    for ik = 1:Nk
        kernel[ik, it] = q_integral[ik]
    end
end

function find_closure2_fⁿ!(kernel2, fₖ, k_array, Ω², cache_arrays)
    Nk = length(Ω²)
    A1 = cache_arrays.A1
    A2 = cache_arrays.A2
    A3 = cache_arrays.A3
    V1 = cache_arrays.V1
    V2 = cache_arrays.V2
    V3 = cache_arrays.V3
    q_integral = cache_arrays.q_integral
    Nk = length(fₖ)
    @turbo for iq = 1:Nk
        fq = fₖ[iq]            
        for ip = 1:Nk
            fp = fₖ[ip]
            A1[iq, ip] = V1[iq, ip]* fp*fq
            A2[iq, ip] = V2[iq, ip]* fp*fq
            A3[iq, ip] = V3[iq, ip]* fp*fq
        end
    end
    bengtzelius!(q_integral, k_array, cache_arrays)
    @turbo for ik = 1:Nk
        q_integral[ik] *= Ω²[ik]
    end
    ik = 1
    @inbounds for ik2 = 1:Nk
        @simd for ik1 = ik2:Nk
            kernel2[ik] =  (q_integral[ik1]*fₖ[ik2] + fₖ[ik1]*q_integral[ik2])/(Ω²[ik1] + Ω²[ik2])
            ik += 1
        end
    end
end

function find_closure2_fⁿ!(parameters, temp_arrays, results::Results, it)
    A1 = temp_arrays.A1
    A2 = temp_arrays.A2
    A3 = temp_arrays.A3
    V1 = parameters.V1
    V2 = parameters.V2
    V3 = parameters.V3
    q_integral = temp_arrays.q_integral
    Nk = parameters.Nk
    ϕ1 = results.ϕ[1]
    kernel2 = results.kernel[2]
    Ω² = parameters.Ω²
    @turbo for iq = 1:Nk
        fq = ϕ1[iq, it]            
        for ip = 1:Nk
            fp = ϕ1[ip, it]
            A1[iq, ip] = V1[iq, ip]* fp*fq
            A2[iq, ip] = V2[iq, ip]* fp*fq
            A3[iq, ip] = V3[iq, ip]* fp*fq
        end
    end
    bengtzelius!(q_integral, parameters.k_array, temp_arrays)
    @turbo for ik = 1:Nk
        q_integral[ik] *= Ω²[ik]
    end
    ik = 1
    @inbounds for ik2 = 1:Nk
        @simd for ik1 = ik2:Nk
            kernel2[ik, it] =  (q_integral[ik1]*ϕ1[ik2, it] + ϕ1[ik1, it]*q_integral[ik2])/(Ω²[ik1] + Ω²[ik2])
            ik += 1
        end
    end
end

function find_closure3_fⁿ!(kernel3, cache_arrays, fₖ, Ω², k_array)
    Nk = length(Ω²)
    A1 = cache_arrays.A1
    A2 = cache_arrays.A2
    A3 = cache_arrays.A3
    V1 = cache_arrays.V1
    V2 = cache_arrays.V2
    V3 = cache_arrays.V3
    q_integral = cache_arrays.q_integral
    linear_indices = cache_arrays.linear_indices
    Nk = length(fₖ)
    @tturbo for iq = 1:Nk
        fq = fₖ[iq]            
        for ip = 1:Nk
            fp = fₖ[ip]
            A1[iq, ip] = V1[iq, ip]* fp*fq
            A2[iq, ip] = V2[iq, ip]* fp*fq
            A3[iq, ip] = V3[iq, ip]* fp*fq
        end
    end
    bengtzelius!(q_integral, k_array, cache_arrays)
    @turbo for ik = 1:Nk
        q_integral[ik] *= Ω²[ik]
    end
    # ikkk = 1 
    @batch for ik3 = 1:Nk
        @inbounds for ik2=ik3:Nk
            @simd for ik1=ik2:Nk
                index = linear_indices[1][ik1] + linear_indices[2][ik2] + linear_indices[3][ik3]
                kernel3[index] =  (q_integral[ik1]fₖ[ik2]*fₖ[ik3] + fₖ[ik1]*q_integral[ik2]*fₖ[ik3] + fₖ[ik1]*fₖ[ik2]*q_integral[ik3])/(Ω²[ik1] + Ω²[ik2] + Ω²[ik3])
                # ikkk += 1
            end
        end
    end
end


function find_closure3_fⁿ!(parameters, temp_arrays, results::Results, it)
    k_array = parameters.k_array
    A1 = temp_arrays.A1
    A2 = temp_arrays.A2
    A3 = temp_arrays.A3
    V1 = parameters.V1
    V2 = parameters.V2
    V3 = parameters.V3
    q_integral = temp_arrays.q_integral
    Nk = parameters.Nk
    fₖ = results.ϕ[1]
    kernel3 = results.kernel[3]
    Ω² = parameters.Ω²
    @turbo for iq = 1:Nk
        fq = fₖ[iq, it]            
        for ip = 1:Nk
            fp = fₖ[ip, it]
            A1[iq, ip] = V1[iq, ip]* fp*fq
            A2[iq, ip] = V2[iq, ip]* fp*fq
            A3[iq, ip] = V3[iq, ip]* fp*fq
        end
    end
    bengtzelius!(q_integral, k_array, temp_arrays)
    @turbo for ik = 1:Nk
        q_integral[ik] *= Ω²[ik]
    end
    ikkk = 1 
    for ik3 = 1:Nk
        @inbounds for ik2=ik3:Nk
            for ik1=ik2:Nk
                kernel3[ikkk, it] = (q_integral[ik1]*fₖ[ik2, it]*fₖ[ik3, it] + fₖ[ik1, it]*q_integral[ik2]*fₖ[ik3, it] + fₖ[ik1, it]*fₖ[ik2, it]*q_integral[ik3])/(Ω²[ik1] + Ω²[ik2] + Ω²[ik3])
                ikkk += 1
            end
        end
    end
end

function find_closure4_fⁿ!(kernel4, cache_arrays, fₖ, Ω², k_array)
    Nk = length(Ω²)
    A1 = cache_arrays.A1
    A2 = cache_arrays.A2
    A3 = cache_arrays.A3
    V1 = cache_arrays.V1
    V2 = cache_arrays.V2
    V3 = cache_arrays.V3
    linear_indices = cache_arrays.linear_indices
    q_integral = cache_arrays.q_integral
    Nk = length(fₖ)
    @tturbo for iq = 1:Nk
        fq = fₖ[iq]            
        for ip = 1:Nk
            fp = fₖ[ip]
            A1[iq, ip] = V1[iq, ip]* fp*fq
            A2[iq, ip] = V2[iq, ip]* fp*fq
            A3[iq, ip] = V3[iq, ip]* fp*fq
        end
    end
    bengtzelius!(q_integral, k_array, cache_arrays)
    @turbo for ik = 1:Nk
        q_integral[ik] *= Ω²[ik]
    end
    @batch for ik4 = 1:Nk
        @inbounds for ik3 = ik4:Nk
            for ik2 = ik3:Nk
                @simd for ik1 = ik2:Nk
                    index = linear_indices[1][ik1] + linear_indices[2][ik2] + linear_indices[3][ik3] + linear_indices[4][ik4]
                    kernel4[index] =  (q_integral[ik1]fₖ[ik2]*fₖ[ik3]*fₖ[ik4] + fₖ[ik1]*q_integral[ik2]*fₖ[ik3]*fₖ[ik4] + fₖ[ik1]*fₖ[ik2]*q_integral[ik3]*fₖ[ik4] + fₖ[ik1]*fₖ[ik2]*fₖ[ik3]*q_integral[ik4])/(Ω²[ik1] + Ω²[ik2] + Ω²[ik3] + Ω²[ik4])
                end
            end
        end
    end
end

function find_closure4_fⁿ!(parameters, temp_arrays, results::Results, it)
    k_array = parameters.k_array
    A1 = temp_arrays.A1
    A2 = temp_arrays.A2
    A3 = temp_arrays.A3
    V1 = parameters.V1
    V2 = parameters.V2
    V3 = parameters.V3
    q_integral = temp_arrays.q_integral
    Nk = parameters.Nk
    fₖ = results.ϕ[1]
    kernel4 = results.kernel[4]
    Ω² = parameters.Ω²
    @turbo for iq = 1:Nk
        fq = fₖ[iq, it]            
        for ip = 1:Nk
            fp = fₖ[ip, it]
            A1[iq, ip] = V1[iq, ip]* fp*fq
            A2[iq, ip] = V2[iq, ip]* fp*fq
            A3[iq, ip] = V3[iq, ip]* fp*fq
        end
    end
    bengtzelius!(q_integral, k_array, temp_arrays)
    @turbo for ik = 1:Nk
        q_integral[ik] *= Ω²[ik]
    end
    ikkkk = 1 
    for ik4 = 1:Nk
        for ik3 = ik4:Nk
            @inbounds for ik2=ik3:Nk
                for ik1=ik2:Nk
                    kernel4[ikkkk, it] = (q_integral[ik1]*fₖ[ik2, it]*fₖ[ik3, it]*fₖ[ik4, it] + fₖ[ik1, it]*q_integral[ik2]*fₖ[ik3, it]*fₖ[ik4, it] + fₖ[ik1, it]*fₖ[ik2, it]*q_integral[ik3]*fₖ[ik4, it] + fₖ[ik1, it]*fₖ[ik2, it]*fₖ[ik3, it]*q_integral[ik4])/(Ω²[ik1] + Ω²[ik2] + Ω²[ik3] + Ω²[ik4])
                    ikkkk += 1
                end
            end
        end
    end
end


function find_closure5_fⁿ!(kernel5, cache_arrays, fₖ, Ω², k_array)
    Nk = length(Ω²)
    A1 = cache_arrays.A1
    A2 = cache_arrays.A2
    A3 = cache_arrays.A3
    V1 = cache_arrays.V1
    V2 = cache_arrays.V2
    V3 = cache_arrays.V3
    linear_indices = cache_arrays.linear_indices
    q_integral = cache_arrays.q_integral
    Nk = length(fₖ)
    @tturbo for iq = 1:Nk
        fq = fₖ[iq]            
        for ip = 1:Nk
            fp = fₖ[ip]
            A1[iq, ip] = V1[iq, ip]* fp*fq
            A2[iq, ip] = V2[iq, ip]* fp*fq
            A3[iq, ip] = V3[iq, ip]* fp*fq
        end
    end
    bengtzelius!(q_integral, k_array, cache_arrays)
    @turbo for ik = 1:Nk
        q_integral[ik] *= Ω²[ik]
    end
    @batch for ik5 = 1:Nk
        @inbounds for ik4 = ik5:Nk
            for ik3 = ik4:Nk
                for ik2 = ik3:Nk
                    @simd for ik1 = ik2:Nk
                        index = linear_indices[1][ik1] + linear_indices[2][ik2] + linear_indices[3][ik3] + linear_indices[4][ik4] + linear_indices[5][ik5]
                        kernel5[index] =  (q_integral[ik1]*fₖ[ik2]*fₖ[ik3]*fₖ[ik4]*fₖ[ik5] + fₖ[ik1]*q_integral[ik2]*fₖ[ik3]*fₖ[ik4]*fₖ[ik5] + fₖ[ik1]*fₖ[ik2]*q_integral[ik3]*fₖ[ik4]*fₖ[ik5] + fₖ[ik1]*fₖ[ik2]*fₖ[ik3]*q_integral[ik4]*fₖ[ik5] + fₖ[ik1]*fₖ[ik2]*fₖ[ik3]*fₖ[ik4]*q_integral[ik5])/(Ω²[ik1] + Ω²[ik2] + Ω²[ik3] + Ω²[ik4] + Ω²[ik5])
                    end
                end
            end
        end
    end
end


function find_closure5_fⁿ!(parameters, temp_arrays, results::Results, it)
    k_array = parameters.k_array
    A1 = temp_arrays.A1
    A2 = temp_arrays.A2
    A3 = temp_arrays.A3
    V1 = parameters.V1
    V2 = parameters.V2
    V3 = parameters.V3
    q_integral = temp_arrays.q_integral
    Nk = parameters.Nk
    fₖ = results.ϕ[1]
    kernel5 = results.kernel[5]
    Ω² = parameters.Ω²
    @turbo for iq = 1:Nk
        fq = fₖ[iq, it]            
        for ip = 1:Nk
            fp = fₖ[ip, it]
            A1[iq, ip] = V1[iq, ip]* fp*fq
            A2[iq, ip] = V2[iq, ip]* fp*fq
            A3[iq, ip] = V3[iq, ip]* fp*fq
        end
    end
    bengtzelius!(q_integral, k_array, temp_arrays)
    @turbo for ik = 1:Nk
        q_integral[ik] *= Ω²[ik]
    end
    ikkkkk = 1 
    for ik5 = 1:Nk
        for ik4 = ik5:Nk
            for ik3 = ik4:Nk
                @inbounds for ik2=ik3:Nk
                    for ik1=ik2:Nk
                        kernel5[ikkkkk, it] = (q_integral[ik1]*fₖ[ik2, it]*fₖ[ik3, it]*fₖ[ik4, it]*fₖ[ik5, it] + fₖ[ik1, it]*q_integral[ik2]*fₖ[ik3, it]*fₖ[ik4, it]*fₖ[ik5, it] + fₖ[ik1, it]*fₖ[ik2, it]*q_integral[ik3]*fₖ[ik4, it]*fₖ[ik5, it] + fₖ[ik1, it]*fₖ[ik2, it]*fₖ[ik3, it]*q_integral[ik4]*fₖ[ik5, it] + fₖ[ik1, it]*fₖ[ik2, it]*fₖ[ik3, it]*fₖ[ik4, it]*q_integral[ik5])/(Ω²[ik1] + Ω²[ik2] + Ω²[ik3] + Ω²[ik4] + Ω²[ik5])
                        ikkkkk += 1
                    end
                end
            end
        end
    end
end