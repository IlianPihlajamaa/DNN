using BenchmarkTools, LoopVectorization, TupleTools, Random

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.2
function symmetric_tensor_size(N, dim)
    return binomial(N-1+dim,dim)
end    

@generated function find_linear_index_array(N::Int, ::Val{dim}) where dim
    if dim == 1
        return :(collect(1:N))
    end
    ex = :(idim_contribution_array = zeros(Int64, N); contribution = 0; count = 0; firstcount = 0)
    ii = Symbol("i$dim")
    ex2 = :(if $ii == 1; firstcount = count; end; contribution = count - firstcount; idim_contribution_array[$ii] = contribution) 
    equalex = :(true)
    for j = 2:dim-1
        ij = Symbol("i$j")
        ijmin1 = Symbol("i$(j-1)")
        equalex = :($equalex && $ijmin1 == $ij)
    end
    iimin1 = Symbol("i$(dim-1)")
    equalex = :($equalex && $iimin1 == N)
    ex2 = :(count += 1; if $equalex; $ex2; end)
    ifexpr = :(true)
    for j = 2:dim
        ij = Symbol("i$j")
        ijmin1 = Symbol("i$(j-1)")
        ifexpr = :($ifexpr && $ijmin1 >= $ij)
    end
    ex2 = :(if $ifexpr; $ex2; end)
    for i = 1:dim
        ii = Symbol("i$i")
        ex2 = :(for $ii = 1:N; $ex2; end)
    end
    return :($ex; $ex2; idim_contribution_array)
end

function find_linear_indices(N, dim)
    contributions = Array{Int64, 1}[]
    i = 1
    while i <= dim
        push!(contributions, find_linear_index_array(N, Val(i)))
        i += 1
    end
    return contributions
end

function check_correct_size(N_elements, N, dim)
    if N_elements == symmetric_tensor_size(N, dim)
        return
    else
        throw(ArgumentError("Size is wrong. The given size is $N_elements, while it should be $(binomial(N-1+dim,dim))"))
        return
    end
end

struct SymmetricTensor{N, dim, T<:Number}
    data::Array{T, 1}
    linear_indices::Array{Array{Int64, 1}, 1}
end

function SymmetricTensor(data::Array{T, 1}, ::Val{N}, ::Val{dim}) where {T, N, dim}
    check_correct_size(length(data), N, dim)
    linear_indices = find_linear_indices(N, dim)
    SymmetricTensor{N, dim, T}(data, linear_indices)
end

import Base.zeros
function zeros(::Type{SymmetricTensor}, N::Int64, dim::Int64, T::Type{<:Number}) 
    return SymmetricTensor(zeros(T, binomial(N-1+dim,dim)), Val(N), Val(dim))
end

import Base.rand
function rand(::Type{SymmetricTensor}, N::Int64, dim::Int64, T::Type{<:Number}) 
    return SymmetricTensor(rand(T, binomial(N-1+dim,dim)), Val(N), Val(dim))
end

find_full_indices(N, dim) = _find_full_indices(N, Val(dim))

@generated function _find_full_indices(N, ::Val{dim}) where {dim}
    if dim == 1
        return :(full_indices = Tuple{Int16}[]; for i = 1:N; push!(full_indices, (Int16(i),)); end; full_indices)
    end
    ex = :(full_indices = NTuple{$dim, Int16}[])
    tupleex = :(i1)
    for i = 2:dim
        ii = Symbol("i$i")
        tupleex = :($tupleex..., $ii)
    end

    forex = :(i1 >= i2)
    for i = 3:dim
        ii = Symbol("i$i")
        imin1 = Symbol("i$(i-1)")
        forex = :($forex && $imin1 >= $ii)
    end

    ex2 = :(push!(full_indices, $tupleex))
    ex2 = :(if $forex; $ex2; end)
    for i = 1:dim
        ii = Symbol("i$i")
        ex2 = :(for $ii = 1:N; $ex2; end)
    end
    return :($ex; $ex2; full_indices)
end

import Base.length
function length(A::SymmetricTensor) 
    return length(A.data)
end

import Base.size
function size(A::SymmetricTensor) 
    return size(A.data)
end

import Base.getindex
@generated function getindex(A::SymmetricTensor{N, dim, T}, I::Int64...) where {T, dim, N}
    if length(I) == 1
        return :(@inbounds A.data[I[1]])
    end
    ex = :(I2 = TupleTools.sort(I, rev=true))
    ex2 = :(ind = 0)
    for i in 1:dim
        ex2 = :($ex2; ind += A.linear_indices[$i][I2[$i]])
    end
    ex3 = :(A.data[ind])
    return ex = :($ex; @inbounds $ex2; @inbounds $ex3)
end

import Base.sizeof
sizeof(A::SymmetricTensor) = sizeof(A.data) + sum([sizeof(A.linear_indices[i]) for i = 1:length(A.linear_indices)])

import Base.setindex!

@generated function setindex!(A::SymmetricTensor{N, dim, T}, value, I::Int64...) where {T, dim, N}
    if length(I) == 1
        return :(@inbounds A.data[I[1]] = value)
    end
    ex = :(I2 = TupleTools.sort(I, rev=true))
    ex2 = :(ind = 0)
    for i in 1:dim
        ex2 = :($ex2; @inbounds ind += A.linear_indices[$i][I2[$i]])
    end
    ex3 = :(@inbounds A.data[ind] = value)
    return ex = :($ex; $ex2; $ex3)
end

import Base.ndims
function ndims(S::SymmetricTensor) 
    return ndims(S.data)
end

import Base.ndims
ndims(S::Type{SymmetricTensor{N, dim, T}} where {T, dim, N}) = 1 

import Base.copyto!
copyto!(dest::SymmetricTensor, args...) = copyto!(dest.data, args...)

import Base.iterate
iterate(dest::SymmetricTensor, args...) = iterate(dest.data, args...)

import Base.axes
axes(::Type{SymmetricTensor{N, dim, T}} where {T, dim, N}) = ntuple(x->Base.OneTo(Val(N)), dim)

@generated function find_multiplicity(::SymmetricTensor{N, dim, Float64}, full_indices) where {dim, N}
    ex = :(mult = zeros(SymmetricTensor, $N, $dim, Int32))
    ex2 = :(uniques = Int64[]; push!(uniques, 1))
    ex3 = :(full_i = full_indices[i])
    for i = 2:dim
        ii = :(full_i[$i])
        ifexp = :($ii != full_i[1])
        for j = 2:i-1
            ij = :(full_i[$j])
            ifexp = :($ifexp && $ii != $ij)
        end
        ex3 = :($ex3; if $ifexp; push!(uniques, 1); else; uniques[end] += 1; end)
    end
    ex3 = :($ex3; mult[i] = factorial($dim)/prod(factorial.(uniques)))
    return :($ex; for i = 1:length(mult); $ex2; $ex3; end; mult)
end

import Base.sum
function sum(S::SymmetricTensor{N, dim, T}, multiplicity) where {T, dim, N}
    s = 0.0
    @simd for i = 1:length(S)
        @inbounds @fastmath s += S.data[i] * multiplicity.data[i]
    end
    return s
end

@generated function sum2(S::SymmetricTensor{N, dim, T}, multiplicity) where {T, dim, N}
    ex = :(s = 0.0)
    ex = :($ex; linear_indices = S.linear_indices)
    linindexdim = Symbol("linear_index$dim")
    ex2 = :(s += S[$linindexdim]*multiplicity[$linindexdim])
    for i = dim:-1:2
        ii =  Symbol("i$i")
        iimin1 = Symbol("i$(i-1)")
        linindexi = Symbol("linear_index$i")
        linindeximin1 = Symbol("linear_index$(i-1)")
        ex2 = :(for $ii = 1:$iimin1; $linindexi = $linindeximin1 + linear_indices[$i][$ii]; $ex2; end)
    end
    ex2 = :(@inbounds for i1 = 1:$N; linear_index1 = linear_indices[1][i1]; $ex2; end)
    ex = :($ex; $ex2; s)
    return ex
end

import Base.copy!
@generated function copy!(A::Array{T, dim}, S::SymmetricTensor{N, dim, T}) where {N, dim, T}
    ex2 = :(i1)
    for i = 2:dim
        ii = Symbol("i$i")
        ex2 = :($ex2..., $ii)
    end
    ex = :(A[$ex2...] = S[$ex2...])
    for i = 1:dim
        ii = Symbol("i$i")
        ex = :(for $ii = 1:$N; $ex; end)
    end
    return ex
end

function test_symm_tensors(N, dim)
    println("\n\nTesting $dim-dimensional tensor with $N elements in each dimension")
    println("This tensor takes $(round(8*symmetric_tensor_size(N, dim)/2^20, digits=2)) MB of memory, instead of $(round(8*N^dim/2^20, digits=2))MB that the full one would take")
    S = rand(SymmetricTensor, N, dim, Float64)
    full_indices = find_full_indices(N, dim)
    multiplicity = find_multiplicity(S, full_indices)
    tensorsize = length(multiplicity)

    A = zeros(ntuple(x->N, dim)...)
    copy!(A, S)

    s =  sum(S, multiplicity)
    s2 = sum(A)
    s3 = sum2(S, multiplicity)
    println("Timing sum of tensor.")
    println("Symmetric: efficient")
    @btime  sum($S, $multiplicity)
    println("Symmetric: worse")
    @btime  sum2($S, $multiplicity)
    println("Full:")
    @btime sum($A)
    println("\nTests:")
    println(s2 ≈ s ≈ s3)
    test_index = rand(1:tensorsize)
    test_idx = rand(1:N, dim)
    println(test_index == sum([S.linear_indices[i][index] for (i,index) in enumerate(full_indices[test_index])]))
    println(S[test_idx...] == S[test_idx[randperm(dim)]...])
end

# N = 10
# for d = 1:9
#     test(N, d)
# end
