module PermutationSymmetricTensors

export SymmetricTensor
export find_full_indices
export find_degeneracy
export symmetric_tensor_size

using  TupleTools, Random

struct SymmetricTensor{T, N, dim} <: AbstractArray{T, dim}
    data::Array{T, 1}
    linear_indices::Array{Array{Int64, 1}, 1}
end

function SymmetricTensor(data::Array{T, 1}, ::Val{N}, ::Val{dim}) where {T, N, dim}
    @assert typeof(N) == typeof(dim) == Int
    if !check_correct_size(length(data), N, dim)
        throw(ArgumentError("Size is wrong. The given size is $N_elements, while it should be $(binomial(N-1+dim,dim))"))
    end
    linear_indices = find_linear_indices(Val(N), Val(dim))
    SymmetricTensor{T, N, dim}(data, linear_indices)
end

import Base.zeros
function zeros(::Type{SymmetricTensor{T, N, dim}}) where {N, dim, T}
    @assert typeof(N) == typeof(dim) == Int
    return SymmetricTensor(zeros(T, symmetric_tensor_size(N, dim)), Val(N), Val(dim))
end

import Base.rand
function rand(::Type{SymmetricTensor{T, N, dim}}) where {N, dim, T}
    @assert typeof(N) == typeof(dim) == Int
    return SymmetricTensor(rand(T, symmetric_tensor_size(N, dim)), Val(N), Val(dim))
end

import Base.ones
function ones(::Type{SymmetricTensor{T, N, dim}}) where {N, dim, T}
    @assert typeof(N) == typeof(dim) == Int
    return SymmetricTensor(ones(T, symmetric_tensor_size(N, dim)), Val(N), Val(dim))
end

import Base.sizeof
sizeof(A::SymmetricTensor) = sizeof(A.data) + sizeof(A.linear_indices) + sizeof(A.linear_indices[1])*length(A.linear_indices)

import Base.length
length(::SymmetricTensor{T, N, dim}) where {T, N, dim} = N^dim

import Base.size
size(::SymmetricTensor{T, N, dim}) where {T, N, dim} = ntuple(x->N, dim)

import Base.ndims
ndims(::Type{SymmetricTensor{T, N, dim}} where {T, dim, N}) = dim 

import Base.axes
axes(::Type{SymmetricTensor{T, N, dim}} where {T, dim, N}) = ntuple(x->Base.OneTo(Val(N)), dim)

# import Base.copyto!
# copyto!(dest::SymmetricTensor, args...) = copyto!(dest.data, args...)

import Base.iterate
@inline function iterate(A::SymmetricTensor{T, N, dim}, i=1) where {T, N, dim}
    (i % UInt) - 1 < length(A) ? (@inbounds A[CartesianIndices(A)[i]], i + 1) : nothing
end

#Similar

function symmetric_tensor_size(N, dim)
    return binomial(N-1+dim, dim)
end

function check_correct_size(N_elements, N, dim)
    if N_elements == symmetric_tensor_size(N, dim)
        return true
    else
        return false
    end
end

import Base.getindex
"""
generated expression, executes

function get_index(A::SymmetricTensor{N, dim, T}, I::Int64...) where {T, dim, N}
    I2 = TupleTools.sort(I, rev = true)     
    ind = 0
    @inbounds begin 
        ind += (A.linear_indices[1])[I2[1]]
        ind += (A.linear_indices[2])[I2[2]]
        ind += (A.linear_indices[3])[I2[3]]
        return A.data[ind]
    end
end
"""
@generated function getindex(A::SymmetricTensor{T, N, dim}, I::Int64...) where {T, dim, N}
    if length(I) == 1
        return :(@inbounds A[CartesianIndices(A)[I[1]]])
    elseif length(I) != dim
        return :(throw(DimensionMismatch("This $dim-dimensional symmetric tensor is being indexed with $(length(I)) indices.")))
    end
    ex = :(I2 = TupleTools.sort(I, rev=true))
    ex1 = :(@boundscheck (I2[1]>N || I2[end]<1) && throw(BoundsError(A, I))) 
    ex2 = :(ind = 0)
    for i in 1:dim
        ex2 = :($ex2; @inbounds ind += A.linear_indices[$i][I2[$i]])
    end
    ex3 = :(@inbounds A.data[ind])
    return ex = :($ex; $ex1; $ex2; $ex3)
end

import Base.setindex!

"""
generated expression, for dim=3, executes

function set_index!(A::SymmetricTensor{N, dim, T}, value, I::Int64...) where {T, dim, N}
    I2 = TupleTools.sort(I, rev = true)     
    ind = 0
    @inbounds begin 
        ind += (A.linear_indices[1])[I2[1]]
        ind += (A.linear_indices[2])[I2[2]]
        ind += (A.linear_indices[3])[I2[3]]
        A.data[ind] = value
    end
end
"""
@generated function setindex!(A::SymmetricTensor{T, N, dim}, value, I::Int64...) where {T, dim, N}
    if length(I) == 1
        return :(@inbounds A[CartesianIndices(A)[I[1]]] = value)
    elseif length(I) != dim
        return :(throw(DimensionMismatch("This $dim-dimensional symmetric tensor is being indexed with $(length(I)) indices.")))
    end
    ex = :(I2 = TupleTools.sort(I, rev=true))
    ex1 = :(@boundscheck (I2[1]>N || I2[end]<1) && throw(BoundsError(A, I))) 
    ex2 = :(ind = 0)
    for i in 1:dim
        ex2 = :($ex2; @inbounds ind += A.linear_indices[$i][I2[$i]])
    end
    ex3 = :(@inbounds A.data[ind] = value)
    return ex = :($ex; $ex1; $ex2; $ex3)
end

function find_full_indices(N, dim) 
    N < typemax(Int8) && return find_full_indices(Int8, N, Val(dim))
    N < typemax(Int16) && return find_full_indices(Int16, N, Val(dim))
    N < typemax(Int32) && return find_full_indices(Int32, N, Val(dim))
    return _find_full_indices(Int64, N, Val(dim))
end

"""
Generated function. For dim=3, evaluates:
    function _find_full_indices(N, Val(3))
    full_indices = NTuple{3, Int16}[]
    for i3 = 1:N
        for i2 = i3:N
            for i1 = i2:N
                push!(full_indices, ((i1..., i2)..., i3))
            end
        end
    end
    full_indices
end
"""
@generated function find_full_indices(T, N, ::Val{dim}) where {dim}
    if dim == 1
        return :(full_indices = Tuple{T}[]; for i = 1:N; push!(full_indices, (T(i),)); end; full_indices)
    end
    ex = :(full_indices = NTuple{$dim, T}[])
    tupleex = :(i1)
    for i = 2:dim
        ii = Symbol("i$i")
        tupleex = :($tupleex..., $ii)
    end

    ex2 = :(push!(full_indices, $tupleex))
    for i = 1:dim
        ii = Symbol("i$i")
        start = i == dim ? 1 : Symbol("i$(i+1)")
        ex2 = :(for $ii = $start:N; $ex2; end)
    end
    return :($ex; $ex2; full_indices)
end

"""
generated function:
generates the following code for dim = 3
    function find_linear_index_array(N::Int, ::Val{3})
        idim_contribution_array = zeros(Int64, N)
        contribution = 0
        count = 0
        firstcount = 0
        for i3 = 1:N
            for i2 = i3:N
                for i1 = i2:N
                    count += 1
                    if ((i1 == i2) && i2 == i3)
                        if i3 == 1
                            firstcount = count
                        end
                        contribution = count - firstcount
                        idim_contribution_array[i3] = contribution
                    end
                end
            end
        end
        idim_contribution_array
    end
"""
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
    for i = 1:dim
        ii = Symbol("i$i")
        i_iplus1 = i!=dim ? Symbol("i$(i+1)") : 1
        ex2 = :(for $ii = $i_iplus1:N; $ex2; end)
    end
    return :($ex; $ex2; idim_contribution_array)
end

function find_linear_indices(::Val{N}, ::Val{dim}) where {N, dim}
    contributions = Array{Int64, 1}[]
    i = 1
    while i <= dim
        push!(contributions, find_linear_index_array(N, Val(i)))
        i += 1
    end
    return contributions
end


"""
example 
reps = zeros(Int, 8)
tup = (1,3,3,5,5,5,5,7)
find_N_repetitions_sorted!(reps, tup)

julia> find_N_repetitions_sorted!(reps, tup)
8-element Vector{Int64}:
 2
 1
 0
 1
 0
 0
 0
 0
 """
function find_N_repetitions_sorted!(reps::Vector{T}, tup) where T<:Integer
    reps .= 0
    lastpos = 1
    reps[1] = 1
    for i = 2:length(tup)
        if tup[i] != tup[i-1]
            reps[1] += 1
            lastpos = 1
        else
            lastpos += 1
            reps[lastpos-1] -= 1
            reps[lastpos] += 1
        end
    end
end

find_degeneracy(N::Int, dim::Int) = find_degeneracy(N, dim, find_full_indices(N, dim))
find_degeneracy(::SymmetricTensor{T, N, dim}) where {dim, N, T} = find_degeneracy(N, dim, find_full_indices(N, dim))

"""
function find_degeneracy(::SymmetricTensor{T, N, dim}, full_indices) where {dim, N, T}
    returns a SymmetricTensor{Int32, N, dim} of which each element specifies the number of index permutations that point to the same element. 
    For a symmetric matrix for example, this would return a matrix filled with 2s with one on the diagonal.
"""
function find_degeneracy(N, dim, full_indices)
    mult = zeros(SymmetricTensor{Int64, N, dim}) 
    factdim = factorial(dim)
    reps = zeros(Int, dim)
    for i in eachindex(mult.data)
        tup = full_indices[i]
        find_N_repetitions_sorted!(reps, tup)
        mult.data[i] = factdim
        for irep = 2:dim
            rep = reps[irep]
            if rep > 0
                mult.data[i] /= factorial(irep)^rep
            end
        end
    end
    return mult
end


end # 