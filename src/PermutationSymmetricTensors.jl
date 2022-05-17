module PermutationSymmetricTensors

export SymmetricTensor
export find_full_indices
export find_degeneracy
export find_symmetric_tensor_size
export rand!
using  StaticArrays, Random

"""
SymmetricTensor{T, N, dim} <: AbstractArray{T, dim} 

This is a symmetric tensor object of dimension dim, with N elements of type T in each dimension. 
"""
struct SymmetricTensor{T, N, dim} <: AbstractArray{T, dim}
    data::Array{T, 1}
    linear_indices::Array{Array{Int64, 1}, 1}
end

"""
SymmetricTensor(data::Array{T, 1}, ::Val{N}, ::Val{dim}) where {T, N, dim}

Low level constructor for the SymmetricTensor type. 

    Example:

    N = 10
    dim = 3
    Ndata = find_symmetric_tensor_size(N, dim)
    T = Float64
    data = rand(T, Ndata)
    SymmetricTensor(data, Val(N), Val(dim))
"""
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
    return SymmetricTensor(zeros(T, find_symmetric_tensor_size(N, dim)), Val(N), Val(dim))
end

import Base.rand, Random.rand!
function rand(::Type{SymmetricTensor{T, N, dim}}) where {N, dim, T}
    @assert typeof(N) == typeof(dim) == Int
    return SymmetricTensor(rand(T, find_symmetric_tensor_size(N, dim)), Val(N), Val(dim))
end

function rand(rng::AbstractRNG, ::Type{SymmetricTensor{T, N, dim}}) where {N, dim, T}
    @assert typeof(N) == typeof(dim) == Int
    return SymmetricTensor(rand(rng, T, find_symmetric_tensor_size(N, dim)), Val(N), Val(dim))
end

function rand(rng::AbstractRNG, range::AbstractArray, ::Type{SymmetricTensor{T, N, dim}}) where {N, dim, T}
    @assert typeof(N) == typeof(dim) == Int
    return SymmetricTensor(rand(rng, range, find_symmetric_tensor_size(N, dim)), Val(N), Val(dim))
end

function rand(range::AbstractArray, ::Type{SymmetricTensor{T, N, dim}}) where {N, dim, T}
    @assert typeof(N) == typeof(dim) == Int
    return SymmetricTensor(rand(range, find_symmetric_tensor_size(N, dim)), Val(N), Val(dim))
end

function rand!(A::SymmetricTensor{T, N, dim}) where {N, dim, T}
    rand!(A.data)
    return nothing
end

function rand!(rng::AbstractRNG, A::SymmetricTensor{T, N, dim}) where {N, dim, T}
    rand!(rng, A.data)
    return nothing
end

function rand!(A::SymmetricTensor{T, N, dim}, range::AbstractArray) where {N, dim, T}
    rand!(A.data, range)
    return nothing
end

function rand!(rng::AbstractRNG, A::SymmetricTensor{T, N, dim}, range::AbstractArray) where {N, dim, T}
    rand!(rng, A.data, range)
    return nothing
end

import Base.ones
function ones(::Type{SymmetricTensor{T, N, dim}}) where {N, dim, T}
    @assert typeof(N) == typeof(dim) == Int
    return SymmetricTensor(ones(T, find_symmetric_tensor_size(N, dim)), Val(N), Val(dim))
end

import Base.sizeof
sizeof(A::SymmetricTensor) = sizeof(A.data) + sizeof(A.linear_indices) + sizeof(A.linear_indices[1])*length(A.linear_indices)

import Base.size
size(::SymmetricTensor{T, N, dim}) where {T, N, dim} = ntuple(x->N, dim)

import Base.similar
function similar(A::SymmetricTensor{T, N, dim}) where {T, N, dim}
    return zeros(typeof(A))
end

"""
function find_symmetric_tensor_size(N, dim)

    Returns the number of elements of a symmetric tensor of dimension dim of N elements in each dimension. 
    The results is given by binomial(N-1+dim, dim).

    Example:
    julia> find_symmetric_tensor_size(20, 6)
    177100
"""
find_symmetric_tensor_size(N, dim) = binomial(N-1+dim, dim)


"""
function check_correct_size(N_elements, N, dim)
    checks if the number of elements corresponds to N and dim.
    returns N_elements == find_symmetric_tensor_size(N, dim)
"""
function check_correct_size(N_elements, N, dim)
    return  N_elements == find_symmetric_tensor_size(N, dim)
end


import Base.getindex
"""
function getindex(A::SymmetricTensor{T, N, dim}, I::Int64...) where {T, dim, N}
    This is a custom getindex method for SymmetricTensor types.
    It is implemented with a generated function, for dim = 3, the following code will be executed:

    function get_index(A::SymmetricTensor{T, N, dim}, I::Int64...) where {T, dim, N}
        I2 = sort(SVector(I...), rev=true)    
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
        boundscheck_ex = :(@boundscheck ((I[1]>N^dim || I[1]<1) && throw(BoundsError(A, I))))
        if dim == 1
            index_ex = :(@inbounds A.data[I[1]])
            return :($boundscheck_ex; $index_ex)
        else
            index_ex = :(@inbounds A[CartesianIndices(A)[I[1]]])
            return :($boundscheck_ex; $index_ex)
        end
    elseif length(I) != dim
        return :( throw(DimensionMismatch("This $dim-dimensional symmetric tensor is being indexed with $(length(I)) indices.")))
    end
    ex = :(I2 = sort(SVector(I), rev=true))
    ex1 = :(@boundscheck (I2[1]>N || I2[end]<1) && throw(BoundsError(A, I))) 
    ex2 = :(ind = 0; lin_ind=A.linear_indices)
    for i in 1:dim
        ex2 = :($ex2; @inbounds ind += lin_ind[$i][I2[$i]])
    end
    ex3 = :(@inbounds A.data[ind])
    return ex = :($ex; $ex1; $ex2; $ex3)
end


import Base.setindex!

"""
function setindex!(A::SymmetricTensor{T, N, dim}, value, I::Int64...) where {T, dim, N}
    This is a custom setindex! method for SymmetricTensor types.
    It is implemented with a generated function, for dim = 3, the following code will be executed:

    function set_index!(A::SymmetricTensor{T, N, dim}, value, I::Int64...) where {T, dim, N}
        I2 = sort(SVector(I...), rev=true)     
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
        boundscheck_ex = :(@boundscheck ((I[1]>N^dim || I[1]<1) && throw(BoundsError(A, I))))
        if dim == 1
            index_ex = :(@inbounds A.data[I[1]] = value)
            return :($boundscheck_ex; $index_ex)
        else
            index_ex = :(@inbounds A[CartesianIndices(A)[I[1]]] = value )
            return :($boundscheck_ex; $index_ex)
        end
    elseif length(I) != dim
        return :( throw(DimensionMismatch("This $dim-dimensional symmetric tensor is being indexed with $(length(I)) indices.")))
    end
    ex = :(I2 = sort(SVector(I), rev=true))
    ex1 = :(@boundscheck (I2[1]>N || I2[end]<1) && throw(BoundsError(A, I))) 
    ex2 = :(ind = 0; lin_ind=A.linear_indices)
    for i in 1:dim
        ex2 = :($ex2; @inbounds ind += lin_ind[$i][I2[$i]])
    end
    ex3 = :(@inbounds A.data[ind] = value)
    return ex = :($ex; $ex1; $ex2; $ex3)
end

find_full_indices(::SymmetricTensor{T, N, dim}) where {dim, N, T} = find_full_indices(Int32, N, Val(dim))

function find_full_indices(N, dim) 
    return find_full_indices(Int32, N, Val(dim))
end

"""
function find_full_indices(N, dim)

    Returns an ordered array of tuples of indices (i1, i2, i3, ..., i{dim}) such that 
    i1 >= i2 >= i3 ... >= i{dim}. This can be used to find the cartesian index that 
    corresponds to a linear index of a SymmetricTensor{T, N, dim}. It will automatically
    choose an appropriate integer type to minimize the amount of required storage.

    Example:
    julia> find_full_indices(3, 3)
    10-element Vector{Tuple{Int8, Int8, Int8}}:
    (1, 1, 1)
    (2, 1, 1)
    (3, 1, 1)
    (2, 2, 1)
    (3, 2, 1)
    (3, 3, 1)
    (2, 2, 2)
    (3, 2, 2)
    (3, 3, 2)
    (3, 3, 3)

    It is implemented with a generated function, for dim = 3, the following code will be executed:
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
Generates the following code for dim = 3:
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
function find_N_repetitions_sorted!(reps::Vector{T}, tup) where T<:Integer
    Given a tuple `tup` it will find the number of times a some element of the tuple occurs
    i times in that tuple. It will store the result in the Vector reps at index i.

    Example: 
    reps = zeros(Int, 8)
    tup = (1, 3, 3, 5, 5, 5, 5, 7)
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
function find_degeneracy(N::Int, dim::Int) 
function find_degeneracy(::SymmetricTensor{T, N, dim}) where {dim, N, T} 
function find_degeneracy(N, dim, full_indices)

    returns a SymmetricTensor{Int64, N, dim} of which each element specifies the number of index permutations that point to the same element. 
    for efficiency can be called with the result of `find_degeneracy(N, dim)` as a third argument.

    Examples: 
    julia> find_degeneracy(3, 3)
    3×3×3 SymmetricTensor{Int64, 3, 3}:
    [:, :, 1] =
    1  3  3
    3  3  6
    3  6  3

    [:, :, 2] =
    3  3  6
    3  1  3
    6  3  3

    [:, :, 3] =
    3  6  3
    6  3  3
    3  3  1

    julia> a = rand(SymmetricTensor{Float64, 2,4});

    julia> find_degeneracy(a)
    2×2×2×2 SymmetricTensor{Int64, 2, 4}:
    [:, :, 1, 1] =
    1  4
    4  6

    [:, :, 2, 1] =
    4  6
    6  4

    [:, :, 1, 2] =
    4  6
    6  4

    [:, :, 2, 2] =
    6  4
    4  1

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