"""
This module provides the `SymmetricTensor` type, representing a tensor whose elements are symmetric under any permutation of their indices.
It allows for efficient storage and manipulation of such tensors.

Key functionalities include:
- Creating `SymmetricTensor` instances (e.g., with random values, zeros, ones).
- Indexing into the tensor using standard array-like notation.
- Calculating the number of unique elements required to store the tensor using `find_symmetric_tensor_size`.
- Determining the degeneracy (number of equivalent permutations) for each element using `find_degeneracy`.
- Retrieving the unique sorted Cartesian indices corresponding to the stored elements via `find_full_indices`.
"""
module PermutationSymmetricTensors

export SymmetricTensor
export find_full_indices
export find_degeneracy
export find_symmetric_tensor_size
export rand!
using  StaticArrays, Random

"""
`SymmetricTensor{T, N, dim} <: AbstractArray{T, dim}`

A tensor of `dim` dimensions, where each dimension has `N` elements of type `T`.
The tensor is symmetric under permutation of its indices.

# Fields
- `data::Vector{T}`: A flat vector storing the unique elements of the symmetric tensor.
  The length of this vector is determined by `find_symmetric_tensor_size(N, dim)`.
- `linear_indices::Vector{Vector{Int64}}`: Precomputed indices to map sorted Cartesian
  indices to the linear index in the `data` vector. This is an internal field used
  for efficient indexing.
"""
struct SymmetricTensor{T, N, dim} <: AbstractArray{T, dim}
    data::Array{T, 1} # TODO: Consider changing to Vector{T} for consistency with docstring
    linear_indices::Array{Array{Int64, 1}, 1} # TODO: Consider Vector{Vector{Int64}}
end

"""
`SymmetricTensor(data::Array{T, 1}, ::Val{N}, ::Val{dim}) where {T, N, dim}`

Low level constructor for the SymmetricTensor type. 

Example:
```julia
N = 10
dim = 3
Ndata = find_symmetric_tensor_size(N, dim)
T = Float64
data = rand(T, Ndata)
SymmetricTensor(data, Val(N), Val(dim))
```
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

"""
`rand!(A::SymmetricTensor, [rng::AbstractRNG], [values])`

Fill the symmetric tensor `A` with random values.

This function populates the underlying data vector of the `SymmetricTensor` with
random numbers. Due to the tensor's symmetry, only the unique elements are stored
and randomized.

# Arguments
- `A::SymmetricTensor`: The symmetric tensor to be filled with random values.
  It is modified in-place.
- `rng::AbstractRNG` (optional): A specific random number generator to use.
  If not provided, the default global RNG is used.
- `values` (optional): A collection of values to sample from (e.g., a range like `0:9`,
  or a specific set like `[1.0, 2.5, 3.0]`). If not provided, `rand` will produce
  values of the tensor's element type (e.g., `Float64` in `[0,1)`).

# Returns
- `A`: The modified tensor `A`, filled with random values. (Note: `rand!` traditionally returns the modified array, but the current implementation returns `nothing`. This docstring reflects the traditional behavior for consistency with `Base.rand!`, though the implementation detail differs.)


# Examples
```julia
julia> N = 2; dim = 2;
julia> ts = zeros(SymmetricTensor{Float64, N, dim});

julia> rand!(ts); # Fill with random Float64 values
julia> ts[1,1] # Will be a random Float64

julia> rand!(ts, MersenneTwister(123)); # Using a specific RNG
julia> ts[1,2] # Will be a random Float64

julia> rand!(ts, 1:10); # Fill with random integers from 1 to 10
julia> ts[2,2] # Will be a random integer between 1 and 10
```
"""
function rand!(A::SymmetricTensor{T, N, dim}) where {N, dim, T}
    rand!(A.data)
    return A
end

function rand!(rng::AbstractRNG, A::SymmetricTensor{T, N, dim}) where {N, dim, T}
    rand!(rng, A.data)
    return A
end

function rand!(A::SymmetricTensor{T, N, dim}, range::AbstractArray) where {N, dim, T}
    rand!(A.data, range)
    return A
end

function rand!(rng::AbstractRNG, A::SymmetricTensor{T, N, dim}, range::AbstractArray) where {N, dim, T}
    rand!(rng, A.data, range)
    return A
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
function similar(::SymmetricTensor{T, N, dim}) where {T, N, dim}
    return SymmetricTensor(Vector{T}(undef, find_symmetric_tensor_size(N, dim)), Val(N), Val(dim))
end

function similar(::SymmetricTensor{T, N, dim}, ::Type{T_new}) where {T_new, T, N, dim}
    return SymmetricTensor(Vector{T_new}(undef, find_symmetric_tensor_size(N, dim)), Val(N), Val(dim))
end

import Base.length
@generated length(::SymmetricTensor{T, N, dim}) where {T, N, dim} = Int128(N)^dim > typemax(Int) ? :(Int128(N)^dim) : :(N^dim)


"""
`function find_symmetric_tensor_size(N, dim)`

Returns the number of elements of a symmetric tensor of dimension dim of N elements in each dimension. 
The results is given by binomial(N-1+dim, dim).

Example:
```julia`
julia> find_symmetric_tensor_size(20, 6)
177100
```
"""
find_symmetric_tensor_size(N, dim) = binomial(N-1+dim, dim)


"""
`check_correct_size(num_elements::Int, N::Int, dim::Int) -> Bool`

Internal helper function to verify if `num_elements` is the correct count
for a `SymmetricTensor` with dimension `dim` and size `N` for each dimension.

# Arguments
- `num_elements::Int`: The number of elements to check (typically `length(data)`).
- `N::Int`: The size of each dimension.
- `dim::Int`: The number of dimensions.

# Returns
- `Bool`: `true` if `num_elements` matches `find_symmetric_tensor_size(N, dim)`, `false` otherwise.
"""
function check_correct_size(num_elements, N, dim)
    return  num_elements == find_symmetric_tensor_size(N, dim)
end


import Base.getindex
"""
`getindex(A::SymmetricTensor{T, N, dim}, I::Int...) -> T`

Retrieve the element at the specified indices `I` from the symmetric tensor `A`.
The indices `I` can be provided in any order; due to the tensor's symmetry,
`A[i, j, k]` is equivalent to `A[k, j, i]`, etc.

The method also supports linear indexing if a single index is provided.

# Arguments
- `A::SymmetricTensor{T, N, dim}`: The symmetric tensor to access.
- `I::Int...`: A sequence of `dim` integer indices, or a single linear index.

# Returns
- `T`: The element at the specified position.

# Examples
```julia
julia> tensor = ones(SymmetricTensor{Float64, 2, 3});
julia> tensor[1, 2, 1]
1.0

julia> tensor[1, 1, 2] # Same as above due to symmetry
1.0

julia> tensor[2] # Linear indexing (equivalent to tensor[2,1,1] in this case based on internal order)
1.0
```

This method is implemented using a `@generated` function for efficiency, which
constructs specialized code based on the tensor's dimension (`dim`). For example,
for `dim = 3`, the internal logic effectively sorts the indices and uses precomputed
values to find the element in the underlying `data` vector.
"""
@generated function getindex(A::SymmetricTensor{T, N, dim}, I::Int64...) where {T, dim, N}
    boundscheck_ex1 = :(@boundscheck ((I[1]>length(A) || I[1]<1) && throw(BoundsError(A, I))))
    if dim == 1 && length(I) == 1 
        index_ex = :(@inbounds A.data[I[1]])
        return :($boundscheck_ex1; $index_ex)
    end
    if dim == 1 && length(I) == 2 
        check_ex = :(I[2] == 1 || throw(DimensionMismatch("This $dim-dimensional symmetric tensor is being indexed with $(length(I)) indices.")))
        index_ex = :(@inbounds A.data[I[1]])
        return :($boundscheck_ex1; $check_ex; $index_ex)
    end
    if dim > 1 && length(I) == 1 
        if big(N)^dim > typemax(Int)
            index_ex = :(@inbounds A[Int128(I[1])])
            return :($boundscheck_ex1; $index_ex)
        else
            index_ex = :(@inbounds A[CartesianIndices(A)[I[1]]])
            return :($boundscheck_ex1; $index_ex)
        end
    end
    if length(I) != dim
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
`setindex!(A::SymmetricTensor{T, N, dim}, value, I::Int...) -> typeof(value)`

Set the element at the specified indices `I` in the symmetric tensor `A` to `value`.
The indices `I` can be provided in any order; due to the tensor's symmetry,
setting `A[i, j, k]` will also affect permutations like `A[k, j, i]`.

The method also supports linear indexing if a single index is provided.

# Arguments
- `A::SymmetricTensor{T, N, dim}`: The symmetric tensor to modify.
- `value`: The value to assign to the element.
- `I::Int...`: A sequence of `dim` integer indices, or a single linear index.

# Returns
- The assigned `value`.

# Examples
```julia
julia> tensor = zeros(SymmetricTensor{Float64, 2, 3});
julia> tensor[1, 2, 1] = 5.0;
julia> tensor[1, 1, 2]
5.0

julia> tensor[1] = 3.0; # Linear indexing
julia> tensor[1,1,1] # Assuming [1,1,1] is the first linear index
3.0
```

This method is implemented using a `@generated` function for efficiency, which
constructs specialized code based on the tensor's dimension (`dim`). For example,
for `dim = 3`, the internal logic effectively sorts the indices and uses precomputed
values to find the element in the underlying `data` vector to update.
"""
@generated function setindex!(A::SymmetricTensor{T, N, dim}, value, I::Int64...) where {T, dim, N}
    boundscheck_ex1 = :(@boundscheck ((I[1]>length(A) || I[1]<1) && throw(BoundsError(A, I))))
    if dim == 1 && length(I) == 1 
        index_ex = :(@inbounds A.data[I[1]] = value)
        return :($boundscheck_ex1; $index_ex)
    end
    if dim == 1 && length(I) == 2 
        check_ex = :(I[2] == 1 || throw(DimensionMismatch("This $dim-dimensional symmetric tensor is being indexed with $(length(I)) indices.")))
        index_ex = :(@inbounds A.data[I[1]] = value)
        return :($boundscheck_ex1; $check_ex; $index_ex)
    end
    if dim > 1 && length(I) == 1 
        if big(N)^dim > typemax(Int)
            index_ex = :(@inbounds A[Int128(I[1])] = value)
            return :($boundscheck_ex1; $index_ex)
        else
            index_ex = :(@inbounds A[CartesianIndices(A)[I[1]]] = value)
            return :($boundscheck_ex1; $index_ex)
        end
    end
    if length(I) != dim
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
`function find_full_indices(N, dim)`

Returns an ordered array of tuples of indices `(i1, i2, i3, ..., i{dim})` such that 
`i1 >= i2 >= i3 ... >= i{dim}`. This can be used to find the cartesian index that 
corresponds to a linear index of a `SymmetricTensor{T, N, dim}`. 
Example:
```julia
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
```
It is implemented with a generated function, for dim = 3, the following code will be executed:
```julia
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
```
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
`find_linear_index_array(N::Int, ::Val{dim}) -> Vector{Int64}`

Internal `@generated` function to compute a vector of index contributions for a specific
dimension, used in calculating the linear index into the `data` array of a `SymmetricTensor`.

This function is part of the mechanism that maps multi-dimensional indices
`(i1, i2, ..., idim)` (sorted descendingly) to a unique linear index.
The `SymmetricTensor` stores `dim` such vectors in its `linear_indices` field.
Each vector `A.linear_indices[k]` corresponds to `find_linear_index_array(N, Val(k))`.

The linear index for `(I1, I2, ..., Ik, ..., Idim)` (where `Ik` are sorted indices)
is roughly `sum(A.linear_indices[k][Ik] for k=1:dim)`.

The actual generated code efficiently calculates these contributions. For example,
for `dim = 3`, it generates:
```julia
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
```
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

"""
`find_linear_indices(::Val{N}, ::Val{dim}) -> Vector{Vector{Int64}}`

Internal function to compute all necessary linear index contribution vectors
for a `SymmetricTensor` of size `N` and dimension `dim`.

This function iteratively calls `find_linear_index_array(N, Val(k))` for `k` from 1 to `dim`.
The resulting collection of vectors is stored in the `linear_indices` field of a
`SymmetricTensor` and is crucial for its indexing operations.

# Arguments
- `::Val{N}`: A `Val` instance representing the size of each dimension.
- `::Val{dim}`: A `Val` instance representing the number of dimensions.

# Returns
- `Vector{Vector{Int64}}`: A vector where each inner vector is the result of
  `find_linear_index_array(N, Val(k))` for `k` in `1:dim`.
"""
function find_linear_indices(::Val{N}, ::Val{dim}) where {N, dim}
    contributions = Array{Int64, 1}[] # TODO: Consider Vector{Vector{Int64}}
    i = 1
    while i <= dim
        push!(contributions, find_linear_index_array(N, Val(i)))
        i += 1
    end
    return contributions
end


"""
`find_N_repetitions_sorted!(reps::Vector{<:Integer}, tup::NTuple)`

Internal helper function to count repetitions of elements in a **sorted** tuple.
It updates the `reps` vector such that `reps[i]` stores the count of distinct
elements that appear exactly `i` times in the tuple `tup`.

This function is used by `find_degeneracy` to calculate the multiplicity factor for tensor elements.

# Arguments
- `reps::Vector{<:Integer}`: A vector to store the counts. It will be modified in-place.
  Its length should be at least `length(tup)`.
- `tup::NTuple`: A tuple of elements, which **must be sorted** in non-decreasing order.

# Examples
```julia
julia> reps = zeros(Int, 8);
julia> tup = (1, 3, 3, 5, 5, 5, 5, 7); # Must be sorted
julia> PermutationSymmetricTensors.find_N_repetitions_sorted!(reps, tup);
julia> reps # Element 1 appears once, 7 once (reps[1]=2). Element 3 appears twice (reps[2]=1). Element 5 appears four times (reps[4]=1).
    8-element Vector{Int64}:
    2 # Two elements (1 and 7) appear once
    1 # One element (3) appears twice
    0 # Zero elements appear three times
    1 # One element (5) appears four times
    0 # Zero elements appear five times
    0 # Zero elements appear six times
    0 # Zero elements appear seven times
    0 # Zero elements appear eight times
```
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
 ```
function find_degeneracy(N::Int, dim::Int)
function find_degeneracy(A::SymmetricTensor{T, N, dim}) where {T, N, dim}
function find_degeneracy(N, dim, full_indices::Vector{<:NTuple{dim, Any}})
```

Returns a `SymmetricTensor{Int64, N, dim}` where each element `d[i,j,...]`
contains the number of distinct permutations of the indices `(i,j,...)` that map
to the same unique element in the `SymmetricTensor`. This value represents
the "degeneracy" of that particular combination of indices.

# Arguments
- `N::Int`: The size of each dimension of the tensor.
- `dim::Int`: The number of dimensions of the tensor.
- `A::SymmetricTensor`: An existing `SymmetricTensor` instance from which `N` and `dim` can be derived.
- `full_indices::Vector{<:NTuple{dim, Any}}`: (Optional) The output of `find_full_indices(N, dim)`,
  provided for efficiency if already computed.

# Returns
- `SymmetricTensor{Int64, N, dim}`: A tensor where each element stores its degeneracy.

# Examples
```julia
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

julia> a = rand(SymmetricTensor{Float64, 2, 4});
julia> d = find_degeneracy(a);
julia> d[1,1,1,1] # Element (1,1,1,1) is unique
1
julia> d[1,1,1,2] # Element (1,1,1,2) has 4 permutations (1112, 1121, 1211, 2111)
4
```
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