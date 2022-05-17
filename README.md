# PermutationSymmetricTensors.jl

[![Build status (Github Actions)](https://github.com/IlianPihlajamaa/PermutationSymmetricTensors.jl/workflows/CI/badge.svg)](https://github.com/IlianPihlajamaa/PermutationSymmetricTensors.jl/actions)
[![codecov.io](http://codecov.io/github/IlianPihlajamaa/PermutationSymmetricTensors.jl/coverage.svg?branch=main)](http://codecov.io/github/IlianPihlajamaa/PermutationSymmetricTensors.jl?branch=main)
[![PermutationSymmetricTensors Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/PermutationSymmetricTensors)](https://pkgs.genieframework.com?packages=PermutationSymmetricTensors).



PermutationSymmetricTensors provides an efficient framework for the use of multidimensional arrays that are symmetric under any permutation of their indices, implemented in pure Julia. Such symmetric tensors are implemented in the `SymmetricTensor{T, N, dim}` type, where `T` is the element type, `dim` is the number of indices required to index the tensor, and `N` is maximal index for each dimension. For example, to index a `SymmetricTensor{ComplexF64, 20, 6}`, you need 6 indices between 1 and 20. Note that we use the computer science definition of a tensor instead of the mathematical one: in the following, a tensor is just a multi-dimensional container of elements of some type `T`. As described above, we refer to the number of indices as the dimension of this tensor, because that is semantically consistent with the definition of a multidimensional array. In mathematics and physics texts, what we call dimension is usually referred to as the order, degree or rank of a tensor. 

This package exports basic constructors of `SymmetricTensor`s, and a few convenience functions for working with them. The main advantage of using a `SymmetricTensor` is that it requires much less memory to store than the full array would. 

## Construction

A `SymmetricTensor` can conveniently be constructed using `zeros`, `ones`, `similar` and `rand`. 

```julia
julia> using PermutationSymmetricTensors

julia> a = rand(SymmetricTensor{Float64, 2, 3})
2×2×2 SymmetricTensor{Float64, 2, 3}:
[:, :, 1] =
 0.117155  0.815916
 0.815916  0.978778

[:, :, 2] =
 0.815916  0.978778
 0.978778  0.825148

julia> b = zeros(SymmetricTensor{ComplexF32, 3, 3})
3×3×3 SymmetricTensor{ComplexF32, 3, 3}:
[:, :, 1] =
 0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im

[:, :, 2] =
 0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im

[:, :, 3] =
 0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im

julia>  c = ones(SymmetricTensor{Bool, 2, 2})
2×2 SymmetricTensor{Bool, 2, 2}:
 1  1
 1  1
 
julia> d = similar(c)
2×2 SymmetricTensor{Bool, 2, 2}:
 0  0
 0  0
```
Since the tensor is parametrized with its size, it is not necessary to provide any other arguments to `zeros`, `ones`, or `rand`. If the standard library `Random` is imported, `rand!(a)` will also work.

In order to create a `SymmetricTensor` from data stored in a `Vector{T}` directly, a constructor `SymmetricTensor(data, Val(N), Val(dim))` can be called. It is important to make sure that the length of the given vector `data` is correct. The function `find_symmetric_tensor_size(N, dim)` is useful for that purpose. Given the number of elements in each dimension `N` and the number of dimensions `dim`, it returns the number of distinct elements that a `SymmetricTensor{T, N, dim}` needs to store.

```julia
julia> L = find_symmetric_tensor_size(3, 3)
10

julia> data = collect(1:L)
10-element Vector{Int64}:
  1
  2
  3
  4
  5
  6
  7
  8
  9
 10

julia> SymmetricTensor(data, Val(3), Val(3))
3×3×3 SymmetricTensor{Int64, 3, 3}:
[:, :, 1] =
 1  2  3
 2  4  5
 3  5  6

[:, :, 2] =
 2  4  5
 4  7  8
 5  8  9

[:, :, 3] =
 3  5   6
 5  8   9
 6  9  10

```

Note that `SymmetricTensor`s leverage symmetry to minimize memory usage. It easy to create `SymmetricTensors` that would have more elements than `typemax(Int64)`, if they had been stored naively.

```julia
julia> d = rand(SymmetricTensor{Float64, 14, 20});

julia> println("This tensor requires ", round(sizeof(d)/2^30, digits=2), "GB memory")
This tensor requires 4.27GB memory

julia> println("a full array of this shape would require ", 14^20/2^30, "GB memory.")
a full array of this shape would require -5.75e9GB memory.

julia> println("a full array of this shape would require ", big(14)^20/2^30, "GB memory.")
a full array of this shape would require 7.79e+13GB memory.
```
In the second line, the computation `14^20/2^30` overflowed, and therefore returned the wrong result. This is important to take into account when calculating the sum of all elements of a `SymmetricTensor{Int, N, dim}`, if it is very large.

## Standard use

The tensors can be indexed and mutated at will.

```julia
julia> a = rand(SymmetricTensor{Int8, 3, 3})
3×3×3 SymmetricTensor{Int8, 3, 3}:
[:, :, 1] =
 -115  -31  117
  -31  110   95
  117   95  -57

[:, :, 2] =
 -31  110    95
 110  -30    33
  95   33  -106

[:, :, 3] =
 117    95   -57
  95    33  -106
 -57  -106    87

julia> a[1,2,3]
95

julia> a[3,1,2]
95

julia> a[3,2,2] = 6
6

julia> a
3×3×3 SymmetricTensor{Int8, 3, 3}:
[:, :, 1] =
 -115  -31  117
  -31  110   95
  117   95  -57

[:, :, 2] =
 -31  110    95
 110  -30     6
  95    6  -106

[:, :, 3] =
 117    95   -57
  95     6  -106
 -57  -106    87

julia> a[:, 1, 1] .= 0
3-element view(::SymmetricTensor{Int8, 3, 3}, :, 1, 1) with eltype Int8:
 0
 0
 0

julia> a
3×3×3 SymmetricTensor{Int8, 3, 3}:
[:, :, 1] =
 0    0    0
 0  110   95
 0   95  -57

[:, :, 2] =
   0  110    95
 110  -30     6
  95    6  -106

[:, :, 3] =
   0    95   -57
  95     6  -106
 -57  -106    87
```

They can also be iterated over and support most operations that arbitrary `AbstractArray`s do.

```julia
julia> a = rand(SymmetricTensor{BigFloat, 2, 4});

julia> for (i,ai) in enumerate(a)
           println((; i, ai))
       end
(i = 1, ai = 0.2504089172066270587643679267492732606636747737542205201257084506876941174077477)
(i = 2, ai = 0.1860567927479512764579583273597475579165054085072190505299583862788352760987293)
(i = 3, ai = 0.1860567927479512764579583273597475579165054085072190505299583862788352760987293)
(i = 4, ai = 0.4835838555693179868982759692607130469187769066045701314885210270407212019687799)
(i = 5, ai = 0.1860567927479512764579583273597475579165054085072190505299583862788352760987293)
(i = 6, ai = 0.4835838555693179868982759692607130469187769066045701314885210270407212019687799)
(i = 7, ai = 0.4835838555693179868982759692607130469187769066045701314885210270407212019687799)
(i = 8, ai = 0.3477955309561294416780053339785331891151633045252627608308216556228621260564572)
(i = 9, ai = 0.1860567927479512764579583273597475579165054085072190505299583862788352760987293)
(i = 10, ai = 0.4835838555693179868982759692607130469187769066045701314885210270407212019687799)
(i = 11, ai = 0.4835838555693179868982759692607130469187769066045701314885210270407212019687799)
(i = 12, ai = 0.3477955309561294416780053339785331891151633045252627608308216556228621260564572)
(i = 13, ai = 0.4835838555693179868982759692607130469187769066045701314885210270407212019687799)
(i = 14, ai = 0.3477955309561294416780053339785331891151633045252627608308216556228621260564572)
(i = 15, ai = 0.3477955309561294416780053339785331891151633045252627608308216556228621260564572)
(i = 16, ai = 0.6260418198971816214133682349248890223510908167601655542512797867183309913317024)

julia> size(a)
(2, 2, 2, 2)

julia> ndims(a)
4

julia> axes(a)
(Base.OneTo(2), Base.OneTo(2), Base.OneTo(2), Base.OneTo(2))

julia> length(a)
16

julia> sum(a)
5.913363165336039474111246622591563552654101882271734108751234567257141929172806

julia> prod(a)
3.515296863282957514709371560171304793858187682416751663789389978162540592035332e-08

julia> extrema(a)
(0.1860567927479512764579583273597475579165054085072190505299583862788352760987293, 0.6260418198971816214133682349248890223510908167601655542512797867183309913317024)

julia> lastindex(a)
16

julia> [i for i in eachindex(a)]
2×2×2×2 Array{CartesianIndex{4}, 4}:
[:, :, 1, 1] =
 CartesianIndex(1, 1, 1, 1)  CartesianIndex(1, 2, 1, 1)
 CartesianIndex(2, 1, 1, 1)  CartesianIndex(2, 2, 1, 1)

[:, :, 2, 1] =
 CartesianIndex(1, 1, 2, 1)  CartesianIndex(1, 2, 2, 1)
 CartesianIndex(2, 1, 2, 1)  CartesianIndex(2, 2, 2, 1)

[:, :, 1, 2] =
 CartesianIndex(1, 1, 1, 2)  CartesianIndex(1, 2, 1, 2)
 CartesianIndex(2, 1, 1, 2)  CartesianIndex(2, 2, 1, 2)

[:, :, 2, 2] =
 CartesianIndex(1, 1, 2, 2)  CartesianIndex(1, 2, 2, 2)
 CartesianIndex(2, 1, 2, 2)  CartesianIndex(2, 2, 2, 2)
```

Currently, broadcasting will always convert a `SymmetricTensor` into a full `N`-dimensional `Array`. For simple broadcasts, such as applying elementwise functions, instead consider using broadcasting on the `data`-field, which holds all data that the symmetric tensor contains. 

```julia
julia> @time a = rand(SymmetricTensor{Float64, 10, 8});
  0.000170 seconds (13 allocations: 191.266 KiB)

julia> @time b = a .* 0;
  2.755832 seconds (4 allocations: 762.940 MiB, 1.64% gc time)

julia> typeof(b)
Array{Float64, 8}

julia> @time c = similar(a);
  0.000125 seconds (13 allocations: 191.266 KiB)
  
julia> @time c.data .= a.data .* 0 .+ 1;
  0.000023 seconds (4 allocations: 128 bytes)
  
julia> sum(c) == length(c)
true
```

Be aware that some inplace operations can give unexpected results:

```julia
julia> a = rand(SymmetricTensor{Float64, 2, 2})
2×2 SymmetricTensor{Float64, 2, 2}:
 0.520987  0.84325
 0.84325   0.693854
 
julia> sort!(a, dims=2)
2×2 SymmetricTensor{Float64, 2, 2}:
 0.520987  0.693854
 0.693854  0.84325
```

## Convenience Functions

`find_full_indices(N, dim)` or `find_full_indices(a::SymmetricTensor)` returns an ordered array of tuples of indices (i1, i2, i3, ..., i{dim}) such that i1 >= i2 >= i3 ... >= i{dim}. This can be used to find the cartesian index that
corresponds to a linear index of a SymmetricTensor{T, N, dim}.
  Example: 
```julia
julia> find_full_indices(3, 3)
  10-element Vector{Tuple{Int32, Int32, Int32}}:
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
  
julia> a = rand(SymmetricTensor{Float64, 2, 8});

julia> find_full_indices(a)
9-element Vector{NTuple{8, Int32}}:
 (1, 1, 1, 1, 1, 1, 1, 1)
 (2, 1, 1, 1, 1, 1, 1, 1)
 (2, 2, 1, 1, 1, 1, 1, 1)
 (2, 2, 2, 1, 1, 1, 1, 1)
 (2, 2, 2, 2, 1, 1, 1, 1)
 (2, 2, 2, 2, 2, 1, 1, 1)
 (2, 2, 2, 2, 2, 2, 1, 1)
 (2, 2, 2, 2, 2, 2, 2, 1)
 (2, 2, 2, 2, 2, 2, 2, 2)
```

`find_degeneracy(N, dim)` or `find_degeneracy(a::SymmetricTensor)` returns a SymmetricTensor{Int64, N, dim} of which each element specifies the number of index permutations that point to the same element.
Examples:
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
```

The latter function is useful for efficient implementations of contractions of tensors. Consider for example the total contraction (sum) over all of the indices of an 8-dimensional `SymmetricTensor`.

```julia
julia> a = ones(SymmetricTensor{Float64, 8, 8});

julia> degeneracy = find_degeneracy(a);

julia> @time sum(a)
  0.482570 seconds (1 allocation: 16 bytes)
1.6777216e7

julia> @time sum(a.data .* degeneracy.data) # would be even more efficient with LinearAlgebra.dot...
  0.000039 seconds (7 allocations: 50.500 KiB)
1.6777216e7

julia> 8^8
16777216
```

## Implementation

A `SymmetricTensor{T, N, dim}` `a` contains two fields. 
 - `a.data` is a `Vector{T}` that stores all the elements of the symmetric tensor. Its length is given by `L = binomial(N-1+dim, dim)`, or more conveniently `L = find_symmetric_tensor_size(N, dim)`. 
 - `a.linear_indices` is a `Vector{Vector{Int64}}` that is needed when `a` is indexed. The outer vector has length `length(a.linear_indices)` equal to `dim`. The length elements of that vector are equal to `N`. To index a `SymmetricTensor{Float64, 50, 3}` at indices `I = (21, 45, 21)`, first the indices are sorted in descending order, which is stored in a new tuple `I2`. Then the linear index is found by evaluating `index = (A.linear_indices[1])[45] + (A.linear_indices[2])[21] + (A.linear_indices[3])[21]`. This linear index can then be used to get the value: `val = a.data[index]`.
 
Methods such as `getindex` and `find_full_indices` for operating with `SymmetricTensors` are implemented using generated functions.

## See also

There are two packages with comparable functionality, [SymmetricTensors.jl](https://github.com/iitis/SymmetricTensors.jl) and [Tensors.jl](https://github.com/Ferrite-FEM/Tensors.jl). 

Tensors.jl provides immutable, stack-allocated 1-, 2-, and 4-dimensional symmetric tensors. This package is preferable if the tensors are small, that is, when they have fewer than roughly 100 elements. It is also more full-featured, implementing many different operations on the tensors instead of just the basic functionality.

SymmetricTensors.jl provides a SymmetricTensor type just like the one exported in this package. Its implementation is based on a blocked memory pattern, sacrificing performance for cache locality. Some benchmarks:

```julia
import SymmetricTensors
using BenchmarkTools
```

Tensor creation:

```julia
julia> T = Float64;
julia> N = 30;
julia> dim = 5;

julia> a  = @btime rand(SymmetricTensors.SymmetricTensor{$T, $dim}, $N);
  2.601 s (21697571 allocations: 1.00 GiB)

julia> sizeof(a.frame)
6075000

julia> b  = @btime rand(PermutationSymmetricTensors.SymmetricTensor{$T, $N, $dim});
  642.400 μs (10 allocations: 2.12 MiB)

julia> sizeof(b)
2227288
```

`getindex` and `setindex!`:

```julia
julia> @btime $a[1,5,2,5,21];
  2.189 μs (24 allocations: 1.38 KiB)

julia> @btime $a[1,5,2,5,21] = 6.0;
  50.800 μs (767 allocations: 51.06 KiB)

julia> @btime $b[1,5,2,5,21];
  4.200 ns (0 allocations: 0 bytes)

julia> @btime $b[1,5,2,5,21] = 6.0;
  4.500 ns (0 allocations: 0 bytes)
```

