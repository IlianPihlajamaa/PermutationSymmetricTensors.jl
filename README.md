# PermutationSymmetricTensors.jl

[![Build status (Github Actions)](https://github.com/IlianPihlajamaa/PermutationSymmetricTensors.jl/workflows/CI/badge.svg)](https://github.com/IlianPihlajamaa/PermutationSymmetricTensors.jl/actions)
[![codecov.io](http://codecov.io/github/IlianPihlajamaa/PermutationSymmetricTensors.jl/coverage.svg?branch=main)](http://codecov.io/github/IlianPihlajamaa/PermutationSymmetricTensors.jl?branch=main)

PermutationSymmetricTensors provides a framework for implementing multidimensional arrays that are symmetric under any permutation of their indices. Such symmetric tensors are implemented in the `SymmetricTensor{T, N, dim}` type, where `T` is the element type, `dim` is the number of indices required to index the tensor, and `N` is maximal index for each dimension. For example, to index a `SymmetricTensor{ComplexF64, 20, 6}`, you need 6 indices between 1 and 20. Note that here, we use the computer science definition of the tensor instead of the mathematical one: in the following, a tensor is just a multi-dimensional container of elements. As described above, we refer to the number of indices as the dimension of this tensor, because that is consistent with the definition of a multidimensional array. In mathematics and physics texts, this is usually refered to as the order or rank of a tensor. 

This package exports basic constructors of `SymmetricTensor`s, and a few convenience functions for working with them. The main advantage of using a `SymmetricTensor` is that it requires much less memory to store than the full array would. 

## Construction

A `SymmetricTensor` can conveniently be constructed using `zeros`, `ones`, and `rand`. Alternatively, it is possible to construct one directly if the underlying data is available.

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
```
Since the tensor is parametrized with its size, it is not necessary to provide any other arguments to `zeros`, `ones`, or `rand`.

In order to create a `SymmetricTensor` from a `Vector{T}`, make sure that the length of that vector is correct. The function `find_symmetric_tensor_size(N, dim)` is useful for that. Given the number of elements in each dimension `N` and the number of dimensions `dim`, it returns the number of distinct elements that a `SymmetricTensor{T, N, dim}` needs to store.
```julia
julia> N_elements = find_symmetric_tensor_size(3, 3)
10

julia> data = collect(1:N_elements)
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

Note that this means that these objects leverage symmetry to minimize memory usage. It easy to create `SymmetricTensors` that would have more elements than `typemax(Int64)`, if they had to be stored naively.

```julia
julia> d = rand(SymmetricTensor{Float64, 14, 20});

julia> println("This tensor requires ", round(sizeof(d)/2^30, digits=2), "GB memory")
This tensor requires 4.27GB memory

julia> println("a full array of this shape would require ", 14^20/2^30, "GB memory.")
a full array of this shape would require -5.75e9GB memory.

julia> println("a full array of this shape would require ", big(14)^20/2^30, "GB memory.")
a full array of this shape would require 7.79e+13GB memory.
```

## Standard use

The tensors can be indexed and mutated at will.

```
julia> a = rand(SymmetricTensor{Int8, 3, 3})
3×3×3 SymmetricTensor{Int8, 3, 3}:
[:, :, 1] =
 -93   67  -42
  67  -31  115
 -42  115   29

[:, :, 2] =
  67   -31   115
 -31    76  -110
 115  -110   -34

[:, :, 3] =
 -42   115   29
 115  -110  -34
  29   -34   64

julia> a[3,2,1] = 6
6

julia> a
3×3×3 SymmetricTensor{Int8, 3, 3}:
[:, :, 1] =
 -93   67  -42
  67  -31    6
 -42    6   29

[:, :, 2] =
  67   -31     6
 -31    76  -110
   6  -110   -34

[:, :, 3] =
 -42     6   29
   6  -110  -34
  29   -34   64

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

## Convenience Functions

`find_full_indices(N, dim)` returns an ordered array of tuples of indices (i1, i2, i3, ..., i{dim}) such that
i1 >= i2 >= i3 ... >= i{dim}. This can be used to find the cartesian index that
corresponds to a linear index of a SymmetricTensor{T, N, dim}. It will automatically choose an appropriate integer type that to minimize the amount of required storage.
```julia
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
```

`find_full_indices(N, dim)` returns a SymmetricTensor{Int64, N, dim} of which each element specifies the number of index permutations that point to the same element.
for efficiency can be called with the result of `find_full_indices(N, dim)` as a third argument. It can also be called on a `SymmetricTensor` as `find_full_indices(a::SymmetriTensor)`.

```julia
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

## See also

There are two packages with comparable functionality, [SymmetricTensors.jl](https://github.com/iitis/SymmetricTensors.jl) and [Tensors.jl](https://github.com/Ferrite-FEM/Tensors.jl). 

Tensors.jl provides immutable, stack-allocated 1-, 2-, and 4-dimensional symmetric tensors. This package is preferable if the tensors are small, that is, when they have less than roughly 100 elements. 

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

