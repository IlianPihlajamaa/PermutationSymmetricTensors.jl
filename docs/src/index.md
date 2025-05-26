```@meta
CurrentModule = PermutationSymmetricTensors
```

# PermutationSymmetricTensors.jl

`PermutationSymmetricTensors.jl` provides efficient tools for working with multidimensional arrays that are symmetric under any permutation of their indices.

This page provides practical examples, usage tips, and performance insights.

---

## Getting Started

```@example 1
using PermutationSymmetricTensors
using Random # For reproducibility if needed
```

---

## Creating Symmetric Tensors

### 1. Low-Level Constructor

```@example 1
N = 3       # Size of each axis
dim = 2     # Number of dimensions
len = find_symmetric_tensor_size(N, dim)  # e.g., 6 for N=3, dim=2
data = rand(Float64, len)

tensor_a = SymmetricTensor(data, Val(N), Val(dim))
```

### 2. Built-In Constructors

```@example 1
tensor_b = rand(SymmetricTensor{Float64, 3, 3})         # Random values
```
```@example 1
tensor_c = zeros(SymmetricTensor{Int, 4, 2})            # Zeros
```
```@example 1
tensor_d = ones(SymmetricTensor{Bool, 2, 4})            # Ones
```
```@example 1
tensor_e = similar(tensor_c)                            # Uninitialized with same type
```
```@example 1
tensor_f = similar(tensor_d, Char)                      # Uninitialized with new type
```

---

## Indexing and Symmetry

Indexing into a symmetric tensor is invariant under permutations of the indices:

```@example 1
A = rand(SymmetricTensor{Float64, 2, 3})

A[1, 2, 1] == A[2, 1, 1] == A[1, 1, 2]  # All access the same element

A[1, 2, 1] = 42.0

@assert A[2, 1, 1] == 42.0
```

You can also slice and broadcast:

```@example 1
A[:, 1, 1] .= 0
```

---

## Utility Functions

### `find_symmetric_tensor_size`

Returns the number of unique values stored in a symmetric tensor of size `N` and dimension `dim`.

```@example 1
find_symmetric_tensor_size(3, 3)  # Returns 10
```

Useful for constructing from raw data:

```@example 1
data = rand(Float64, find_symmetric_tensor_size(4, 3))
T = SymmetricTensor(data, Val(4), Val(3));
```

---

### `find_degeneracy`

Returns a tensor indicating how many permutations of the indices point to each element.

```@example 1
A = rand(SymmetricTensor{Float64, 2, 3})
D = find_degeneracy(A)

@show D[1, 1, 2] 
```

---

### `find_full_indices`

Gives you the sorted list of canonical index tuples that correspond to the linear storage layout.

```@example 1
inds = find_full_indices(3, 2)
for (i, idx) in enumerate(inds)
    println("Linear index $i maps to Cartesian index $idx")
end
```

---

## Performance Tips

### Memory Savings

```@example 1
A = rand(SymmetricTensor{Float64, 14, 16})

println("Compressed size: ", Base.format_bytes(Base.summarysize(A)))
println("Full array would require: ", round(Float64(big(14)^16 * 8)/2^30, digits=2), " GiB")
```

### Efficient Aggregations

Use the internal `.data` field with the degeneracy weights:

```@example 1
deg = find_degeneracy(A)
sum(A.data .* deg.data)  # Correct full sum over symmetric elements
```

### Broadcasting Performance

Avoid converting to full arrays unintentionally:

```@example 1
A.data .= log.(A.data .+ 1e-8)  # Efficient

#B = A .* 0  # WARNING: returns a full Array{Float64, N}. Will overflow RAM
```

---

## Example: Exploring Internal Representation

```@example 1
A = rand(SymmetricTensor{Float64, 3, 3})
deg = find_degeneracy(A)
inds = find_full_indices(A)

for i in eachindex(A.data)
    println("data[$i] = ", A.data[i], ", index: ", inds[i], ", deg: ", deg[inds[i]...])
end
```

---

## Summary of Public API

| Feature                          | Description                                               |
|----------------------------------|-----------------------------------------------------------|
| `SymmetricTensor{T, N, dim}`     | Core symmetric tensor type                               |
| `find_symmetric_tensor_size`     | Number of stored unique elements                         |
| `find_degeneracy`                | Permutation multiplicity tensor                          |
| `find_full_indices`              | List of canonical index tuples                           |
| `zeros`, `ones`, `rand`, `similar` | Tensor constructors                                      |
| `getindex`, `setindex!`          | Symmetric indexing and mutation                          |

---

## Full API Reference

For a complete overview of all exported functions and types:

```@autodocs
Modules = [PermutationSymmetricTensors]
Order   = [:module, :constant, :type, :macro, :function]
```