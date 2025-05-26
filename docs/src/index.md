# PermutationSymmetricTensors.jl

@meta
CurrentModule = PermutationSymmetricTensors


## Example Usage

This section provides practical examples of how to use `PermutationSymmetricTensors.jl`.

First, let's bring the module into scope:
```julia
using PermutationSymmetricTensors
using Random # For seeding, if desired
```

### Creating a `SymmetricTensor`

There are several ways to construct a `SymmetricTensor`.

**1. Using the low-level constructor with explicit data:**
```julia
# Define dimensions and size
N = 3 # Size of each dimension (e.g., indices from 1 to 3)
dim = 2 # Number of dimensions

# Calculate the required number of unique elements
len_data = find_symmetric_tensor_size(N, dim) # For N=3, dim=2, this is 6

# Create some data (e.g., random)
Random.seed!(123) # for reproducibility
data_vector = rand(Float64, len_data)

# Construct the tensor
tensor_a = SymmetricTensor(data_vector, Val(N), Val(dim))
println("Tensor A (from data_vector):")
display(tensor_a)
println("\\n")
```

**2. Using `rand` for random initialization:**
```julia
# Create a 2x2x2 tensor with Float64 elements, random values in [0,1)
tensor_b = rand(SymmetricTensor{Float64, 2, 3})
println("Tensor B (randomly initialized):")
display(tensor_b)
println("\\n")
```

**3. Using `zeros` for zero initialization:**
```julia
# Create a 2x2 tensor with Int elements, initialized to zero
tensor_c = zeros(SymmetricTensor{Int, 2, 2})
println("Tensor C (zero-initialized):")
display(tensor_c)
println("\\n")
```

### Getting and Setting Elements

Elements are accessed using standard indexing. Due to symmetry, the order of indices does not matter.

```julia
# Using tensor_b from above (2x2x2 Float64 tensor)
println("Original tensor_b[1,2,1]: ", tensor_b[1,2,1])

# Set an element
tensor_b[1,2,1] = 0.5
println("After tensor_b[1,2,1] = 0.5:")
println("tensor_b[1,2,1]: ", tensor_b[1,2,1])
println("tensor_b[2,1,1] (should be same): ", tensor_b[2,1,1]) # Permuted index
println("tensor_b[1,1,2] (should be same): ", tensor_b[1,1,2]) # Permuted index
println("\\n")

# Modifying tensor_c
tensor_c[1,2] = 5
println("Tensor C after tensor_c[1,2] = 5:")
display(tensor_c)
println("tensor_c[2,1] (should be same): ", tensor_c[2,1])
println("\\n")
```

### Using Utility Functions

**1. `find_symmetric_tensor_size`:**
Calculate the number of unique elements required for a symmetric tensor.
```julia
N_val = 4
dim_val = 3
num_elements = find_symmetric_tensor_size(N_val, dim_val)
println("Number of unique elements for a $N_val^$dim_val symmetric tensor: ", num_elements) # Binomial(4+3-1, 3) = 20
println("\\n")
```

**2. `find_degeneracy`:**
Get a tensor where each element shows how many permutations of its indices map to it.
```julia
# Using tensor_c (2x2 Int tensor)
degeneracy_c = find_degeneracy(tensor_c)
println("Degeneracy tensor for Tensor C (2x2):")
display(degeneracy_c)
# For a 2x2 tensor:
# d[1,1] = 1 (only 1,1)
# d[1,2] = 2 (1,2 and 2,1)
# d[2,1] is same as d[1,2]
# d[2,2] = 1 (only 2,2)
println("\\n")

degeneracy_b = find_degeneracy(SymmetricTensor{Int, 2, 3}) # For a generic 2x2x2 shape
println("Degeneracy for a 2x2x2 tensor shape:")
display(degeneracy_b)
# For a 2x2x2 tensor:
# d[1,1,1] = 1
# d[1,1,2] = 3 (112, 121, 211)
# d[1,2,2] = 3 (122, 212, 221)
# d[2,2,2] = 1
println("\\n")

```

**3. `find_full_indices`:**
Get the list of unique Cartesian indices corresponding to the linear storage.
The indices are sorted such that `i1 >= i2 >= ... >= idim`.
```julia
# For a 3x3 tensor (N=3, dim=2)
full_idx_3_2 = find_full_indices(3, 2)
println("Full indices for N=3, dim=2 (sorted i1>=i2):")
for (i, idx) in enumerate(full_idx_3_2)
    println("Linear index $i maps to Cartesian index $idx")
end
# Expected order for N=3, dim=2: (1,1), (2,1), (2,2), (3,1), (3,2), (3,3)
# Let's re-verify against the code's logic:
# The @generated function for find_full_indices (T, N, ::Val{dim}) has nested loops:
# for i_dim = start_outer:N ... for i_1 = start_inner:N.
# For dim=2, (i1, i2): for i2 = 1:N; for i1 = i2:N; push!((i1,i2)).
# So for N=3, dim=2:
# i2=1: i1=1 -> (1,1)
#       i1=2 -> (2,1)
#       i1=3 -> (3,1)
# i2=2: i1=2 -> (2,2)
#       i1=3 -> (3,2)
# i2=3: i1=3 -> (3,3)
# Output: (1,1), (2,1), (3,1), (2,2), (3,2), (3,3)
println("\\n")

# Example: Relating to tensor_a (N=3, dim=2)
println("tensor_a was created with N=3, dim=2. Its data has length: ", length(tensor_a.data))
println("The first element tensor_a.data[1] corresponds to Cartesian index: ", full_idx_3_2[1]) # Should be (1,1)
println("The fourth element tensor_a.data[4] corresponds to Cartesian index: ", full_idx_3_2[4]) # Should be (2,2)
println("\\n")
```

## Public API

This section highlights the core functionalities and types provided by the `PermutationSymmetricTensors.jl` package, serving as a quick reference to its main features.


## Full API Reference

A complete list of all exported names from the `PermutationSymmetricTensors` module. This provides a comprehensive overview of all functionalities available to users.

```@autodocs
Modules = [PermutationSymmetricTensors]
Order   = [:module, :constant, :type, :macro, :function]
Public  = true
Private = false
```
