using Test, PermutationSymmetricTensors # This load both the test suite and our MyAwesomePackage

a = rand(SymmetricTensor{Float64, 10, 2})

@test length(a.data) ==  symmetric_tensor_size(10, 2)
