using Test, PermutationSymmetricTensors # This load both the test suite and our MyAwesomePackage


@testset "Creation" begin
    for T in [Float64, ComplexF32, BigFloat, Bool]
        for N in [2, 5, 10]
            for dim in [2, 5, 10]
                for func in [rand, ones, zeros]
                    a = func(SymmetricTensor{T, N, dim})
                    @test length(a.data) ==  symmetric_tensor_size(N, dim)
                    @test length(a) == N^dim
                    @test sizeof(a) > sizeof(a.data)
                    @test ndims(a) == dim
                    @test axes(a,1) == Base.OneTo(N)
                end
            end
        end
    end
end

@testset "degeneracy" begin
    for N in [2, 5, 10]
        for dim in [2, 5, 6]
            a = ones(SymmetricTensor{BigInt, N, dim})
            d = find_degeneracy(a)
            @test sum(d.data.*a.data) == length(a)

            a = rand(SymmetricTensor{Float64, N, dim})
            d = find_degeneracy(a)
            @test sum(a) ≈ sum(d.data.*a.data)
        end
    end
end

