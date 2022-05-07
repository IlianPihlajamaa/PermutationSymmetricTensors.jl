using Test, PermutationSymmetricTensors, TupleTools, Random 


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
                    b = similar(a)
                    @test axes(a) == axes(b)
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


@testset "getindex" begin
    for N in [2, 5, 10]
        for dim in [2, 5, 6]
            a = rand(SymmetricTensor{Float64, N, dim})
            for i = 1:100
                idx = sort(rand(1:N, dim))
                idx_shuffle = shuffle(idx)
                idxt = tuple(idx...)
                idx_shufflet = tuple(idx_shuffle...)
                @test a[idxt...] == a[idx_shufflet...]
            end
        end
    end
end

@testset "setindex!" begin
    for N in [2, 5, 10]
        for dim in [2, 5, 6]
            a = rand(SymmetricTensor{Float64, N, dim})
            for i = 1:100
                idx = sort(rand(1:N, dim))
                idx_shuffle = shuffle(idx)
                idxt = tuple(idx...)
                idx_shufflet = tuple(idx_shuffle...)
                val = rand(Float64)
                a[idx_shufflet...] = val
                @test a[idxt...] == val
            end
        end
    end
end