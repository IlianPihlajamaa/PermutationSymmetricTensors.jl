using Test, PermutationSymmetricTensors, TupleTools, Random 

println("Testing PermutationSymmetricTensors")
@testset "Creation" begin
    for T in [Float64, ComplexF32, BigFloat, Bool]
        for N in [2, 5, 10]
            for dim in [2, 5, 10]
                for func in [rand, ones, zeros]
                    a = func(SymmetricTensor{T, N, dim})
                    @test length(a.data) ==  find_symmetric_tensor_size(N, dim)
                    @test length(a) == N^dim
                    @test sizeof(a) > sizeof(a.data)
                    @test ndims(a) == dim
                    @test axes(a,1) == Base.OneTo(N)
                    b = similar(a)
                    @test axes(a) == axes(b)
                    @test sum(Float64.(real.(b.data)) .* find_degeneracy(b).data) == 0
                    rand!(b, 1:1)
                    @test sum(Float64.(real.(b.data)) .* find_degeneracy(b).data) == length(b)
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
            d2 = find_degeneracy(N, dim)
            @test all(d.data .== d2.data)
            f_i = find_full_indices(a)
            d3 = find_degeneracy(N, dim, f_i)
            @test all(d.data .== d3.data)
            @test sum(d.data.*a.data) == length(a)
            a = rand(SymmetricTensor{Float64, N, dim})
            d = find_degeneracy(a)
            @test sum(a) ≈ sum(d.data.*a.data)
        end
    end
    a = rand(SymmetricTensor{Float64, 2,3})
    indxs =  find_full_indices(a)
    @test all(indxs .== find_full_indices(2,3))
    @test indxs[3] == (2,2,1)
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