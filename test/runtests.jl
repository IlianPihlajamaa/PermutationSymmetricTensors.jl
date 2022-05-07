using Test, PermutationSymmetricTensors # This load both the test suite and our MyAwesomePackage

out = plusTwo(3)

@test out == 5               # This is the actual test condition. You can add as many tests as you wish.