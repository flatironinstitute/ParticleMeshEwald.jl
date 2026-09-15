using ParticleMeshEwald
using FINUFFT
using LinearAlgebra
using Random
using StaticArrays
using Test

# `using ExTinyMD` here (in the test target, not module ParticleMeshEwald) is
# what triggers Julia to load ext/ParticleMeshEwaldExTinyMDExt.jl for the rest
# of this process -- exercising the extension is the whole point of Task 4.
using ExTinyMD

@testset "ParticleMeshEwald.jl" begin
    include("energy.jl")
    include("extinymd_adapter.jl")
end
