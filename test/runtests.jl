using ParticleMeshEwald
using FINUFFT
using LinearAlgebra
using Random
using StaticArrays
using Test

@testset "ParticleMeshEwald.jl" begin
    include("energy.jl")
end
