using ParticleMeshEwald
using FINUFFT
using EwaldSummations, ExTinyMD
using LinearAlgebra
using Random
using Test

@testset "ParticleMeshEwald.jl" begin
    include("energy.jl")
end
