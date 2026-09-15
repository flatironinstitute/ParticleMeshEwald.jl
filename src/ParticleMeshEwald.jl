module ParticleMeshEwald

using SpecialFunctions, LoopVectorization, LinearAlgebra
using CellListMap
using FINUFFT
using KernelAbstractions
include(joinpath(dirname(pathof(KernelAbstractions)), "../examples/utils.jl"))

export PME
# energy/energy_short/energy_long are deliberately NOT exported: callers write
# ParticleMeshEwald.energy(...). See README.

# include("horner.jl")

include("types.jl")
include("energy.jl")

end
