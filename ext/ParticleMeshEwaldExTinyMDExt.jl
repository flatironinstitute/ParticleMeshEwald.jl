module ParticleMeshEwaldExTinyMDExt

# ExTinyMD adapter: supplies ExTinyMD.energy for PME, nothing else.
#
# ## Why PME is not placed in sys.interactions
#
# `MDSys`'s constructor requires `interactions::Vector{T_INTERACTION}` with
# `T_INTERACTION <: Tuple{ExTinyMD.AbstractInteraction, ExTinyMD.AbstractNeighborFinder}`
# (see ExTinyMD's src/types.jl). Satisfying that means `PME <: ExTinyMD.AbstractInteraction`,
# and a struct's supertype is fixed where the struct is defined -- Julia has no
# mechanism for an extension, loaded later and conditionally, to retroactively
# add a supertype to an already-compiled type. `PME` is defined in this
# package's src/, which does not depend on ExTinyMD at all (ExTinyMD is only a
# weak dependency), so it is never a subtype of `ExTinyMD.AbstractInteraction`
# -- not "unless you remember an annotation", but structurally, in every build
# of this package. Phase 1's ExTinyMD-internal interaction types did not face
# this because they are defined inside ExTinyMD itself, where the abstract
# type already exists at struct-definition time.
#
# Declaring a second, local `AbstractInteraction`-alike in this package's src/
# would not help either: `MDSys` checks against *ExTinyMD's* abstract type
# specifically, not against a structurally similar one from somewhere else.
#
# What multiple dispatch DOES allow, with no such restriction, is adding a
# new *method* to `ExTinyMD.energy` for the concrete type `PME` -- dispatch is
# on the argument's actual type, not on it subtyping anything in particular.
# That is what this file does. The method below is therefore only callable
# directly (`ExTinyMD.energy(pme, finder, sys, info)`), not through
# `sys.interactions`/`simulate!`, which is consistent with this package having
# no forces and so no way to drive an MD run regardless (see below).

using ParticleMeshEwald
using ExTinyMD

"Gather charges in slot order, honouring the id/slot indirection (sys.atoms is indexed by id, info.particle_info by slot)."
function _gather_charges(sys::ExTinyMD.MDSys{T}, info::ExTinyMD.SimulationInfo{T}) where T
    charges = Vector{T}(undef, length(info.particle_info))
    @inbounds for i in eachindex(info.particle_info)
        charges[i] = sys.atoms[info.particle_info[i].id].charge
    end
    return charges
end

"Gather positions in slot order as AoS (NTuple{3,T}, no conversion layer needed by ParticleMeshEwald's core)."
function _gather_positions(info::ExTinyMD.SimulationInfo{T}) where T
    poses = Vector{NTuple{3,T}}(undef, length(info.particle_info))
    @inbounds for i in eachindex(info.particle_info)
        p = info.particle_info[i].position
        poses[i] = (p[1], p[2], p[3])
    end
    return poses
end

# A NoNeighborFinder carries no usable list (ParticleMeshEwald's energy_short
# rebuilds its own cell list in that case); every other finder exposes one.
_finder_list(::ExTinyMD.NoNeighborFinder) = nothing
_finder_list(f) = f.neighbor_list

"""
    ExTinyMD.energy(pme::PME, neighborfinder, sys::MDSys, info::SimulationInfo)

Adapter between ExTinyMD's MD types and [`ParticleMeshEwald.energy`](@ref).
Gathers positions and charges from `info`/`sys` in slot order (honouring
ExTinyMD's id/slot indirection: `sys.atoms` is indexed by particle id,
`info.particle_info` by storage slot), refreshes `neighborfinder` if it is not
current, and calls into the framework-free core.

`PME` cannot be placed in `sys.interactions` -- see the comment at the top of
this file for why -- so call this method directly rather than through
`simulate!`.

This package has never computed forces, and `ExTinyMD.update_acceleration!` is
therefore **not** provided by this extension: there is no way to drive an MD
run with a `PME` interaction. Use ExTinyMD's own `PME3D` (which provides the
same reciprocal-space method with forces) for that.
"""
function ExTinyMD.energy(pme::ParticleMeshEwald.PME{T}, neighborfinder,
                         sys::ExTinyMD.MDSys{T}, info::ExTinyMD.SimulationInfo{T}) where T
    ExTinyMD.update_finder!(neighborfinder, info)
    poses = _gather_positions(info)
    charges = _gather_charges(sys, info)
    return ParticleMeshEwald.energy(pme, poses, charges; neighbor_list = _finder_list(neighborfinder))
end

end
