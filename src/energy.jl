# Array-of-structs query API.
#
# `poses` is any AbstractVector whose elements support `p[1]`, `p[2]`, `p[3]` --
# `SVector{3,T}`, `NTuple{3,T}` and ExTinyMD's `Point{3,T}` all qualify with no
# conversion layer. `charges` is a plain `AbstractVector{<:Real}`; the
# `Complex{T}` conversion FINUFFT needs internally is done into a plan-owned
# buffer (`pme.qs`), never exposed to the caller.
#
# Every query scatters `poses`/`charges` into `pme`'s own scratch (`pme.pos`,
# `pme.xs`/`ys`/`zs`, `pme.qs`) rather than mutating the caller's arrays or
# allocating new ones -- in particular this replaces the previous
# `energy_long` implementation, which scaled the caller's coordinate vectors
# in place and divided them back afterwards, corrupting caller data if the
# NUFFT call threw in between.

"Scatter `poses` (scaled by 2π/L) and `charges` (promoted to Complex{T}) into `pme`'s FINUFFT scratch."
function _scatter_long!(pme::PME{T}, poses, charges) where T
    kx = 2 * T(π) / pme.L[1]
    ky = 2 * T(π) / pme.L[2]
    kz = 2 * T(π) / pme.L[3]
    @inbounds @fastmath for i in 1:pme.N
        p = poses[i]
        pme.xs[i] = T(p[1]) * kx
        pme.ys[i] = T(p[2]) * ky
        pme.zs[i] = T(p[3]) * kz
        pme.qs[i] = Complex{T}(charges[i])
    end
    return nothing
end

"""
    energy_long(pme, poses, charges) -> T

Reciprocal-space (long-range) part of the PME energy, evaluated with FINUFFT.
`poses` is AoS; `charges` are plain reals. Neither is mutated.
"""
function energy_long(pme::PME{T}, poses, charges) where T
    @assert length(poses) == pme.N
    @assert length(charges) == pme.N

    _scatter_long!(pme, poses, charges)

    finufft_setpts!(pme.plan, pme.xs, pme.ys, pme.zs)
    finufft_exec!(pme.plan, pme.qs, pme.rho)

    t = zero(T)
    loops = (2 * pme.n_k[1] + 1) * (2 * pme.n_k[2] + 1) * (2 * pme.n_k[3] + 1)
    @inbounds @fastmath @simd for i in 1:loops
        t += (pme.rho[i].re ^ 2 + pme.rho[i].im ^ 2) * pme.D[i]
    end

    return t / (pme.L[1] * pme.L[2] * pme.L[3]) / 2
end

"Scatter `poses` into `pme.pos`, the Matrix{T} CellListMap's InPlaceNeighborList reads."
function _scatter_pos!(pme::PME{T}, poses) where T
    @inbounds @fastmath for i in 1:pme.N
        p = poses[i]
        pme.pos[1, i] = T(p[1])
        pme.pos[2, i] = T(p[2])
        pme.pos[3, i] = T(p[3])
    end
    return pme.pos
end

function energy_short_single(alpha::T, neighbor_list, charges) where T
    Es = zero(T)
    for k in 1:length(neighbor_list)
        i, j, r = neighbor_list[k]
        @inbounds @fastmath qi, qj = charges[i], charges[j]
        @inbounds @fastmath t = qi * qj * erfc(alpha * r) / r
        Es += t
    end
    return Es
end

# Task-partitioned reduction: the neighbour list is split into
# `Threads.nthreads()` contiguous chunks, and each task accumulates into the
# slot given by its OWN LOOP INDEX (`t`, 1:nthreads()), never by
# `Threads.threadid()`. A previous version keyed the per-task accumulator by
# `Threads.threadid() - 1`, which is unsound for two reasons: (1) it is not a
# safe key under Julia's task-migration scheduler regardless of its value, and
# (2) inside `@inbounds`, `threadid() == 1` (index 0) is not a BoundsError but
# a silent out-of-bounds write -- confirmed by hand to produce a wrong energy
# value (not a crash) when Julia's `:interactive` thread pool is sized 0, so
# thread 1 participates in `Threads.@threads` scheduling. Keying by the static
# loop-partition index `t` sidesteps both problems: each of the `nthreads()`
# tasks owns exactly one slot for the lifetime of the reduction, however many
# OS threads or task migrations actually execute it.
function energy_short_threaded(alpha::T, neighbor_list, charges) where T
    n = length(neighbor_list)
    nt = Threads.nthreads()
    chunk_energy = zeros(T, nt)
    Threads.@threads for t in 1:nt
        lo = ((t - 1) * n) ÷ nt + 1
        hi = (t * n) ÷ nt
        local_E = zero(T)
        @inbounds for k in lo:hi
            i, j, r = neighbor_list[k]
            @fastmath qi, qj = charges[i], charges[j]
            @fastmath local_E += qi * qj * erfc(alpha * r) / r
        end
        chunk_energy[t] = local_E
    end
    return sum(chunk_energy)
end

"""
    energy_short(pme, poses, charges; neighbor_list = nothing) -> T

Real-space (short-range) part of the PME energy. `poses` is AoS; `charges` are
plain reals. Neither is mutated.

Pass `neighbor_list` (a `CellListMap`-style list of `(i, j, r)` triples) to
reuse a list already maintained elsewhere and skip rebuilding `pme`'s own
cell list from `poses`.
"""
function energy_short(pme::PME{T}, poses, charges; neighbor_list = nothing) where T
    @assert length(poses) == pme.N
    @assert length(charges) == pme.N

    if neighbor_list === nothing
        _scatter_pos!(pme, poses)
        update!(pme.celllist, xpositions = pme.pos)
        nb = neighborlist!(pme.celllist)
    else
        nb = neighbor_list
    end

    Es = if Threads.nthreads() == 1 || length(nb) == 0
        energy_short_single(pme.alpha, nb, charges)
    else
        energy_short_threaded(pme.alpha, nb, charges)
    end

    t = pme.alpha / sqrt(T(π))
    @inbounds for i in 1:pme.N
        Es -= charges[i]^2 * t
    end

    return Es / (4 * T(π))
end

"""
    energy(pme, poses, charges; neighbor_list = nothing) -> T

Total PME energy (long + short range). See [`energy_long`](@ref) and
[`energy_short`](@ref).
"""
function energy(pme::PME{T}, poses, charges; neighbor_list = nothing) where T
    return energy_long(pme, poses, charges) +
           energy_short(pme, poses, charges; neighbor_list = neighbor_list)
end
