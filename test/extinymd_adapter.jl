@testset "ExTinyMD adapter" begin
    n, L = 60, 20.0
    boundary = Boundary((L, L, L), (1, 1, 1))
    atoms = Atom{Float64}[]
    for _ in 1:(n ÷ 2);      push!(atoms, Atom(type = 1, mass = 1.0, charge =  1.0)); end
    for _ in (n ÷ 2 + 1):n;  push!(atoms, Atom(type = 2, mass = 1.0, charge = -1.0)); end
    info = SimulationInfo(n, atoms, (0.0, L, 0.0, L, 0.0, L), boundary;
                          min_r = 1.0, temp = 1.0)
    info.running_step = 1

    # In stock ExTinyMD `particle_info[i].id == i`, so a gather that confuses slot
    # with id (e.g. `sys.atoms[i]` instead of `sys.atoms[info.particle_info[i].id]`)
    # looks correct forever. Permute the mapping so the two differ, following the
    # same pattern ExTinyMD's own adapter test uses ("adapter is correct when slot
    # order differs from id order" in ../ExTinyMD.jl/test/electrostatics/test_adapter.jl):
    # give every id a distinct charge (so a mix-up cannot cancel out), reverse the
    # slot order, and keep ids attached to their particles via info.id_dict.
    atoms = [Atom(type = a.type, mass = 1.0, charge = (isodd(i) ? 1.0 : -1.0) * (1 + 0.01 * i))
             for (i, a) in enumerate(atoms)]
    reverse!(info.particle_info)
    for i in eachindex(info.particle_info)
        info.id_dict[info.particle_info[i].id] = i
    end
    @test info.particle_info[1].id != 1     # the mapping really is permuted

    pme = PME(0.5, (L, L, L), 4.0, n)        # r_c = s/α = 8.0 < min(L)/2 = 10.0
    finder = CellList3D(info, pme.r_c, boundary, 1)

    # PME cannot subtype ExTinyMD.AbstractInteraction (see
    # ext/ParticleMeshEwaldExTinyMDExt.jl for why), so it cannot be placed in
    # sys.interactions/driven through simulate!. `sys` below exists only so the
    # adapter has something to gather `sys.atoms` charges from -- its
    # `interactions` is empty (typed as ExTinyMD's own no-op placeholders) --
    # and ExTinyMD.energy is therefore called on pme/finder directly rather
    # than through sys.interactions.
    sys = MDSys(n_atoms = n, atoms = atoms, boundary = boundary,
                interactions = Tuple{NoInteraction, NoNeighborFinder{Float64}}[],
                loggers = [TemperatureLogger(100; output = false)],
                simulator = VerletProcess(dt = 1e-4))

    poses   = [SVector(p.position[1], p.position[2], p.position[3]) for p in info.particle_info]
    charges = [atoms[p.id].charge for p in info.particle_info]

    @test ExTinyMD.energy(pme, finder, sys, info) ≈
          ParticleMeshEwald.energy(pme, poses, charges)
end
