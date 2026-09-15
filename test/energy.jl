@testset "fft" begin

    function rho_nufft(pme::PME, x::Vector{T}, y::Vector{T}, z::Vector{T}, q::Vector{Complex{T}}) where T

        rho_n = zeros(Complex{T}, 2 * pme.n_k[1] + 1, 2 * pme.n_k[2] + 1, 2 * pme.n_k[3] + 1)
    
        x .*= 2π / pme.L[1]
        y .*= 2π / pme.L[2]
        z .*= 2π / pme.L[3]
    
        finufft_setpts!(pme.plan, x, y, z)
        finufft_exec!(pme.plan, q, rho_n)
    
        x ./= 2π / pme.L[1]
        y ./= 2π / pme.L[2]
        z ./= 2π / pme.L[3]
    
        return rho_n
    end
    
    function rho_direct(pme::PME, x::Vector{T}, y::Vector{T}, z::Vector{T}, q::Vector{Complex{T}}) where T
    
        rho_n = zeros(Complex{T}, 2 * pme.n_k[1] + 1, 2 * pme.n_k[2] + 1, 2 * pme.n_k[3] + 1)
    
        kx0 = 2π / pme.L[1]
        ky0 = 2π / pme.L[2]
        kz0 = 2π / pme.L[3]
    
        for i in 1:2 * pme.n_k[1] + 1, j in 1:2 * pme.n_k[2] + 1, k in 1:2 * pme.n_k[3] + 1
            kx = (i - pme.n_k[1] - 1) * kx0
            ky = (j - pme.n_k[2] - 1) * ky0
            kz = (k - pme.n_k[3] - 1) * kz0
            for n in 1:pme.N
                rho_n[i, j, k] += q[n] * exp(im * (kx * x[n] + ky * y[n] + kz * z[n]))
            end
        end
    
        return rho_n
    end

    p = PME(1.0, (10.0, 10.0, 10.0), 4.9, 1000)
    x = rand(1000)
    y = rand(1000)
    z = rand(1000)
    q = rand(ComplexF64, 1000)
    rho_n = rho_nufft(p, x, y, z, q)
    rho_direct = rho_direct(p, x, y, z, q)
    @test norm(rho_n .- rho_direct) < 1e-7
end

@testset "AoS query API" begin
    Random.seed!(2026)
    n = 100
    L = (20.0, 20.0, 20.0)
    α, s = 0.5, 4.0            # r_c = s/α = 8.0 < min(L)/2 = 10.0
    poses = [SVector(rand() * L[1], rand() * L[2], rand() * L[3]) for _ in 1:n]
    charges = [isodd(i) ? 1.0 : -1.0 for i in 1:n]

    pme = PME(α, L, s, n)
    E = ParticleMeshEwald.energy(pme, poses, charges)
    @test isfinite(E)

    # the same positions as NTuple and as a 3-column read must agree
    @test ParticleMeshEwald.energy(pme, [Tuple(p) for p in poses], charges) ≈ E
end

@testset "energy does not mutate the caller's positions" begin
    Random.seed!(7)
    n = 50
    L = (20.0, 20.0, 20.0)
    poses = [SVector(rand() * L[1], rand() * L[2], rand() * L[3]) for _ in 1:n]
    charges = [isodd(i) ? 1.0 : -1.0 for i in 1:n]
    before = deepcopy(poses)

    pme = PME(0.5, L, 4.0, n)      # r_c = 8.0 < min(L)/2 = 10.0
    ParticleMeshEwald.energy(pme, poses, charges)
    @test poses == before
end

@testset "regression: baseline energies recorded pre-CellListMap-0.10 (Task 1)" begin
    # These reference values were captured by running this exact loop (fixed seed,
    # PME-only, no other package touching the global RNG) against the pre-Task-1
    # source, on CellListMap 0.9.17 -- i.e. before InPlaceNeighborList/update! were
    # updated to the 0.10 keyword spellings. They must be reproduced exactly (to
    # floating-point noise) after the bump, since the bump is a call-site fix, not a
    # numerical change.
    #
    # An earlier capture attempt reused the *old* "compare ewald" testset body, which
    # also built an ExTinyMD `SimulationInfo` in the same loop; that constructor
    # consumes `rand()` internally (for its own default random placement, immediately
    # overwritten here), which shifts the global RNG stream and silently changes every
    # x/y/z/q draw from the second loop iteration on. That produced seven wrong
    # "baseline" numbers that only coincidentally matched in the first iteration. This
    # version omits ExTinyMD/EwaldSummations entirely so the only randomness consumed
    # is the one this testset itself draws, and is what is actually reproducible.
    #
    # Separately, and for the reason explained above the old testset is retired here:
    # every published EwaldSummations/ExTinyMD version pins CellListMap = "0.9", which
    # conflicts with this package's own CellListMap = "0.10" compat at the Pkg-resolve
    # stage (not a runtime error) as soon as both are requested in the same (test)
    # environment -- the same class of conflict this whole phase exists to remove.
    #
    # alpha = 1.0, s = 4.9 => r_c = s/alpha = 4.9, strictly less than min(L)/2 = 5.0
    # for every L below (10.0 and 20.0 per axis).
    Random.seed!(42)
    n_atoms = 1000
    baseline_energy = Dict(
        (10.0, 10.0, 10.0) =>  0.49214543119605736,
        (10.0, 10.0, 20.0) => -0.8683258929918347,
        (10.0, 20.0, 10.0) => -0.6751434370589928,
        (10.0, 20.0, 20.0) => -0.6851927045492889,
        (20.0, 10.0, 10.0) =>  0.6714441133230378,
        (20.0, 10.0, 20.0) =>  0.11265984868790824,
        (20.0, 20.0, 10.0) => -0.2650868484349327,
        (20.0, 20.0, 20.0) => -0.7503004782366607,
    )

    for Lx in [10.0, 20.0], Ly in [10.0, 20.0], Lz in [10.0, 20.0]

        p = PME(1.0, (Lx, Ly, Lz), 4.9, n_atoms)
        x = rand(n_atoms) .* Lx
        y = rand(n_atoms) .* Ly
        z = rand(n_atoms) .* Lz

        q0 = rand(n_atoms);
        q0 .-= sum(q0) / n_atoms;

        # These reference values were recorded against the SoA API (energy_short/
        # energy_long taking x, y, z, ComplexF64.(q) separately); Task 2 replaced that
        # API with the AoS form exercised here, so the draws above are converted to
        # poses/charges rather than reshaping the recorded test. The physics -- and
        # therefore the recorded numbers -- do not change between the two call shapes.
        poses = [SVector(x[i], y[i], z[i]) for i in 1:n_atoms]
        E_pme_short = ParticleMeshEwald.energy_short(p, poses, q0)
        E_pme_long = ParticleMeshEwald.energy_long(p, poses, q0)
        E_pme = E_pme_short + E_pme_long

        @test E_pme ≈ baseline_energy[(Lx, Ly, Lz)] atol=1e-8
    end
end