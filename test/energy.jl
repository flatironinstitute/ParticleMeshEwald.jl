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

        q = ComplexF64.(q0)
        E_pme_short = ParticleMeshEwald.energy_short(p, x, y, z, q)
        E_pme_long = ParticleMeshEwald.energy_long(p, x, y, z, q)
        E_pme = E_pme_short + E_pme_long

        @test E_pme ≈ baseline_energy[(Lx, Ly, Lz)] atol=1e-8
    end
end