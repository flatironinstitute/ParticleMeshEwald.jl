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

@testset "compare ewald" begin

    n_atoms = 1000
    for Lx in [10.0, 20.0], Ly in [10.0, 20.0], Lz in [10.0, 20.0]

        p = PME(1.0, (Lx, Ly, Lz), 4.9, n_atoms)
        x = rand(n_atoms) .* Lx
        y = rand(n_atoms) .* Ly
        z = rand(n_atoms) .* Lz  

        q0 = rand(n_atoms);
        q0 .-= sum(q0) / n_atoms;

        q = ComplexF64.(q0)
        # E_pme = ParticleMeshEwald.energy(p, x, y, z, q)
        E_pme_short = ParticleMeshEwald.energy_short(p, x, y, z, q)
        E_pme_long = ParticleMeshEwald.energy_long(p, x, y, z, q)
        E_pme = E_pme_short + E_pme_long

        boundary = Boundary((Lx, Ly, Lz), (1, 1, 1))

        atoms = Vector{Atom{Float64}}()
        for i in 1:n_atoms
            push!(atoms, Atom(type = 1, mass = 1.0, charge = q[i].re))
        end

        info = SimulationInfo(n_atoms, atoms, (0.0, Lx, 0.0, Ly, 0.0, Lz), boundary; min_r = 0.01, temp = 1.0)
        for i in 1:n_atoms
            info.particle_info[i].position = Point(x[i], y[i], z[i])
        end
        Ewald3D_interaction = Ewald3DInteraction(n_atoms, 4.9, 1.0, (Lx, Ly, Lz))

        neighbor = CellList3D(info, Ewald3D_interaction.r_c, boundary, 1)
        
        charge = [atoms[info.particle_info[i].id].charge for i in 1:Ewald3D_interaction.n_atoms]
        position = [info.particle_info[i].position for i in 1:Ewald3D_interaction.n_atoms]

        energy_ewald_long = EwaldSummations.Ewald3D_long_energy(Ewald3D_interaction, position, charge)
        energy_ewald_short = EwaldSummations.Ewald3D_short_energy(Ewald3D_interaction, neighbor, position, charge)

        energy_ewald = energy_ewald_long + energy_ewald_short

        @test abs(E_pme - energy_ewald) < 1e-7
    end
end