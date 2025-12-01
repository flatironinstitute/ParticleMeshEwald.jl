function energy_long(pme::PME, x::Vector{T}, y::Vector{T}, z::Vector{T}, q::Vector{Complex{T}}) where T

    @assert length(x) == length(y) == length(z) == length(q) == pme.N

    x .*= 2π / pme.L[1]
    y .*= 2π / pme.L[2]
    z .*= 2π / pme.L[3]

    finufft_setpts!(pme.plan, x, y, z)
    finufft_exec!(pme.plan, q, pme.rho)

    x ./= 2π / pme.L[1]
    y ./= 2π / pme.L[2]
    z ./= 2π / pme.L[3]

    t = zero(T)
    loops = (2 * pme.n_k[1] + 1) * (2 * pme.n_k[2] + 1) * (2 * pme.n_k[3] + 1)
    @inbounds @fastmath @simd for i in 1:loops
        t += (pme.rho[i].re ^2 + pme.rho[i].im ^2) * pme.D[i]
    end

    return t / (pme.L[1] * pme.L[2] * pme.L[3]) / 2
end

@kernel function energy_short_kernel!(@Const(alpha), @Const(neighbor_list), @Const(q), output)
    idx = @index(Global)
    i, j, r = neighbor_list[idx]
    @inbounds @fastmath qi, qj = q[i].re, q[j].re
    @inbounds @fastmath t = qi * qj * erfc(alpha * r) / r
    @inbounds output[Threads.threadid() - 1] += t
end

function energy_short_single(alpha::T, neighbor_list, q::Vector{Complex{T}}) where T
    Es = zero(T)
    for i in 1:length(neighbor_list)
        i, j, r = neighbor_list[i]
        @inbounds @fastmath qi, qj = q[i].re, q[j].re
        @inbounds @fastmath t = qi * qj * erfc(alpha * r) / r
        Es += t
    end
    return Es
end

function energy_short(pme::PME, x::Vector{T}, y::Vector{T}, z::Vector{T}, q::Vector{Complex{T}}; backend = CPU()) where T

    @fastmath @inbounds for i in 1:pme.N
        pme.pos[1, i] = x[i]
        pme.pos[2, i] = y[i]
        pme.pos[3, i] = z[i]
    end

    update!(pme.celllist, pme.pos)
    nb = neighborlist!(pme.celllist)

    Es = zero(T)
    if Threads.nthreads() == 1
        Es = energy_short_single(pme.alpha, nb, q)
    else
        Es_thread = zeros(T, Threads.nthreads())
        kernel = energy_short_kernel!(backend, Threads.nthreads(), size(nb, 1))
        kernel(pme.alpha, nb, q, Es_thread, ndrange = size(nb, 1))
        KernelAbstractions.synchronize(backend)
        Es = sum(Es_thread)
    end
    
    t = pme.alpha / sqrt(π)
    for i in 1:pme.N
        Es -= q[i].re^2 * t
    end

    return Es / 4π
end

function energy(pme::PME, x::Vector{T}, y::Vector{T}, z::Vector{T}, q::Vector{Complex{T}}; backend = CPU()) where T
    return energy_long(pme, x, y, z, q) + energy_short(pme, x, y, z, q; backend = backend)
end