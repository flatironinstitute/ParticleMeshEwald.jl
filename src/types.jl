mutable struct PME{T, TP, TD, TR, TC}
    alpha::T
    L::NTuple{3, T}
    s::T
    N::Int

    eps::T
    r_c::T
    k_c::T
    
    plan::TP
    n_k::Tuple{Int, Int, Int}
    D::TD
    rho::TR

    pos::Matrix{T}
    celllist::TC

    function PME(alpha::T, L::NTuple{3, T}, s::T, N::Int) where T
        eps = exp(-s^2) / s^2
        r_c = s / alpha
        k_c = 2 * s * alpha
    
        n_k = (ceil(Int, k_c / (2π / L[1])), ceil(Int, k_c / (2π / L[2])), ceil(Int, k_c / (2π / L[3])))
        D = zeros(T, (2 * n_k[1] + 1, 2 * n_k[2] + 1, 2 * n_k[3] + 1))

        exp_kxs = [exp(- (2π * i / L[1])^2 / (4 * alpha^2)) for i in -n_k[1]:n_k[1]]
        exp_kys = [exp(- (2π * j / L[2])^2 / (4 * alpha^2)) for j in -n_k[2]:n_k[2]]
        exp_kzs = [exp(- (2π * k / L[3])^2 / (4 * alpha^2)) for k in -n_k[3]:n_k[3]]

        @fastmath @inbounds for i in 1:2 * n_k[1] + 1, j in 1:2 * n_k[2] + 1, k in 1:2 * n_k[3] + 1
            k_x = (i - n_k[1] - 1) * (2π / L[1])
            k_y = (j - n_k[2] - 1) * (2π / L[2])
            k_z = (k - n_k[3] - 1) * (2π / L[3])
            D[i, j, k] = exp_kxs[i] * exp_kys[j] * exp_kzs[k] / (k_x^2 + k_y^2 + k_z^2)
        end
        D[n_k[1] + 1, n_k[2] + 1, n_k[3] + 1] = zero(T)

        rho = zeros(Complex{T}, 2 * n_k[1] + 1, 2 * n_k[2] + 1, 2 * n_k[3] + 1)

        # plan = PlanNUFFT(T, (2 * n_k[1] + 1, 2 * n_k[2] + 1, 2 * n_k[3] + 1); m = HalfSupport(8), σ = 2.0, fftshift=true) # mechine precision to prevent error from nufft
        plan = finufft_makeplan(1, [2 * n_k[1] + 1, 2 * n_k[2] + 1, 2 * n_k[3] + 1], +1, 1, eps, dtype=T)

        pos = zeros(T, 3, N);
        celllist = InPlaceNeighborList(x=pos, cutoff=r_c, unitcell=[L[1], L[2], L[3]], parallel=true)

        new{T, typeof(plan), typeof(D), typeof(rho), typeof(celllist)}(alpha, L, s, N, eps, r_c, k_c, plan, n_k, D, rho, pos, celllist)
    end
end