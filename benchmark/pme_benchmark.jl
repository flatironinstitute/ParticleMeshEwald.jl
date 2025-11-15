using ParticleMeshEwald
using CSV, DataFrames
using BenchmarkTools
using Random

function benchmark_pme(n_atoms, L, alpha, s)
    pme = PME(alpha, (L, L, L), s, n_atoms)
    x = rand(n_atoms) .* L
    y = rand(n_atoms) .* L
    z = rand(n_atoms) .* L
    q = rand(ComplexF64, n_atoms)

    t_short = @belapsed ParticleMeshEwald.energy_short($(pme), $(x), $(y), $(z), $(q))
    t_long = @belapsed ParticleMeshEwald.energy_long($(pme), $(x), $(y), $(z), $(q))
    t_total = @belapsed ParticleMeshEwald.energy($(pme), $(x), $(y), $(z), $(q))

    return t_short, t_long, t_total
end

function main()
    rho = 1.0
    ns = Int.(ceil.(10.0 .^ range(3, 7, length = 30)))

    sa = [(2, 1.5)]
    for (s, alpha) in sa
        df = CSV.write(joinpath(@__DIR__, "pme_benchmark_s$(s).csv"), DataFrame(n = Int[], s = Float64[], alpha = Float64[], t_short = Float64[], t_long = Float64[], t_total = Float64[]))
        for n in ns
            L = (n / rho) ^ (1/3)
            t_short, t_long, t_total = benchmark_pme(n, L, alpha, Float64(s))
            CSV.write(df, DataFrame(n = n, s = Float64(s), alpha = Float64(alpha), t_short = t_short, t_long = t_long, t_total = t_total), append = true)
        end
    end
end

main()