@testset "core works without ExTinyMD" begin
    # The requirement this whole phase exists for: ParticleMeshEwald's core
    # query API must work with ExTinyMD never loaded. A @testset inside the
    # normal suite does not prove that on its own -- test/runtests.jl itself
    # does `using ExTinyMD` (for the adapter test), so by the time this
    # testset runs, ExTinyMD is already loaded in *this* process. The only way
    # to prove the core doesn't need it is to run in a fresh process that
    # never imports it.
    script = """
    using ParticleMeshEwald, StaticArrays
    @assert !haskey(Base.loaded_modules, Base.PkgId(
        Base.UUID("fec76197-d59f-46dd-a0ed-76a83c21f7aa"), "ExTinyMD"))
    n = 50; L = (20.0, 20.0, 20.0)
    poses = [SVector(rand()*L[1], rand()*L[2], rand()*L[3]) for _ in 1:n]
    charges = [isodd(i) ? 1.0 : -1.0 for i in 1:n]
    pme = PME(0.5, L, 4.0, n)   # r_c = s/α = 8.0 < min(L)/2 = 10.0
    E = ParticleMeshEwald.energy(pme, poses, charges)
    @assert isfinite(E)
    print("OK")
    """
    out = read(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $script`, String)
    @test out == "OK"
end
