using SIMD

@generated function horner_scalar(x::T, coeffs::NTuple{N,T}) where {N,T}
    ex = :(coeffs[$N])
    for k = N-1:-1:1
        ex = :(muladd(x, $ex, coeffs[$k]))
    end
    ex
end


@generated function horner_vec(x::Vec{W,T}, coeffs::NTuple{N,T}) where {W,N,T}
    ex = :(Vec{W,T}(coeffs[$N]))
    for k = N-1:-1:1
        ex = :(muladd(x, $ex, Vec{W,T}(coeffs[$k])))
    end
    ex
end

function horner!(ys::Vector{T}, xs::Vector{T}, coeffs::NTuple{N,T}) where {N,T<:Union{Float32,Float64}}
    @assert length(ys) == length(xs)

    W = T === Float32 ? 16 : 8  # AVX-512: 16xF32 or 8xF64
    len = length(xs)
    i = 1

    @inbounds while i <= len - W + 1
        xv = vload(Vec{W,T}, xs, i)      # load from xs[i:i+W-1]
        yv = horner_vec(xv, coeffs)      # use zmm
        vstore(yv, ys, i)            
        i += W
    end

    while i <= len
        ys[i] = horner_scalar(xs[i], coeffs)
        i += 1
    end

    return ys
end