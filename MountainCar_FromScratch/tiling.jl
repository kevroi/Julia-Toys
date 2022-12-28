struct Tiling{N,Tr<:AbstractRange}
    ranges::NTuple{N,Tr}
    inds::LinearIndices{N,NTuple{N,Base.OneTo{Int}}}
end

Tiling(ranges::AbstractRange...) = Tiling(
    ranges,
    LinearIndices(Tuple(length(r) - 1 for r in ranges))
)

Base.length(t::Tiling) = reduce(*, (length(r) - 1 for r in t.ranges))

function Base.:-(t::Tiling, xs)
    Tiling((r .- x for (r, x) in zip(t.ranges, xs))...)
end

encode(range::AbstractRange, x) = floor(Int, div(x - range[1], step(range)) + 1)

encode(t::Tiling, xs) = t.inds[CartesianIndex(Tuple(map(encode,  t.ranges, xs)))]