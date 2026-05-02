#=
algorithms/has.jl

Hierarchical Adaptive Sampling (HAS) — scaffold / stub.

=== Algorithm summary (not yet implemented) ===

HAS was introduced by the Kleist Lab (https://github.com/KleistLab/HAS) as a
sub-quadratic exact SSA for large heterogeneous networks.  The key idea is to
replace the flat rate list of the Direct method with a complete binary tree of
partial-rate sums:

  • Leaf i stores the total hazard  a_i = spont_i + infect_i  for node i.
  • Each internal node stores the sum of hazards in its subtree.
  • The root stores  Λ = Σ a_i.

Event selection (O(log N)):
  Starting from the root, at each level draw a uniform [0,1) and compare it
  to the left-child fraction to decide whether to descend left or right.
  Reaching a leaf identifies the firing node.

Per-event update (O(Δ log N)):
  When node v fires and changes compartment, recompute a_v and propagate the
  difference up the tree (O(log N)).  Also update each of v's Δ neighbours
  (O(Δ log N) total).  For fixed-degree graphs this is O(log N) per event.

Comparison with CompositionRejection:
  Both achieve sub-linear update cost; HAS gives O(log N) deterministically
  regardless of rate heterogeneity, while Composition–Rejection is O(1)
  amortised when rates are well-clustered.  HAS is preferred for large N
  (≥ 10^4) or highly heterogeneous degree distributions (power-law networks).

Integration with TimeVaryingNetwork (planned):
  A graph update (add/remove edge) affects hazards for 2 nodes → 2 tree
  updates, each O(log N).

See  docs/HAS_PLAN.md  for the full implementation roadmap.
See  https://github.com/KleistLab/HAS  for the reference Cython implementation.
=#

"""
    HAS <: OutbreakAlgorithm

Hierarchical Adaptive Sampling algorithm (stub — not yet implemented).

Dispatch via `simulate(spec; algorithm = HAS())` will raise an error until
the implementation is complete.  See `docs/HAS_PLAN.md` for the roadmap.
"""
struct HAS <: OutbreakAlgorithm end

function _simulate_impl(::HAS, spec::OutbreakSpec, seed::UInt64, keep::Symbol)
    throw(ErrorException(
        "HAS algorithm not yet implemented; see docs/HAS_PLAN.md for the " *
        "implementation plan and tracking issue."))
end
