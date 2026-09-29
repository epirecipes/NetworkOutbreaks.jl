#=
interventions.jl

Declarative intervention types for modifying a simulation at scheduled times or when a compartment count crosses a
threshold.

- `ScheduledRateChange`   — change a transition rate at a fixed time
- `ScheduledStateChange`  — move nodes into a compartment at a fixed time (vaccination pulses, importations)
- `ThresholdIntervention` — apply one of the above when a count crosses a threshold (fires at most once)

Interventions are supported by `DirectSSA`, `NextReaction` and `HAS`; `CompositionRejection` rejects them.

Every run works on its own copy of the transition rates, so a rate change never modifies `spec.model`, other runs
of an ensemble, or later calls to `simulate` (M1). Scheduled interventions and time-varying-network updates are
processed in one time-ordered queue (M5), and a run does not stop at zero total rate while any of them is still
pending within `tspan` (M4).
=#

"""
    AbstractIntervention

Supertype of `ScheduledRateChange`, `ScheduledStateChange` and `ThresholdIntervention`; collect them in an
`InterventionPlan` and pass it to `simulate(spec; interventions = plan)`.
"""
abstract type AbstractIntervention end

"""
    ScheduledRateChange(time, from, to, type, new_rate; via = nothing)

At simulation time `time`, set the rate of the transition `from → to` of type `type` (`:infection`,
`:contact_trace` or `:spontaneous`) to `new_rate`.

The change applies to the current run only: `spec.model` is never modified. The transition must be unique: if the
model has several transitions `from → to` of this type (for example one per infector, distinguished by `via`), pass
`via` (a vector of compartment names, compared as a set) to select one; a contact transition declared with an empty
`via` is matched by the names of the infectious compartments, which is what an empty `via` means. An ambiguous or
unmatched rate change is an `ArgumentError` when the simulation starts.
"""
struct ScheduledRateChange <: AbstractIntervention
    time::Float64
    from::Symbol
    to::Symbol
    type::Symbol
    new_rate::Float64
    via::Union{Nothing, Vector{Symbol}}
    # The only constructor, so every instance is validated.
    function ScheduledRateChange(time::Real, from::Symbol, to::Symbol, type::Symbol, new_rate::Real;
                                 via = nothing)
        new_rate >= 0 || throw(ArgumentError("new_rate must be non-negative; got $(new_rate)"))
        type in (:infection, :contact_trace, :spontaneous) ||
            throw(ArgumentError("unknown transition type $(type)"))
        return new(Float64(time), from, to, type, Float64(new_rate),
                   via === nothing ? nothing : Symbol[v for v in via])
    end
end

"""
    ScheduledStateChange(time, compartment, fraction; from = nothing, basis = :population)

At simulation time `time`, move nodes into `compartment`, chosen uniformly at random without replacement from the
*eligible* nodes: those in one of the compartments `from` (default: every compartment except `compartment`) that are
not already in `compartment`.

- `basis = :population` (default): move `round(fraction · N)` nodes (capped at the number eligible), so
  `fraction` is a fraction of the whole population.
- `basis = :eligible`: move `round(fraction · #eligible)` nodes.

Rounding is to the nearest integer with ties away from zero. A vaccination pulse `S → V` is
`ScheduledStateChange(t, :V, ρ; from = [:S])`; without `from`, infectious and recovered nodes are moved as well.
A node moved into an infected compartment (an importation) is counted as an infection by `final_size` and
`reinfection_histogram`.
"""
struct ScheduledStateChange <: AbstractIntervention
    time::Float64
    compartment::Symbol
    fraction::Float64
    from::Union{Nothing, Vector{Symbol}}
    basis::Symbol
end

function ScheduledStateChange(time::Real, compartment::Symbol, fraction::Real;
                              from = nothing, basis::Symbol = :population)
    (isfinite(fraction) && 0 <= fraction <= 1) ||
        throw(ArgumentError("ScheduledStateChange fraction must lie in [0, 1]; got $(fraction)"))
    basis in (:population, :eligible) ||
        throw(ArgumentError("basis must be :population or :eligible; got $(basis)"))
    return ScheduledStateChange(Float64(time), compartment, Float64(fraction),
                                from === nothing ? nothing : Symbol[c for c in from], basis)
end

"""
    ThresholdIntervention(compartment, direction, threshold, action)

When the count in `compartment` is `≥ threshold` (`direction = :above`) or `≤ threshold` (`:below`), apply `action`
(a `ScheduledRateChange` or `ScheduledStateChange`; its `time` field is ignored, conventionally `NaN`). Fires at
most once per run. The condition is checked at the start of the run, after every event and after every scheduled
intervention, so a threshold that holds is acted on at the first time it holds.
"""
struct ThresholdIntervention <: AbstractIntervention
    compartment::Symbol
    direction::Symbol   # :above | :below
    threshold::Int
    action::AbstractIntervention
    function ThresholdIntervention(comp, dir, thresh, act)
        dir in (:above, :below) ||
            throw(ArgumentError("direction must be :above or :below"))
        act isa ThresholdIntervention &&
            throw(ArgumentError("the action of a ThresholdIntervention cannot itself be a ThresholdIntervention"))
        return new(comp, dir, thresh, act)
    end
end

"""
    InterventionPlan(interventions)

An ordered collection of interventions to apply during a simulation. Scheduled interventions are sorted by time
(stably, so interventions at the same time are applied in the order given); threshold interventions are checked at
the start, after each event and after each scheduled intervention.
"""
struct InterventionPlan
    scheduled::Vector{AbstractIntervention}    # sorted by time
    thresholds::Vector{ThresholdIntervention}
end

function InterventionPlan(interventions::AbstractVector{<:AbstractIntervention})
    scheduled = AbstractIntervention[]
    thresholds = ThresholdIntervention[]
    for iv in interventions
        if iv isa ThresholdIntervention
            push!(thresholds, iv)
        else
            push!(scheduled, iv)
        end
    end
    sort!(scheduled; by = _intervention_time, alg = Base.Sort.DEFAULT_STABLE)
    return InterventionPlan(scheduled, thresholds)
end

InterventionPlan() = InterventionPlan(AbstractIntervention[], ThresholdIntervention[])

_intervention_time(iv::ScheduledRateChange) = iv.time
_intervention_time(iv::ScheduledStateChange) = iv.time
_intervention_time(::ThresholdIntervention) = Inf

Base.isempty(plan::InterventionPlan) =
    Base.isempty(plan.scheduled) && Base.isempty(plan.thresholds)

# ---------------------------------------------------------------------------------------------------------------
# Validation (at the start of `simulate`)
# ---------------------------------------------------------------------------------------------------------------

function _check_compartment(model::OutbreakModel, c::Symbol, what)
    haskey(model.index_of, c) || throw(ArgumentError("unknown compartment $(c) in $(what)"))
    return nothing
end

# Index of the unique transition a rate change applies to.
function _rate_change_target(iv::ScheduledRateChange, model::OutbreakModel)
    matches = Int[]
    for (j, tr) in pairs(model.transitions)
        tr.from == iv.from && tr.to == iv.to && tr.type == iv.type || continue
        iv.via === nothing || Set(_effective_via(tr, model)) == Set(iv.via) || continue
        push!(matches, j)
    end
    isempty(matches) &&
        throw(ArgumentError("ScheduledRateChange: the model has no $(iv.type) transition $(iv.from) → $(iv.to)" *
                            (iv.via === nothing ? "" : " with via = $(iv.via)")))
    length(matches) > 1 &&
        throw(ArgumentError("ScheduledRateChange: $(length(matches)) $(iv.type) transitions $(iv.from) → $(iv.to) " *
                            "match (via = $([model.transitions[j].via for j in matches])); pass `via` to choose one"))
    return matches[1]
end

_validate_intervention(iv::ScheduledRateChange, model) = (_rate_change_target(iv, model); nothing)

function _validate_intervention(iv::ScheduledStateChange, model)
    _check_compartment(model, iv.compartment, "ScheduledStateChange")
    iv.from === nothing || foreach(c -> _check_compartment(model, c, "ScheduledStateChange(from = …)"), iv.from)
    return nothing
end

function _validate_intervention(iv::ThresholdIntervention, model)
    _check_compartment(model, iv.compartment, "ThresholdIntervention")
    return _validate_intervention(iv.action, model)
end

function _validate_interventions(plan::InterventionPlan, model::OutbreakModel, tspan)
    for iv in plan.scheduled
        t = _intervention_time(iv)
        isfinite(t) ||
            throw(ArgumentError("scheduled intervention time must be finite; got $(t) (NaN is only for the " *
                                "action of a ThresholdIntervention)"))
        t >= tspan[1] ||
            throw(ArgumentError("scheduled intervention at t = $(t) precedes the start of the run ($(tspan[1]))"))
        _validate_intervention(iv, model)
    end
    foreach(iv -> _validate_intervention(iv, model), plan.thresholds)
    return nothing
end

# ---------------------------------------------------------------------------------------------------------------
# Application (per run). `rm` is the run's `_RunModel`, `state` its `OutbreakState`.
# Each method returns the nodes it moved (a vector), or `nothing` when rates changed and every hazard may differ.
# ---------------------------------------------------------------------------------------------------------------

function _apply_intervention!(iv::ScheduledRateChange, rm, state, n, rng)
    j = _rate_change_target(iv, rm.model)
    rm.rates[j] = iv.new_rate            # the run's own copy; spec.model is untouched
    _refresh_spont_totals!(rm)
    return nothing
end

function _apply_intervention!(iv::ScheduledStateChange, rm, state, n, rng)
    model = rm.model
    target = model.index_of[iv.compartment]
    allowed = falses(rm.C)
    if iv.from === nothing
        allowed .= true
    else
        for c in iv.from
            allowed[model.index_of[c]] = true
        end
    end
    allowed[target] = false
    eligible = Int[]
    @inbounds for v in 1:n
        allowed[state.node_state[v]] && push!(eligible, v)
    end
    base = iv.basis === :population ? n : length(eligible)
    k = min(round(Int, iv.fraction * base, RoundNearestTiesAway), length(eligible))
    k <= 0 && return Int[]
    selected = sample(rng, eligible, k; replace = false)
    for v in selected
        _move_node!(state, rm, v, target)
    end
    return selected
end
