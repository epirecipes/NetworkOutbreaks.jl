#=
interventions.jl

Declarative intervention types for modifying simulation state at
scheduled times or when state-based conditions are met.

Intervention types:
- `ScheduledRateChange`  — modify a transition rate at a fixed time
- `ScheduledStateChange` — move nodes between compartments at a fixed time
- `ThresholdIntervention` — fire a state change when a compartment count
                           crosses a threshold

All interventions are processed in the DirectSSA main loop between events.
Other algorithms raise ArgumentError if interventions are provided.
=#

abstract type AbstractIntervention end

"""
    ScheduledRateChange(time, from, to, type, new_rate)

At simulation time `time`, change the rate of the transition
`from → to` (type `:infection` or `:spontaneous`) to `new_rate`.
"""
struct ScheduledRateChange <: AbstractIntervention
    time::Float64
    from::Symbol
    to::Symbol
    type::Symbol
    new_rate::Float64
end

"""
    ScheduledStateChange(time, compartment, fraction)

At simulation time `time`, move `fraction` of all nodes to `compartment`.
Nodes are selected uniformly at random from those NOT already in
`compartment`. This models vaccination pulses (S → V) or mass treatment.
"""
struct ScheduledStateChange <: AbstractIntervention
    time::Float64
    compartment::Symbol
    fraction::Float64
end

"""
    ThresholdIntervention(compartment, direction, threshold, action)

When the count in `compartment` crosses `threshold` in `direction`
(`:above` or `:below`), apply `action` (a `ScheduledRateChange` or
`ScheduledStateChange` with `time = NaN` — the time is filled in at
trigger). Fires at most once.
"""
struct ThresholdIntervention <: AbstractIntervention
    compartment::Symbol
    direction::Symbol   # :above | :below
    threshold::Int
    action::AbstractIntervention
    function ThresholdIntervention(comp, dir, thresh, act)
        dir in (:above, :below) ||
            throw(ArgumentError("direction must be :above or :below"))
        return new(comp, dir, thresh, act)
    end
end

"""
    InterventionPlan(interventions)

An ordered collection of interventions to apply during a simulation.
Scheduled interventions are sorted by time; threshold interventions
are checked after each event.
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
    sort!(scheduled; by = _intervention_time)
    return InterventionPlan(scheduled, thresholds)
end

InterventionPlan() = InterventionPlan(AbstractIntervention[], ThresholdIntervention[])

_intervention_time(iv::ScheduledRateChange) = iv.time
_intervention_time(iv::ScheduledStateChange) = iv.time
_intervention_time(::ThresholdIntervention) = Inf

Base.isempty(plan::InterventionPlan) =
    Base.isempty(plan.scheduled) && Base.isempty(plan.thresholds)
