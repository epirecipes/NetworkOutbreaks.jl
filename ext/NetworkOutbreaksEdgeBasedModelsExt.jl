module NetworkOutbreaksEdgeBasedModelsExt

using NetworkOutbreaks
using EdgeBasedModels: DiseaseProgression, DiseaseStage, DiseaseTransition
using EdgeBasedModels.Symbolics: Symbolics

"""
    OutbreakModel(prog::DiseaseProgression, parameters::AbstractDict)

Convert an EBM `DiseaseProgression` into a `NetworkOutbreaks.OutbreakModel`.

The susceptible state is mapped to its own compartment (with no infection
transmission of its own). Each `DiseaseStage` becomes a compartment whose
`infectious` flag is set when its `transmission_rate` evaluates to a
positive number after substituting `parameters`.

For each infectious stage, an infection transition `susceptible → entry`
is added with `via = [stage]` and rate equal to that stage's
transmission rate. This faithfully represents per-stage transmission
heterogeneity (e.g., distinct β for E vs I).

Spontaneous transitions are passed through with their rate substituted.
"""
function NetworkOutbreaks.OutbreakModel(prog::DiseaseProgression,
                                        parameters::AbstractDict)
    sus = prog.susceptible
    stage_names = [s.name for s in prog.stages]
    comps = vcat([sus], stage_names)

    # Substitute parameters into a symbolic rate expression and reduce to
    # a Float64. Accepts Symbol, Real, or Symbolics.Num.
    rate_value(r) = _eval_rate(r, parameters)

    inf_flags = vcat(false,
                     [rate_value(s.transmission_rate) > 0 for s in prog.stages])

    trs = NetworkOutbreaks.OutbreakTransition[]
    # Infection transitions: susceptible → entry, one per infectious stage,
    # with the via-list pinning that stage as the catalyst.
    for s in prog.stages
        β = rate_value(s.transmission_rate)
        β > 0 || continue
        push!(trs, NetworkOutbreaks.OutbreakTransition(sus, prog.entry,
                                                       β, :infection;
                                                       via = [s.name]))
    end
    # Spontaneous transitions: source → target at the named rate.
    for tr in prog.transitions
        push!(trs, NetworkOutbreaks.OutbreakTransition(tr.source, tr.target,
                                                       rate_value(tr.rate),
                                                       :spontaneous))
    end

    return NetworkOutbreaks.OutbreakModel(comps, inf_flags, trs;
                                          name = :ebm_outbreak)
end

# --- internal: collapse a possibly-symbolic rate to a Float64 ---

_eval_rate(r::Real, ::AbstractDict) = Float64(r)
function _eval_rate(r::Symbol, parameters::AbstractDict)
    haskey(parameters, r) ||
        throw(ArgumentError("missing parameter $(r) when adapting EBM progression"))
    return Float64(parameters[r])
end
function _eval_rate(r, parameters::AbstractDict)
    # Symbolics.Num or similar
    if r isa Symbolics.Num
        # Build a substitution Dict mapping every symbolic param to a Num.
        subs = Dict()
        for (k, v) in parameters
            subs[Symbolics.variable(k)] = v
        end
        substituted = Symbolics.simplify(Symbolics.substitute(r, subs))
        value = Symbolics.value(substituted)
        value isa Real && return Float64(value)
        if isempty(Symbolics.get_variables(substituted))
            f = Symbolics.build_function(substituted; expression = Val{false})
            f = f isa Tuple ? first(f) : f
            return Float64(f())
        end
        throw(ArgumentError("unresolved symbolic rate after substitution: $r"))
    end
    return Float64(r)
end

end # module
