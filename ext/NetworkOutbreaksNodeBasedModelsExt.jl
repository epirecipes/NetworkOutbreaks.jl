module NetworkOutbreaksNodeBasedModelsExt

using NetworkOutbreaks
using NodeBasedModels: CompartmentalModel, Compartment, Transition

"""
    OutbreakModel(model::CompartmentalModel, parameters::Dict{Symbol, <:Real})

Convert an NBM `CompartmentalModel` into a `NetworkOutbreaks.OutbreakModel`
by substituting symbolic transition rates from `parameters`.
"""
function NetworkOutbreaks.OutbreakModel(model::CompartmentalModel,
                                        parameters::AbstractDict)
    comps = [c.name for c in model.compartments]
    inf_flags = [c.infectious for c in model.compartments]
    trs = NetworkOutbreaks.OutbreakTransition[]
    for tr in model.transitions
        haskey(parameters, tr.rate) ||
            throw(ArgumentError("missing parameter $(tr.rate) in NBM->OutbreakModel adapter"))
        rate_val = Float64(parameters[tr.rate])
        push!(trs, NetworkOutbreaks.OutbreakTransition(tr.from, tr.to,
                                                       rate_val, tr.type))
    end
    return NetworkOutbreaks.OutbreakModel(comps, inf_flags, trs;
                                          name = model.name)
end

end # module
