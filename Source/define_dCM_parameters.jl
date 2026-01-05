function define_dCM_parameters!(dCM::Dict, data::Dict, ts::DataFrame, scenario_overview_row)
    scen_number = scenario_overview_row["scen_number"]
    
    # Initialize inflexible capacity demand (passive HV load)
    dCM["C_dCM_inflexible"] = 0.0
    
    # In scenarios 6-7, inflexible HV load needs capacity coverage
    # Active consumers (LV_LOW, etc.) can also demand capacity via D_dCM
    # Market clears: sum(C_dCM[Gen]) + sum(C_dCM[Cons]) = C_dCM_inflexible
    if scen_number == 6 || scen_number == 7
        max_load = maximum(ts.HV_LOAD)  # Peak passive load in GW
        dCM["C_dCM_inflexible"] = max_load
        println("Setting inflexible dCM demand (passive HV load): $(round(max_load, digits=4)) GW")
    end
    
    return dCM
end