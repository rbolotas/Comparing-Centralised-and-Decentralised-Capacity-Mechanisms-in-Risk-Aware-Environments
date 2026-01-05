using CSV, DataFrames
using Statistics
function define_consumer_parameters!(mod::Model, data::Dict, ts::DataFrame, m::String, scen_number::Int)
    # Parameters - note consumers are rescaled (total number of consumers x share of this type of consumer)
    # Ensure timeseries columns are numeric even if CSV parsed as strings
    numeric_col = name -> map(x -> x isa AbstractString ? parse(Float64, x) : Float64(x), ts[!, name])
    mod.ext[:timeseries][:D_ts] = data["General"]["totConsumers"] * data["Consumers"][m]["Share"] .* numeric_col(data["Consumers"][m]["D"]) 
    mod.ext[:timeseries][:PV] = data["General"]["totConsumers"] * data["Consumers"][m]["Share"] .* data["Consumers"][m]["PV_cap"] .* numeric_col(data["Consumers"][m]["PV_AF"]) 

    # Always define λ_EOM_cap (for all scenarios)
    mod.ext[:parameters][:λ_EOM_cap] = data["Consumers"][m]["λ_EOM_cap"]

    # Compute λ_EOM_ref for elastic scenarios (2+)
    # For scenario 1 (inelastic), this value won't be used but we set a dummy value
    if scen_number == 1
        # Scenario 1 doesn't use λ_EOM_ref (inelastic demand)
        # Set dummy value - will be computed from Scenario 1 results for later scenarios
        λ_EOM_ref = 0.1  # Placeholder
        println("agent = ", m, " (Scenario 1: inelastic, λ_EOM_ref not used)")
    else
        # For scenarios 2+, read from Scenario 1 results
        df = CSV.read("Results/Scenario_1_ref.csv", DataFrame; delim=';')
        demand_col = "D_" * m
        if hasproperty(df, Symbol(demand_col))
            # Use agent-specific reference from Scenario 1
            agent_demand = df[!, Symbol(demand_col)]
            numerator = sum(df.Price .* agent_demand)
            denominator = sum(agent_demand)
            λ_EOM_ref = numerator / denominator
        else
            # Fallback: use system average (for new consumer types added after initial Scenario 1 run)
            λ_EOM_ref = sum(df.Price .* df.Total_Demand) / sum(df.Total_Demand)
            println("  ⚠ New consumer type $(m) - using system average λ_EOM_ref")
        end
        println("agent = ", m, " λ_EOM_ref = ", round(λ_EOM_ref, digits=3))
    end
    mod.ext[:parameters][:λ_EOM_ref] = λ_EOM_ref

    mod.ext[:parameters][:D_EOM_ref] = mod.ext[:timeseries][:D_ts]
    #println("agent = ", m, " λ_EOM_cap = ", mod.ext[:parameters][:λ_EOM_cap])

    # Read D_cap_ratio from config (default to 0.8 if not specified for backward compatibility)
    D_cap_ratio = get(data["Consumers"][m], "D_cap_ratio", 0.8)
    mod.ext[:parameters][:D_EOM_cap] = D_cap_ratio .* mod.ext[:timeseries][:D_ts]
    println("agent = ", m, " D_cap_ratio = ", D_cap_ratio, " (can reduce to ", round(D_cap_ratio*100, digits=0), "% of reference demand)")

    # E_EOM is now (λ_EOM_ref - λ_EOM_cap) / (D_ts - D_EOM_cap)
    mod.ext[:parameters][:E_EOM] = (mod.ext[:parameters][:λ_EOM_ref] .- mod.ext[:parameters][:λ_EOM_cap]) ./ 
                                   (mod.ext[:timeseries][:D_ts] .- mod.ext[:parameters][:D_EOM_cap])

    # λ_EOM_0 is now λ_EOM_ref - E_EOM * D_ts
    mod.ext[:parameters][:λ_EOM_0] = mod.ext[:parameters][:λ_EOM_ref] .- (mod.ext[:parameters][:E_EOM] .* mod.ext[:timeseries][:D_ts])

    println("\n--- Consumer Demand Curve Parameter Statistics ---")
    println("λ_EOM_cap:   value = ", round(mod.ext[:parameters][:λ_EOM_cap], digits=3))
    println("E_EOM:       min = ", round(minimum(mod.ext[:parameters][:E_EOM]), digits=3),
            ", max = ", round(maximum(mod.ext[:parameters][:E_EOM]), digits=3),
            ", mean = ", round(mean(mod.ext[:parameters][:E_EOM]), digits=3))
    println("λ_EOM_0:     min = ", round(minimum(mod.ext[:parameters][:λ_EOM_0]), digits=3),
            ", max = ", round(maximum(mod.ext[:parameters][:λ_EOM_0]), digits=3),
            ", mean = ", round(mean(mod.ext[:parameters][:λ_EOM_0]), digits=3))
    println("-------------------------------------------------\n")
    
    # Add risk parameters from global RiskMetrics (changed from consumer-specific)
    mod.ext[:parameters][:β_co] = data["RiskMetrics"]["β_co"]
    mod.ext[:parameters][:γ_co] = data["RiskMetrics"]["γ_co"]

    # scarcity price limit // RO strike price
    mod.ext[:parameters][:λ_EOM_scr] = data["dCM"]["λ_EOM_scr"]
    return mod
end

