function ADMM_subroutine!(m::String, results::Dict, ADMM::Dict, EOM::Dict, cCM::Dict, dCM::Dict, mod::Model, agents::Dict, scen_number::Int, TO::TimerOutput)
TO_local = TimerOutput()
# Calculate penalty terms ADMM and update price to most recent value 
@timeit TO_local "Compute ADMM penalty terms" begin
    mod.ext[:parameters][:g_bar] = results["g"][m][end] - (1 / (EOM["nAgents"] + 1)) * ADMM["Imbalances"]["EOM"][end]
    mod.ext[:parameters][:λ_EOM] = results["λ"]["EOM"][end] 
    mod.ext[:parameters][:ρ_EOM] = ADMM["ρ"]["EOM"][end]
    # Update capacity market parameters
    mod.ext[:parameters][:C_cCM_bar] = results["C_cCM"][m][end]- (1 / (cCM["nAgents"] + 1)) * ADMM["Imbalances"]["cCM"][end]
    mod.ext[:parameters][:λ_cCM] = results["λ"]["cCM"][end]
    mod.ext[:parameters][:ρ_cCM] = ADMM["ρ"]["cCM"][end]

    # Update decentralised capacity market parameters
    if m in agents[:Gen]
        # For generators: use C_dCM target (capacity supplied)
        mod.ext[:parameters][:C_dCM_bar] = results["C_dCM"][m][end]- (1 / (dCM["nAgents"] + 1)) * ADMM["Imbalances"]["dCM"][end]
    elseif m in agents[:Cons]
        # For consumers: use D_dCM target (capacity demanded)
        # Since C_dCM = -D_dCM for consumers, D_dCM_bar = -C_dCM_bar
        mod.ext[:parameters][:C_dCM_bar] = results["C_dCM"][m][end]- (1 / (dCM["nAgents"] + 1)) * ADMM["Imbalances"]["dCM"][end]
        mod.ext[:parameters][:D_dCM_bar] = -mod.ext[:parameters][:C_dCM_bar]
    end
    mod.ext[:parameters][:λ_dCM] = results["λ"]["dCM"][end]
    mod.ext[:parameters][:ρ_dCM] = ADMM["ρ"]["dCM"][end]

end
#println(m)

# Solve agents decision problems:
if m in agents[:Gen]
    @timeit TO_local "Solve generator problems" begin
        solve_generator_agent!(mod, scen_number, m)  # Pass generator type 'm'
    end
elseif m in agents[:Cons]
    # For Scenario 1 (inelastic), hard-mask ENS: ENS = 0 whenever price ≤ cap; allow ENS only when price > cap
    if scen_number == 1
        let λ = results["λ"]["EOM"][end],
            λ_cap = value(mod.ext[:parameters][:λ_EOM_cap]),
            ENS = mod.ext[:variables][:ENS],
            JH = mod.ext[:sets][:JH]
            for jh in JH
                if λ[jh] <= λ_cap
                    # ENS not allowed below/at cap
                    if !has_upper_bound(ENS[jh]) || upper_bound(ENS[jh]) != 0.0
                        set_upper_bound(ENS[jh], 0.0)
                    end
                else
                    # ENS allowed above cap
                    if has_upper_bound(ENS[jh]) && upper_bound(ENS[jh]) == 0.0
                        delete_upper_bound(ENS[jh])
                    end
                end
            end
        end
    end
    @timeit TO_local "Solve consumer problems" begin
        solve_consumer_agent!(mod, scen_number)  
    end
end

# Query results
@timeit TO_local "Query results" begin
    status = termination_status(mod)
    
    if status in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
        push!(results["g"][m], collect(value.(mod.ext[:variables][:g])))
        if m in agents[:Gen]
            push!(results["C"][m], value(mod.ext[:variables][:C]))
        end
        push!(results["C_cCM"][m], value(mod.ext[:variables][:C_cCM]))
        push!(results["C_dCM"][m], value(mod.ext[:variables][:C_dCM]))
        
        if m in agents[:Cons]
            push!(results["ENS"][m], collect(value.(mod.ext[:variables][:ENS])))
            push!(results["D"][m], collect(value.(mod.ext[:variables][:D])))
            push!(results["D_elastic"][m], collect(value.(mod.ext[:variables][:D_elastic])))
            push!(results["PV_curt"][m], collect(value.(mod.ext[:variables][:PV_curt])))
            
            # Separate ENS into involuntary vs elastic based on price conditions
            ens_values = collect(value.(mod.ext[:variables][:ENS]))
            d_elastic_values = collect(value.(mod.ext[:variables][:D_elastic]))
            lambda_eom = collect(value.(mod.ext[:parameters][:λ_EOM]))
            
            # Initialize separated components
            ens_involuntary = zeros(length(ens_values))
            d_elastic_separated = zeros(length(d_elastic_values))
            
            # Get price cap for all scenarios (exists in config)
            lambda_eom_cap = value(mod.ext[:parameters][:λ_EOM_cap])
            
            if scen_number == 1
                # Scenario 1 (inelastic): all curtailment is involuntary ENS, no voluntary elastic response
                for jh in eachindex(ens_values)
                    ens_involuntary[jh] = ens_values[jh]
                    d_elastic_separated[jh] = 0.0
                end
            else
                # For scenarios 2+: derive voluntary/involuntary from actual demand reduction, not ENS
                # Shortfall per hour is the reduction from reference demand: max(0, D_ts - D)
                lambda_eom_0 = collect(value.(mod.ext[:parameters][:λ_EOM_0]))
                D_ts = collect(mod.ext[:timeseries][:D_ts])
                D_vals = collect(value.(mod.ext[:variables][:D]))
                for jh in eachindex(D_vals)
                    shortfall = max(0.0, D_ts[jh] - D_vals[jh])
                    if lambda_eom[jh] <= lambda_eom_0[jh]
                        # if price <= λ_0: no participation
                        ens_involuntary[jh] = 0.0
                        d_elastic_separated[jh] = 0.0
                    elseif lambda_eom[jh] > lambda_eom_cap
                        # if price > λ_cap: any shortfall is involuntary ENS
                        ens_involuntary[jh] = shortfall
                        d_elastic_separated[jh] = 0.0
                    else
                        # λ_0 < price ≤ λ_cap: shortfall is voluntary (elastic)
                        ens_involuntary[jh] = 0.0
                        d_elastic_separated[jh] = shortfall
                    end
                end
            end
            
            push!(results["ENS_involuntary"][m], ens_involuntary)
            # Update D_elastic with the separated values
            push!(results["D_elastic"][m], d_elastic_separated)
        end
    else
        # Simplified error message without num_constraints
        error("""
        Optimization failed for agent: $m
        Status: $status
        Agent type: $(m in agents[:Gen] ? "Generator" : "Consumer")
        
        Current parameters:
        g_bar: $(mod.ext[:parameters][:g_bar][1:3])... (first 3 values)
        λ_EOM: $(mod.ext[:parameters][:λ_EOM][1:3])... (first 3 values)
        ρ_EOM: $(mod.ext[:parameters][:ρ_EOM])
        """)
    end
end

# Merge local TO with TO:
merge!(TO,TO_local)
end