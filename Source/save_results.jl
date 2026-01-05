using CSV
using DataFrames

# Save results
function save_results(
    mdict::Dict, 
    EOM::Dict, 
    cCM::Dict,
    dCM::Dict,
    ADMM::Dict, 
    results::Dict, 
    data::Dict, 
    agents::Dict, 
    scenario_overview_row::DataFrameRow, 
    sens::String
) 
    # Use the home directory from the calling context
    home_dir = dirname(@__DIR__)  # Go up one level from Source to main directory

    # ---------- 1. Build the output vector ---------------------------------
    # capacity volumes; zero when the market is not active
    C_cCM_tot = scenario_overview_row["scen_number"] == 4 || scenario_overview_row["scen_number"] == 5 ?
                  sum(results["C_cCM"][m][end] for m in agents[:Gen]) : 0.0
    C_dCM_tot = scenario_overview_row["scen_number"] == 6 || scenario_overview_row["scen_number"] == 7 ?
                  sum(results["C_dCM"][m][end] for m in agents[:Gen]) : 0.0

    # Simple convergence check: did it converge before hitting max iterations?
    converged = ADMM["n_iter"] < data["ADMM"]["max_iter"]

    vector_output = [
        scenario_overview_row["scen_number"],
        sens,
        ADMM["n_iter"],
        ADMM["walltime"],
        converged,
        ADMM["Residuals"]["Primal"]["EOM"][end],
        ADMM["Residuals"]["Dual"]["EOM"][end],
        ADMM["Residuals"]["Primal"]["cCM"][end],
        ADMM["Residuals"]["Dual"]["cCM"][end],
        ADMM["Residuals"]["Primal"]["dCM"][end],
        ADMM["Residuals"]["Dual"]["dCM"][end],
        results["λ"]["cCM"][end],
        results["λ"]["dCM"][end],
        sum(results["C"][m][end] for m in agents[:Gen]),   # total installed capacity
        C_cCM_tot,                                        # cleared cCM volume
        C_dCM_tot,                                        # cleared dCM volume
        data["RiskMetrics"]["β_co"],                     # beta_co
        data["RiskMetrics"]["β_gc"],                     # beta_gc
    ]

    cols = ["scen_number","sensitivity","n_iter","walltime",
            "converged","PrimalResidual_EOM","DualResidual_EOM",
            "PrimalResidual_cCM","DualResidual_cCM",
            "PrimalResidual_dCM","DualResidual_dCM",
            "λ_cCM","λ_dCM",
            "C_system_total","C_cCM_vol","C_dCM_vol",
            "beta_co","beta_gc"]

    df_row = DataFrame(Symbol.(cols) .=> Ref.(vector_output))

    overview_path = joinpath(home_dir, "overview_results.csv")

    if isfile(overview_path)
        CSV.write(overview_path, df_row; append=true, delim=';')
    else
        CSV.write(overview_path, df_row; delim=';')  # writes header
    end

    # Prepare scenario-specific CSV with g, ENS, and D for each consumer
    n_timesteps = data["General"]["nTimesteps"]
    n_agents = length(agents[:all])
    n_consumers = length(agents[:Cons])

    # Initialize matrices
    g_out = zeros(n_timesteps, n_agents)
    ens_out = zeros(n_timesteps, n_consumers)   # Now treated as time-series
    d_out = zeros(n_timesteps, n_consumers)
    d_elastic_out = zeros(n_timesteps, n_consumers)
    ens_involuntary_out = zeros(n_timesteps, n_consumers)

    # Populate g_out with generation data for all agents
    for (i, m) in enumerate(agents[:all])
        g_values = results["g"][m][end]
        if length(g_values) != n_timesteps
            @error "Mismatch in timesteps for agent $m in g: expected $n_timesteps, got $(length(g_values))"
            return
        end
        g_out[:, i] = g_values
    end

    # Populate ens_out with ENS values, assuming a vector per timestep
    # If your ENS is truly a single final value, revert here to your old approach
    for (i, m) in enumerate(agents[:Cons])
        ens_values = results["ENS"][m][end]
        if length(ens_values) != n_timesteps
            @error "Mismatch in timesteps for consumer $m in ENS: expected $n_timesteps, got $(length(ens_values))"
            ens_out[:, i] = fill(missing, n_timesteps)
        else
            ens_out[:, i] = ens_values
        end
    end

    # Populate d_out with D values per consumer
    for (i, m) in enumerate(agents[:Cons])
        d_values = results["D"][m][end]
        if length(d_values) != n_timesteps
            @error "Mismatch in timesteps for consumer $m in D: expected $n_timesteps, got $(length(d_values))"
            d_out[:, i] = fill(missing, n_timesteps)
        else
            d_out[:, i] = d_values
        end
    end

    # Populate d_elastic_out with separated D_elastic values per consumer
    for (i, m) in enumerate(agents[:Cons])
        d_elastic_values = results["D_elastic"][m][end]
        if length(d_elastic_values) != n_timesteps
            @error "Mismatch in timesteps for consumer $m in D_elastic: expected $n_timesteps, got $(length(d_elastic_values))"
            d_elastic_out[:, i] = fill(missing, n_timesteps)
        else
            d_elastic_out[:, i] = d_elastic_values
        end
    end

    # Populate ens_involuntary_out with separated involuntary ENS values per consumer
    for (i, m) in enumerate(agents[:Cons])
        ens_involuntary_values = results["ENS_involuntary"][m][end]
        if length(ens_involuntary_values) != n_timesteps
            @error "Mismatch in timesteps for consumer $m in ENS_involuntary: expected $n_timesteps, got $(length(ens_involuntary_values))"
            ens_involuntary_out[:, i] = fill(missing, n_timesteps)
        else
            ens_involuntary_out[:, i] = ens_involuntary_values
        end
    end

    # Debugging: Verify shapes
    #=println("g_out Shape: ", size(g_out))
    println("ens_out Shape: ", size(ens_out))
    println("d_out Shape: ", size(d_out))
    =#

    # Create the DataFrame for the results
    df = DataFrame(
        Timestep = 1:n_timesteps,
        Price = results["λ"]["EOM"][end],
        HV_LOAD = EOM["HV_LOAD"]  # External industrial Demand
    )
    # Add beta columns (repeat across timesteps)
    df[!, "beta_co"] = fill(data["RiskMetrics"]["β_co"], n_timesteps)
    df[!, "beta_gc"] = fill(data["RiskMetrics"]["β_gc"], n_timesteps)

    # Add input demand timeseries (per consumer and total)
    input_demand = zeros(n_timesteps, n_consumers)
    for (i, m) in enumerate(agents[:Cons])
        # This matches the logic in define_consumer_parameters!
        input_demand[:, i] = data["General"]["totConsumers"] * data["Consumers"][m]["Share"] .* ts[!, data["Consumers"][m]["D"]]
        df[!, "Input_D_$(m)"] = input_demand[:, i]
    end
    df[!, "Total_Input_Demand"] = vec(sum(input_demand, dims=2))

    # --- Add consumer demand curve parameters as columns ---
    for m in agents[:Cons]
        mod = mdict[m]
        # Price cap exists for all scenarios
        df[!, "lambda_EOM_cap_$(m)"] = fill(mod.ext[:parameters][:λ_EOM_cap], n_timesteps)
        
        if scenario_overview_row["scen_number"] >= 2
            # Full elastic demand parameters for scenarios 2+
            df[!, "E_EOM_$(m)"] = mod.ext[:parameters][:E_EOM]
            df[!, "lambda_EOM_0_$(m)"] = mod.ext[:parameters][:λ_EOM_0]
            df[!, "D_EOM_cap_$(m)"] = mod.ext[:parameters][:D_EOM_cap]
        else
            # For Scenario 1, only price cap exists, others are not defined
            df[!, "E_EOM_$(m)"] = fill(0.0, n_timesteps)
            df[!, "lambda_EOM_0_$(m)"] = fill(0.0, n_timesteps)
            df[!, "D_EOM_cap_$(m)"] = fill(0.0, n_timesteps)
        end
    end

    # Define generation order by merit order
    merit_order = ["WindOnshore", "WindOffshore", "Baseload", "MidMerit", "Peak"]

    # Add generator columns in merit order
    for gen_type in merit_order
        if gen_type in agents[:Gen]
            column_name = "G_$gen_type"
            df[!, column_name] = g_out[:, findfirst(isequal(gen_type), agents[:all])]
        end
    end

    # Add remaining generator columns (if any weren't in the merit_order list)
    for m in agents[:Gen]
        if !(m in merit_order)
            column_name = "G_$m"
            df[!, column_name] = g_out[:, findfirst(isequal(m), agents[:all])]
        end
    end

    # Add per-generator installed capacity columns (scalar repeated across timesteps)
    for m in agents[:Gen]
        df[!, "C_$(m)"] = fill(results["C"][m][end], n_timesteps)
    end

    # Add consumer columns
    for m in agents[:Cons]
        column_name = "G_$m"
        df[!, column_name] = g_out[:, findfirst(isequal(m), agents[:all])]
    end

    # Add ENS_ columns for each consumer
    for m in agents[:Cons]
        column_name = "ENS_$m"
        df[!, column_name] = ens_out[:, findfirst(isequal(m), agents[:Cons])]
    end

    # Add D_ columns for each consumer
    for m in agents[:Cons]
        column_name = "D_$m"
        df[!, column_name] = d_out[:, findfirst(isequal(m), agents[:Cons])]
    end
    
    # Add D_elastic_ columns for each consumer
    for m in agents[:Cons]
        column_name = "D_elastic_$m"
        df[!, column_name] = d_elastic_out[:, findfirst(isequal(m), agents[:Cons])]
    end
    
    # Add ENS_involuntary_ columns for each consumer
    for m in agents[:Cons]
        column_name = "ENS_involuntary_$m"
        df[!, column_name] = ens_involuntary_out[:, findfirst(isequal(m), agents[:Cons])]
    end
    
    # Add PV generation columns for each consumer (actual solar generation)
    for m in agents[:Cons]
        column_name = "PV_$m"
        # Get the actual PV generation from the consumer agent results
        if haskey(results, "PV_curt") && haskey(results["PV_curt"], m)
            pv_values = results["PV_curt"][m][end]
            if length(pv_values) == n_timesteps
                df[!, column_name] = pv_values
            else
                @error "PV_curt data for $m has wrong length: $(length(pv_values)) vs $n_timesteps"
                df[!, column_name] = fill(0.0, n_timesteps)
            end
        else
            @error "No PV_curt data found for consumer $m"
            df[!, column_name] = fill(0.0, n_timesteps)
        end
    end
    
    # Add total PV generation
    total_pv = zeros(n_timesteps)
    for m in agents[:Cons]
        column_name = "PV_$m"
        if column_name in names(df)
            total_pv .+= df[!, column_name]
        end
    end
    df[!, "PV_Total"] = total_pv
    
    # Add total ENS and total demand columns
    df[!, "Total_ENS"] = vec(sum(ens_out, dims=2))
    df[!, "Consumer_Demand"] = vec(sum(d_out, dims=2))
    df[!, "Total_Demand"] = vec(sum(d_out, dims=2)) + EOM["HV_LOAD"]  # Include industrial demand

    if scenario_overview_row["scen_number"] == 4 || scenario_overview_row["scen_number"] == 5
        # Capacity-market results: clearing prices and awarded volumes
        df[!, "λ_cCM"] = fill(results["λ"]["cCM"][end], n_timesteps)

        # System totals (repeat in every row for convenience)
        df[!, "C_system_total"] = fill(sum(results["C"][m][end] for m in agents[:Gen]), n_timesteps)
        df[!, "C_cCM_vol"] = fill(C_cCM_tot, n_timesteps)

        # Individual generator awards (scalar, repeated over time-steps)
        for m in agents[:Gen]
            df[!, "C_cCM_$m"] = fill(results["C_cCM"][m][end], n_timesteps)
        end
    
    elseif scenario_overview_row["scen_number"] == 6 || scenario_overview_row["scen_number"] == 7
        # Add dCM related fields
        df[!, "λ_dCM"] = fill(results["λ"]["dCM"][end], n_timesteps)

        # System totals
        df[!, "C_system_total"] = fill(sum(results["C"][m][end] for m in agents[:Gen]), n_timesteps)
        df[!, "C_dCM_vol"] = fill(C_dCM_tot, n_timesteps)

        # Individual generator awards
        for m in agents[:Gen]
            df[!, "C_dCM_$m"] = fill(results["C_dCM"][m][end], n_timesteps)
        end

        #individual consumer procurement
        for m in agents[:Cons]
            df[!, "C_dCM_$m"] = fill(results["C_dCM"][m][end], n_timesteps)
        end
    end

    # debug cap price
    if scenario_overview_row["scen_number"] == 4 || scenario_overview_row["scen_number"] == 5
        @info "λ_cCM = $(results["λ"]["cCM"][end])"
        @info "C_cCM total = $C_cCM_tot"
        for m in agents[:Gen]
            @info "C_cCM_$m = $(results["C_cCM"][m][end])"
        end
       
    elseif scenario_overview_row["scen_number"] == 6 || scenario_overview_row["scen_number"] == 7
        @info "λ_dCM = $(results["λ"]["dCM"][end])"
        @info "C_dCM total = $C_dCM_tot"
        for m in agents[:Gen]
            @info "C_dCM_$m = $(results["C_dCM"][m][end])"
        end
        for m in agents[:Cons]
            @info "C_dCM_$m = $(results["C_dCM"][m][end])"
        end
    
    end

    # Round all numeric columns to 5 decimal places
    for col in names(df)
        if eltype(df[!, col]) <: Number
            df[!, col] = round.(df[!, col], digits=5)
        end
    end
    

    # Write to CSV
    csv_path = joinpath(home_dir, "Results", "Scenario_$(scenario_overview_row["scen_number"])_$(sens).csv")
    CSV.write(csv_path, df, delim=";")
    println("Results saved to $csv_path")

    # --- Export consumer demand curve parameters for Python analysis ---
    # Remove the separate export of ConsumerParams_Scenario_X.csv
end