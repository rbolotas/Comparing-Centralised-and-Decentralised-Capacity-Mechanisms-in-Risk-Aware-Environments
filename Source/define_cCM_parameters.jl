using CSV

function define_cCM_parameters!(cCM::Dict, data::Dict, ts::DataFrame, scenario_overview_row, scen_number)
   
    # Determine project root and candidate reference files
    home_dir = dirname(@__DIR__)
    candidate_paths = [
        joinpath(home_dir, "Results", "Scenario_2_ref.csv"),           # legacy reference
        joinpath(home_dir, "Results", "Scenario_3_beta_1.0.csv"),     # beta-sweep equivalent (β=1.0)
        joinpath(home_dir, "Results", "Scenario_3_beta_1.csv"),       # alternative naming
    ]
    scenario_ref_path = nothing
    for p in candidate_paths
        if isfile(p)
            scenario_ref_path = p
            break
        end
    end
    if scenario_ref_path === nothing
        error("Reference results not found. Expected one of: $(candidate_paths). " *
              "Please run Scenario 3 with β=1.0 (or restore Scenario_2_ref.csv).")
    end
    scenario2_data = CSV.read(scenario_ref_path, DataFrame, delim=';')
    
    # Calculate maximum total generation instance from Scenario 2
    # Max instance = maximum sum of dispatchable generators only (excluding wind)
    generation_columns = ["G_Baseload", "G_MidMerit", "G_Peak"]  # Only dispatchable units
    
    # Calculate total generation at each timestep
    timestep_totals = []
    for i in 1:nrow(scenario2_data)
        timestep_total = 0.0
        for gen_col in generation_columns
            if gen_col in names(scenario2_data)
                timestep_total += scenario2_data[i, gen_col]
            end
        end
        push!(timestep_totals, timestep_total)
    end
    
    # Find the maximum total generation instance
    total_installed_capacity = maximum(timestep_totals)
    
    margin = get(data["cCM"], "reserve_margin", 0.0)    # default 0 %
    cCM["C_cCM"] = total_installed_capacity * (1 + margin)
    
    if scen_number == 4 || scen_number == 5
        # Find the timestep where maximum occurred for detailed output
        max_timestep = argmax(timestep_totals)
        println("Maximum total dispatchable generation instance from reference (Scenario 2 or Scenario 3 at β=1.0): $(round(total_installed_capacity, digits=4)) GW")
        println("Occurred at timestep: $max_timestep")
        
        # Show breakdown at that timestep
        for gen_col in generation_columns
            if gen_col in names(scenario2_data)
                gen_value = scenario2_data[max_timestep, gen_col]
                println("  $(gen_col): $(round(gen_value, digits=3)) GW")
            end
        end
        
        println("Capacity Volume Target with $(round(margin*100, digits=1))% reserve margin: $(round(cCM["C_cCM"], digits=3)) GW")
    end
    
    return cCM
end 