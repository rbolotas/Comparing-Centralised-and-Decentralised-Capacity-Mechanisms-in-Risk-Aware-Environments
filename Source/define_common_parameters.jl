function define_common_parameters!(m::String, mod::Model, data::Dict, ts::DataFrame, agents::Dict, scenario_overview_row::DataFrameRow)
    # Solver settings
    # Define dictionaries for sets, parameters, timeseries, variables, constraints & expressions
    mod.ext[:sets] = Dict()
    mod.ext[:parameters] = Dict()
    mod.ext[:timeseries] = Dict()
    mod.ext[:variables] = Dict()
    mod.ext[:constraints] = Dict()
    mod.ext[:expressions] = Dict()

    # Sets
    mod.ext[:sets][:JH] = 1:data["General"]["nTimesteps"]
  
    # Parameters related to the EOM
    mod.ext[:parameters][:λ_EOM] = zeros(data["General"]["nTimesteps"])   # Price structure
    mod.ext[:parameters][:g_bar] = zeros(data["General"]["nTimesteps"])   # ADMM penalty term
    mod.ext[:parameters][:ρ_EOM] = data["ADMM"]["rho_EOM"]                # ADMM rho value
    mod.ext[:parameters][:W] = 1/data["General"]["nTimesteps"] .* ones(data["General"]["nTimesteps"]) # Weighting factor for each timestep

    # Parameters related to the cCM
    mod.ext[:parameters][:C_cCM_bar] = 0.0 # ADMM penalty term (scalar, capacity cleared once for whole year)
    mod.ext[:parameters][:λ_cCM] = 0.0 # cCM prices (scalar, not per timestep)
    mod.ext[:parameters][:ρ_cCM] = data["ADMM"]["rho_cCM"]                # ADMM rho value
   

    # Parameters related to the dCM
    mod.ext[:parameters][:C_dCM_bar] = 0.0 # ADMM penalty term (scalar, not per timestep)
    mod.ext[:parameters][:D_dCM_bar] = 0.0 # ADMM penalty term for consumer demand
    mod.ext[:parameters][:λ_dCM] = 0.0 # dCM prices (scalar, capacity cleared once for whole year)
    mod.ext[:parameters][:ρ_dCM] = data["ADMM"]["rho_dCM"]                # ADMM rho value 
  
    
    return mod, agents
end