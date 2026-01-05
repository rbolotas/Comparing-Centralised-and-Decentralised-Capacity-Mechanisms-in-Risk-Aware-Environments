function build_consumer_agent!(mod::Model, scen_number::Int)
    # Extract sets
    JH = mod.ext[:sets][:JH]

    # Extract time series data
    #D = mod.ext[:timeseries][:D] 
    PV = mod.ext[:timeseries][:PV] 
    D_ts = mod.ext[:timeseries][:D_ts]
    
    # Extract parameters
    λ_EOM = mod.ext[:parameters][:λ_EOM] # EOM prices
    g_bar = mod.ext[:parameters][:g_bar] # element in ADMM penalty term related to EOM
    ρ_EOM = mod.ext[:parameters][:ρ_EOM] # rho-value in ADMM related to EOM auctions
    
    # Debug: Print λ_EOM_cap value
    λ_EOM_cap = mod.ext[:parameters][:λ_EOM_cap]
    

    λ_cCM = mod.ext[:parameters][:λ_cCM] # cCM prices
    C_cCM_bar = mod.ext[:parameters][:C_cCM_bar] # element in ADMM penalty term related to cCM
    ρ_cCM = mod.ext[:parameters][:ρ_cCM] # rho-value in ADMM related to cCM

    λ_dCM = mod.ext[:parameters][:λ_dCM] # dCM prices
    C_dCM_bar = mod.ext[:parameters][:C_dCM_bar] # element in ADMM penalty term related to dCM
    D_dCM_bar = mod.ext[:parameters][:D_dCM_bar] # element in ADMM penalty term for consumer demand
    ρ_dCM = mod.ext[:parameters][:ρ_dCM] # rho-value in ADMM related to dCM



    # Create variables
    g = mod.ext[:variables][:g] = @variable(mod, [jh=JH], base_name="net interaction with system")
    D = mod.ext[:variables][:D] = @variable(mod, [jh=JH], lower_bound=0, base_name="energy demand")
    ENS = mod.ext[:variables][:ENS] = @variable(mod, [jh=JH], lower_bound=0, base_name="energy not served")
    D_elastic = mod.ext[:variables][:D_elastic] = @variable(mod, [jh=JH], lower_bound=0, base_name="elastic_demand_response")
    PV_curt = mod.ext[:variables][:PV_curt] = @variable(mod, [jh=JH], lower_bound=0, base_name="PV curtailment")
    C_cCM = mod.ext[:variables][:C_cCM] = @variable(mod, upper_bound=0, base_name="cCM cap")
    C_dCM = mod.ext[:variables][:C_dCM] = @variable(mod, upper_bound=0, base_name="dCM cap")
    D_cCM = mod.ext[:variables][:D_cCM] = @variable(mod, lower_bound=0, base_name="cCM demand")
    D_dCM = mod.ext[:variables][:D_dCM] = @variable(mod, lower_bound=0, base_name="dCM demand")
    RO_comp = mod.ext[:variables][:RO_comp] = @variable(mod, [jh=JH], lower_bound=0, base_name="reliability option compensation")

    
    # Add CVaR variables
    α_co = mod.ext[:variables][:α_co] = @variable(mod, base_name="consumer_value_at_risk")
    u_co = mod.ext[:variables][:u_co] = @variable(mod, [jh=JH],lower_bound=0, base_name="consumer_cvar_auxiliary")

    # Objective 
    mod.ext[:objective] = @objective(mod, Min,0)

    mod.ext[:constraints][:energy_balance] = @constraint(mod, [jh in JH],
            g[jh] == PV_curt[jh] - D[jh]
        )

    # PV Curtailment
    mod.ext[:constraints][:PV_curtailment] = @constraint(mod, [jh in JH],
        0 <= PV_curt[jh] <= PV[jh]
    )

    if scen_number == 1
         # Define the energy inelastic demand constraint
        mod.ext[:constraints][:energy_inelastic_demand] = @constraint(mod, [jh in JH],
            D[jh] + ENS[jh] == D_ts[jh] #constraint relaxation (<=) gives ADMM "wiggle room" to converge without getting stuck at discontinuities around λ=λ_EOM_cap, but makes ENS zero as it absorbes the slack.
        )
    end
    if scen_number == 2
        # Define the energy elastic demand constraint
        mod.ext[:constraints][:energy_elastic_demand] = @constraint(mod, [jh in JH],
            D[jh] + ENS[jh] <= 0
        )
    end
    if scen_number == 3    
        mod.ext[:constraints][:CVAR_constr] = @constraint(mod, [jh in JH],
            u_co[jh] >= 0  # Dummy constraint to be updated in solve_consumer_agent.jl
        )
        # Define the energy elastic demand constraint
        mod.ext[:constraints][:energy_elastic_demand] = @constraint(mod, [jh in JH],
            D[jh] + ENS[jh] <= 1e6
        )


    elseif scen_number == 4
        # Define the energy elastic demand constraint
        mod.ext[:constraints][:energy_elastic_demand] = @constraint(mod, [jh in JH],
            D[jh] + ENS[jh] <= 1e6
        )  
    elseif scen_number == 5
        # Add CVaR constraint for scenario 5
        mod.ext[:constraints][:CVAR_constr] = @constraint(mod, [jh in JH],
            u_co[jh] >= 0  # Dummy constraint to be updated in solve_consumer_agent.jl
        )
        # Define the energy elastic demand constraint
        mod.ext[:constraints][:energy_elastic_demand] = @constraint(mod, [jh in JH],
            D[jh] + ENS[jh] <= 1e6
        )
    elseif scen_number == 6
        # Initialize with non-binding constraints that will be replaced
        mod.ext[:constraints][:energy_elastic_demand] = @constraint(mod, [jh in JH],
            D[jh] + ENS[jh] <= 1e6  # Large upper bound
        )
        
        # Initialize dCM demand with non-binding constraint
        mod.ext[:constraints][:dCM_demand] = @constraint(mod, [jh in JH],
            D_dCM >= 0  # Simple non-negative constraint
        )
        mod.ext[:constraints][:dCM_balance] = @constraint(mod,
           C_dCM == -D_dCM  # Demand expressed as a negative value
        )
        mod.ext[:constraints][:reliability_option1] = @constraint(mod, [jh in JH],
        RO_comp[jh] <= 0
        )
        mod.ext[:constraints][:reliability_option2] = @constraint(mod, [jh in JH],
        RO_comp[jh] <= 0
        )
    elseif scen_number == 7
        # Initialize with non-binding constraints that will be replaced
        mod.ext[:constraints][:energy_elastic_demand] = @constraint(mod, [jh in JH],
            D[jh] + ENS[jh] <= 1e6  # Large upper bound
        )
        mod.ext[:constraints][:CVAR_constr] = @constraint(mod, [jh in JH],
            u_co[jh] >= 0  # Dummy constraint to be updated in solve_consumer_agent.jl
        )
        mod.ext[:constraints][:dCM_demand] = @constraint(mod, [jh in JH],
            D_dCM >= 0  # Simple non-negative constraint
        )
        mod.ext[:constraints][:dCM_balance] = @constraint(mod,
            C_dCM == -D_dCM  # Demand expressed as a negative value
        )
        # Add reliability option constraints initialization
        mod.ext[:constraints][:reliability_option1] = @constraint(mod, [jh in JH],
            RO_comp[jh] <= 0
        )
        #mod.ext[:constraints][:reliability_option2] = @constraint(mod, [jh in JH],
        #    RO_comp[jh] <= 0
        #)
    end 
    
    return mod
end
