function build_generator_agent!(mod::Model, scen_number::Int, m::String)
    # Extract sets
    JH = mod.ext[:sets][:JH]

    # Extract time series data
    AC = mod.ext[:timeseries][:AC]

    # Extract parameters
    A = mod.ext[:parameters][:A] 
    B = mod.ext[:parameters][:B]  
    λ_EOM = mod.ext[:parameters][:λ_EOM] # EOM prices
    g_bar = mod.ext[:parameters][:g_bar] # element in ADMM penalty term related to EOM
    ρ_EOM = mod.ext[:parameters][:ρ_EOM] # rho-value in ADMM related to EOM auctions
    
    λ_cCM = mod.ext[:parameters][:λ_cCM] # cCM prices
    C_cCM_bar = mod.ext[:parameters][:C_cCM_bar] # element in ADMM penalty term related to cCM
    ρ_cCM = mod.ext[:parameters][:ρ_cCM] # rho-value in ADMM related to cCM
    #F_cCM = 1 # derating factor for capacity
    F_cCM = mod.ext[:parameters][:F_cCM] # derating factor for capacity

    λ_dCM = mod.ext[:parameters][:λ_dCM] # dCM prices
    C_dCM_bar = mod.ext[:parameters][:C_dCM_bar] # element in ADMM penalty term related to dCM
    ρ_dCM = mod.ext[:parameters][:ρ_dCM] # rho-value in ADMM related to dCM

    β_gc = mod.ext[:parameters][:β_gc] #Genco CVAR confidence interval
    γ_gc = mod.ext[:parameters][:γ_gc]  #Genco CVAR confidence weight parameter

    # Create variables
    g = mod.ext[:variables][:g] = @variable(mod, [jh=JH], lower_bound=0, base_name="generation")
    C = mod.ext[:variables][:C] = @variable(mod, lower_bound=0, base_name="installed capacity")
    C_cCM = mod.ext[:variables][:C_cCM] = @variable(mod, lower_bound=0, base_name="cCM cap")
    C_dCM = mod.ext[:variables][:C_dCM] = @variable(mod, lower_bound=0, base_name="dCM cap")
    α_gc = mod.ext[:variables][:α_gc] = @variable(mod, base_name="Value-at-Risk")
    u_gc= mod.ext[:variables][:u_gc] = @variable(mod, [jh=JH],lower_bound=0, base_name="auxiliary variables for the CV@R calculation")

    # Wind capacity constraints: Fix at config values
    if m == "WindOnshore"
        # Remove existing bounds and set fixed capacity
        if has_lower_bound(C)
            delete_lower_bound(C)
        end
        if has_upper_bound(C)
            delete_upper_bound(C)
        end
        # Add fixed capacity constraint using config value
        wind_capacity = mod.ext[:parameters][:C_wind]  # Get capacity from config
        mod.ext[:constraints][:wind_onshore_fixed_capacity] = @constraint(mod, C == wind_capacity)
    elseif m == "WindOffshore"
        # Remove existing bounds and set fixed capacity
        if has_lower_bound(C)
            delete_lower_bound(C)
        end
        if has_upper_bound(C)
            delete_upper_bound(C)
        end
        # Add fixed capacity constraint using config value
        wind_capacity = mod.ext[:parameters][:C_wind]  # Get capacity from config
        mod.ext[:constraints][:wind_offshore_fixed_capacity] = @constraint(mod, C == wind_capacity)
    end

    if m == "WindOnshore" || m == "WindOffshore"
        # Wind generators use availability factor (they have fixed capacity + AC)
        mod.ext[:constraints][:cap_upper_bound] = @constraint(mod, [jh in JH],
            g[jh] <= C * AC[jh]
        )
    else
        # Dispatchable generators use full capacity (no AC needed)
        mod.ext[:constraints][:cap_upper_bound] = @constraint(mod, [jh in JH],
            g[jh] <= C
        )
    end

    if scen_number == 3 || scen_number == 5 || scen_number == 7
        mod.ext[:constraints][:CVAR_constr] = @constraint(mod, [jh in JH],
            u_gc[jh] >= 0  # Negative of profit in every scenario, because min problem
        )
    end

    if scen_number == 4 || scen_number == 5
        # Capacity market availability constraint for centralized CM
        mod.ext[:constraints][:cCM_derate] = @constraint(mod,
        C_cCM <= F_cCM * C
        )
    end
    
    if scen_number == 6 || scen_number == 7
        # Capacity market availability constraint for decentralized CM
        mod.ext[:constraints][:dCM_derate] = @constraint(mod,
        C_dCM <= F_cCM * C
        )
      
    end

    return mod
end
