function solve_generator_agent!(mod::Model, scen_number::Int, generator_type::String)
    # Extract sets
    JH = mod.ext[:sets][:JH]

    # Extract time series data
    AC = mod.ext[:timeseries][:AC]

    # Extract parameters
    A = mod.ext[:parameters][:A] 
    B = mod.ext[:parameters][:B]
    W = mod.ext[:parameters][:W]
    W_CVAR=  mod.ext[:parameters][:W]
    INV_h = mod.ext[:parameters][:INV_h] # investment cost
    λ_EOM = mod.ext[:parameters][:λ_EOM] # EOM prices
    g_bar = mod.ext[:parameters][:g_bar] # ADMM penalty term related to EOM
    ρ_EOM = mod.ext[:parameters][:ρ_EOM] # rho-value in ADMM related to EOM auctions
    C_cCM_bar = mod.ext[:parameters][:C_cCM_bar] # ADMM penalty term related to capacity market
    ρ_cCM = mod.ext[:parameters][:ρ_cCM] # rho-value in ADMM related to capacity market
    F_cCM = 1 # derating factor for capacity
    #F_cCM = mod.ext[:parameters][:F_cCM] # derating factor for capacity
    F_dCM = 1 # derating factor for capacity (similar to F_cCM)
    λ_cCM = mod.ext[:parameters][:λ_cCM] # cCM prices

    # Extract dCM parameters
    C_dCM_bar = mod.ext[:parameters][:C_dCM_bar] # ADMM penalty term related to dCM
    ρ_dCM = mod.ext[:parameters][:ρ_dCM] # rho-value in ADMM related to dCM
    λ_dCM = mod.ext[:parameters][:λ_dCM] # dCM prices

    # Extract variables
    g = mod.ext[:variables][:g]  
    C = mod.ext[:variables][:C]
    C_cCM = mod.ext[:variables][:C_cCM]
    C_dCM = mod.ext[:variables][:C_dCM]
    
    # Extract risk parameters
    β_gc = mod.ext[:parameters][:β_gc]
    γ_gc = mod.ext[:parameters][:γ_gc]

    # extract new variables for CVaR calculation
    α_gc = mod.ext[:variables][:α_gc] 
    u_gc = mod.ext[:variables][:u_gc] 

    # EOM with inelastic consumer Demand
    if scen_number == 1
        # Define profit expression
        profit = @expression(mod, [jh in JH],
            λ_EOM[jh] * g[jh] - (A/2 * g[jh]^2 + B * g[jh]) - INV_h * C
        )
        
        mod.ext[:objective] = @objective(mod, Min,
            -sum(W[jh] * profit[jh] for jh in JH)
            + sum(ρ_EOM/2 * (g[jh] - g_bar[jh])^2 for jh in JH)
        )

    # EOM with elastic consumer Demand
    elseif scen_number == 2
        # Define profit expression
        profit = @expression(mod, [jh in JH],
            λ_EOM[jh] * g[jh] - (A/2 * g[jh]^2 + B * g[jh]) - INV_h * C
        )
        
        mod.ext[:objective] = @objective(mod, Min,
            -sum(W[jh] * profit[jh] for jh in JH)
            + sum(ρ_EOM/2 * (g[jh] - g_bar[jh])^2 for jh in JH)
        )
  
                        

    # CVaR+ EOM + elastic consumer Demand
    elseif scen_number == 3 
        # Define profit expression
        profit = @expression(mod, [jh in JH],
            (λ_EOM[jh] * g[jh] - (A/2 * g[jh]^2 + B * g[jh]) - INV_h * C)
        )
        
        # Store profit expression
        mod.ext[:expressions][:profit] = profit
        
        mod.ext[:objective] = @objective(mod, Min,
            -γ_gc * sum(W[jh] * profit[jh] for jh in JH)
            + sum(ρ_EOM/2 * (g[jh] - g_bar[jh])^2 for jh in JH)
            - (1-γ_gc) * (α_gc - 1/β_gc * sum(W[jh] * u_gc[jh] for jh in JH))
        )

        # CVaR constraint
        for jh in JH
            delete(mod,mod.ext[:constraints][:CVAR_constr][jh])
        end
        mod.ext[:constraints][:CVAR_constr] = @constraint(mod, [jh in JH],
            u_gc[jh] >= -profit[jh] + α_gc
        )

    # EOM + Centralized Capacity Market (without CVaR)
    elseif scen_number == 4
        # Define profit expression including cCM revenue
        profit = @expression(mod, [jh in JH],
            λ_EOM[jh] * g[jh] - (A/2 * g[jh]^2 + B * g[jh]) - INV_h * C + λ_cCM * C_cCM
        )
        
        mod.ext[:objective] = @objective(mod, Min,
            -sum(W[jh] * profit[jh] for jh in JH)
            + sum(ρ_EOM/2 * (g[jh] - g_bar[jh])^2 for jh in JH)
            + ρ_cCM/2 * (C_cCM - C_cCM_bar)^2
        )

    # EOM + Centralized Capacity Market + CVaR
    elseif scen_number == 5
        # Define profit expression including cCM revenue
        profit = @expression(mod, [jh in JH],
            λ_EOM[jh] * g[jh] - (A/2 * g[jh]^2 + B * g[jh]) - INV_h * C + λ_cCM * C_cCM
        )
        
        # Store profit expression
        mod.ext[:expressions][:profit] = profit
        
        mod.ext[:objective] = @objective(mod, Min,
            -γ_gc * sum(W[jh] * profit[jh] for jh in JH)
            + sum(ρ_EOM/2 * (g[jh] - g_bar[jh])^2 for jh in JH)
            + ρ_cCM/2 * (C_cCM - C_cCM_bar)^2
            - (1-γ_gc) * (α_gc - 1/β_gc * sum(W[jh] * u_gc[jh] for jh in JH))
        )

        # CVaR constraint
        for jh in JH
            delete(mod,mod.ext[:constraints][:CVAR_constr][jh])
        end
        mod.ext[:constraints][:CVAR_constr] = @constraint(mod, [jh in JH],
            u_gc[jh] >= -profit[jh] + α_gc
        )

    # EOM + Decentralized Capacity Market with Reliability Options
    elseif scen_number == 6
        # Extract reliability option parameters
        λ_RO_strike = mod.ext[:parameters][:λ_EOM_scr]  # Strike price for reliability option
        
        # Define profit expression including reliability option revenues and compensation costs
        # Generators receive λ_dCM * C_dCM for selling reliability options
        # Generators pay max(λ_EOM - λ_RO_strike, 0) * C_dCM as compensation when energy price exceeds strike
        profit = @expression(mod, [jh in JH],
            λ_EOM[jh] * g[jh] - (A/2 * g[jh]^2 + B * g[jh]) - INV_h * C + λ_dCM * C_dCM
            - maximum([0, λ_EOM[jh] - λ_RO_strike]) * C_dCM)
        
        mod.ext[:objective] = @objective(mod, Min,
            -sum(W[jh] * profit[jh] for jh in JH)
            + sum(ρ_EOM/2 * (g[jh] - g_bar[jh])^2 for jh in JH)
            + ρ_dCM/2 * (C_dCM - C_dCM_bar)^2
        )

        # Capacity market availability constraint
        mod.ext[:constraints][:dCM_derate] = @constraint(mod,
            C_dCM <= F_dCM * C
        )

             
    # EOM + CVaR + Reliability Options
    elseif scen_number == 7
        # Extract reliability option parameters
        λ_RO_strike = mod.ext[:parameters][:λ_EOM_scr]  # Strike price for reliability option
        
        # Define profit expression including reliability option revenues and compensation costs
        # Generators receive λ_dCM * C_dCM for selling reliability options
        # Generators pay max(λ_EOM - λ_RO_strike, 0) * C_dCM as compensation when energy price exceeds strike
        profit = @expression(mod, [jh in JH],
            λ_EOM[jh] * g[jh] - (A/2 * g[jh]^2 + B * g[jh]) 
            - maximum([0, λ_EOM[jh] - λ_RO_strike]) * C_dCM
        )
        
        # Store profit expression
        mod.ext[:expressions][:profit] = profit
        
        mod.ext[:objective] = @objective(mod, Min,
            -(- INV_h * C + λ_dCM * C_dCM) -γ_gc * sum(W[jh] * profit[jh] for jh in JH)
            + sum(ρ_EOM/2 * (g[jh] - g_bar[jh])^2 for jh in JH)
            + ρ_dCM/2 * (C_dCM - C_dCM_bar)^2
            - (1-γ_gc) * (α_gc - 1/β_gc * sum(W[jh] * u_gc[jh] for jh in JH))
        )

        # CVaR constraint
        for jh in JH
            delete(mod,mod.ext[:constraints][:CVAR_constr][jh])
        end
        mod.ext[:constraints][:CVAR_constr] = @constraint(mod, [jh in JH],
            u_gc[jh] >= -profit[jh] + α_gc
        )

        # Capacity market availability constraint
        mod.ext[:constraints][:dCM_derate] = @constraint(mod,
            C_dCM <= F_dCM * C
        )
    end


    #= if scen_number == 3
        println("--- Generator Agent: Key Diagnostics ---")
        println("λ_EOM: [$(round(minimum(λ_EOM), digits=2)), $(round(maximum(λ_EOM), digits=2))] | ρ_EOM: $(round(ρ_EOM, digits=2))")
        println("Profit est: [$(round(-INV_h * maximum(g_bar), digits=2)), $(round(maximum(λ_EOM) * maximum(g_bar) - INV_h * maximum(g_bar), digits=2))]")
        println("VaR bounds: [$(has_lower_bound(α_gc) ? lower_bound(α_gc) : "-inf"), $(has_upper_bound(α_gc) ? upper_bound(α_gc) : "inf")]")
        println("Risk params: β=$(β_gc), γ=$(γ_gc)")
    end =#

    optimize!(mod)

    #= # Simplified post-optimization diagnostics
    if scen_number == 3 && termination_status(mod) == MOI.OPTIMAL
        println("VaR value: $(round(value(α_gc), digits=2)) | Status: $(termination_status(mod))")
    end =#

    return mod
end
