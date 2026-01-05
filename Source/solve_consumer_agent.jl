function solve_consumer_agent!(mod::Model, scen_number::Int)
    # Extract sets
    JH = mod.ext[:sets][:JH]

    # Extract parameters
    λ_EOM = mod.ext[:parameters][:λ_EOM]           # EOM prices
    g_bar = mod.ext[:parameters][:g_bar]           # element in ADMM penalty term related to EOM
    ρ_EOM = mod.ext[:parameters][:ρ_EOM]           # rho-value in ADMM related to EOM auctions
    W = mod.ext[:parameters][:W]                    # Weight of each timestep
    D_ts = mod.ext[:timeseries][:D_ts] 
    E_EOM = mod.ext[:parameters][:E_EOM]           # Inverse price-elasticity of energy demand
    λ_EOM_0 = mod.ext[:parameters][:λ_EOM_0]       # y_intercept price for inverse demand curve
    D_EOM_ref = mod.ext[:parameters][:D_EOM_ref]   # reference energy demand
    PV = mod.ext[:timeseries][:PV]                 # PV generation
    λ_EOM_cap = mod.ext[:parameters][:λ_EOM_cap]   # personal price cap for EOM
    D_EOM_cap = mod.ext[:parameters][:D_EOM_cap]   # personal demand cap of EOM

    # Add cCM parameters
    ρ_cCM = mod.ext[:parameters][:ρ_cCM]           # rho-value in ADMM related to cCM
    C_cCM = mod.ext[:variables][:C_cCM]            # centralized capacity
    C_cCM_bar = mod.ext[:parameters][:C_cCM_bar]   # ADMM penalty term related to cCM

    # Add dCM parameters
    λ_dCM = mod.ext[:parameters][:λ_dCM]
    C_dCM = mod.ext[:variables][:C_dCM]
    C_dCM_bar = mod.ext[:parameters][:C_dCM_bar]
    D_dCM_bar = mod.ext[:parameters][:D_dCM_bar]  # ADMM target for consumer demand
    ρ_dCM = mod.ext[:parameters][:ρ_dCM]
    λ_EOM_scr = mod.ext[:parameters][:λ_EOM_scr]   # scarcity price limit above which unprocured capacity is curtailed


    # Extract risk parameters
    β_co = mod.ext[:parameters][:β_co]
    γ_co = mod.ext[:parameters][:γ_co]

    # extract new variables for CVaR calculation
    α_co = mod.ext[:variables][:α_co] 
    u_co = mod.ext[:variables][:u_co] 

    # Create variables
    g = mod.ext[:variables][:g]                     # net interaction with system
    ENS = mod.ext[:variables][:ENS]                 # energy not served
    D = mod.ext[:variables][:D]                     # energy demand
    PV_curt = mod.ext[:variables][:PV_curt]        # PV curtailable demand
    D_dCM = mod.ext[:variables][:D_dCM]            # decentr capacity demand
    #D_cCM = mod.ext[:variables][:D_cCM]            # centr capacity demand
    RO_comp = mod.ext[:variables][:RO_comp]        # reliability option compensation
   
    # Define objectives and constraints based on scenario number
    if scen_number == 1
        # Define consumer surplus expression
        surplus = @expression(mod, [jh in JH],
            (λ_EOM_cap - λ_EOM[jh]) * D[jh] + λ_EOM[jh] * PV_curt[jh]
        )
        
        mod.ext[:objective] = @objective(mod, Min,
            -sum(W[jh] * surplus[jh] for jh in JH)
            + sum(ρ_EOM/2 * (g[jh] - g_bar[jh])^2 for jh in JH)
        )

    elseif scen_number == 2
        # Define consumer surplus expression with elastic demand
        surplus = @expression(mod, [jh in JH],
            0.5 * (λ_EOM_cap - λ_EOM[jh]) * (D[jh] + D_EOM_cap[jh]) + λ_EOM[jh] * PV_curt[jh] 
        )
        
        mod.ext[:objective] = @objective(mod, Min,
            -sum(W[jh] * surplus[jh] for jh in JH)
            + sum(ρ_EOM/2 * (g[jh] - g_bar[jh])^2 for jh in JH)
        )

        for jh in JH
            delete(mod,mod.ext[:constraints][:energy_elastic_demand][jh])
        end
        # Define the energy elastic demand constraint
        mod.ext[:constraints][:energy_elastic_demand] = @constraint(mod, [jh in JH],
            D[jh] + ENS[jh] == maximum([0,(λ_EOM[jh] - λ_EOM_0[jh]) / E_EOM[jh]]) 
        )

    elseif scen_number == 3   # Scenario 3: CVaR + EOM + Elastic Consumer Demand
        # Define consumer surplus expression with elastic demand
        utility = @expression(mod, [jh in JH],
            0.5 * (λ_EOM_cap - λ_EOM[jh]) * (D[jh] + D_EOM_cap[jh]) + λ_EOM[jh] * PV_curt[jh]
        )
        
        # Store surplus expression
        mod.ext[:expressions][:utility] = utility
        
        mod.ext[:objective] = @objective(mod, Min,
            -γ_co * sum(W[jh] * utility[jh] for jh in JH)
            + sum(ρ_EOM/2 * (g[jh] - g_bar[jh])^2 for jh in JH)
            - (1-γ_co) * (α_co - 1/β_co * sum(W[jh] * u_co[jh] for jh in JH))
        )

        # Delete and redefine constraints
        for jh in JH
            delete(mod, mod.ext[:constraints][:energy_elastic_demand][jh])
            delete(mod, mod.ext[:constraints][:CVAR_constr][jh])
        end

        # Energy elastic demand with safeguard against negative demand
        mod.ext[:constraints][:energy_elastic_demand] = @constraint(mod, [jh in JH],
            D[jh] + ENS[jh] == maximum([0, (λ_EOM[jh] - λ_EOM_0[jh]) / E_EOM[jh]])
        )

        # CVaR constraint
        mod.ext[:constraints][:CVAR_constr] = @constraint(mod, [jh in JH],
            u_co[jh] >= -utility[jh] + α_co
        )

       
    elseif scen_number == 4 # cCM without CVaR
        # Define consumer surplus expression including cCM costs
        surplus = @expression(mod, [jh in JH],
            0.5 * (λ_EOM_cap - λ_EOM[jh]) * (D[jh] + D_EOM_cap[jh]) + λ_EOM[jh] * PV_curt[jh]
        )
        
        mod.ext[:objective] = @objective(mod, Min,
            -sum(W[jh] * surplus[jh] for jh in JH)
            + sum(ρ_EOM/2 * (g[jh] - g_bar[jh])^2 for jh in JH)
            + ρ_cCM/2 * (C_cCM - C_cCM_bar)^2
        )

        for jh in JH
            delete(mod,mod.ext[:constraints][:energy_elastic_demand][jh])
        end
        # Define the energy elastic demand constraint
        mod.ext[:constraints][:energy_elastic_demand] = @constraint(mod, [jh in JH],
            D[jh] + ENS[jh] == maximum([0, (λ_EOM[jh] - λ_EOM_0[jh]) / E_EOM[jh]])
        )
    
    elseif scen_number == 5 # Scenario 5: CVaR + EOM + cCM
        # Define consumer surplus expression with elastic demand
        utility = @expression(mod, [jh in JH],
            0.5 * (λ_EOM_cap - λ_EOM[jh]) * (D[jh] + D_EOM_cap[jh]) + λ_EOM[jh] * PV_curt[jh]
        )
        
        # Store surplus expression
        mod.ext[:expressions][:utility] = utility
        
        mod.ext[:objective] = @objective(mod, Min,
            -γ_co * sum(W[jh] * utility[jh] for jh in JH)
            + sum(ρ_EOM/2 * (g[jh] - g_bar[jh])^2 for jh in JH)
            + ρ_cCM/2 * (C_cCM - C_cCM_bar)^2
            - (1-γ_co) * (α_co - 1/β_co * sum(W[jh] * u_co[jh] for jh in JH))
        )

        # Delete and redefine constraints
        for jh in JH
            delete(mod, mod.ext[:constraints][:energy_elastic_demand][jh])
            delete(mod, mod.ext[:constraints][:CVAR_constr][jh])
        end

        # Energy elastic demand with safeguard against negative demand
        mod.ext[:constraints][:energy_elastic_demand] = @constraint(mod, [jh in JH],
            D[jh] + ENS[jh] == maximum([0, (λ_EOM[jh] - λ_EOM_0[jh]) / E_EOM[jh]])
        )

        # CVaR constraint
        mod.ext[:constraints][:CVAR_constr] = @constraint(mod, [jh in JH],
            u_co[jh] >= -utility[jh] + α_co
        )
                     
    elseif scen_number == 6 # Scenario 6: EOM + Reliability Options (without CVaR)
        # Extract reliability option parameters
        λ_RO_strike = λ_EOM_scr  # Strike price for reliability option
        

        # Define consumer surplus expressions
        surplus = @expression(mod, [jh in JH],
            0.5 * (λ_EOM_cap - λ_EOM[jh]) * (D[jh] + D_EOM_cap[jh]) + λ_EOM[jh] * PV_curt[jh]
        )
        
        # Consumer maximizes: surplus + RO_comp - premium_paid
        # So minimize the negative: -(surplus + RO_comp - λ_dCM * D_dCM)
        
        mod.ext[:objective] = @objective(mod, Min,
            -(sum(W[jh] * (surplus[jh] + RO_comp[jh]) for jh in JH) - λ_dCM * D_dCM)
            + sum(ρ_EOM/2 * (g[jh] - g_bar[jh])^2 for jh in JH)
            + ρ_dCM/2 * (C_dCM - C_dCM_bar)^2
        )

        # Delete and redefine constraints with proper bounds
        for jh in JH
            delete(mod, mod.ext[:constraints][:energy_elastic_demand][jh])
            delete(mod, mod.ext[:constraints][:dCM_demand][jh])
            delete(mod, mod.ext[:constraints][:reliability_option1][jh])
            delete(mod, mod.ext[:constraints][:reliability_option2][jh])
        end

        # Energy elastic demand with safeguard against negative demand
        mod.ext[:constraints][:energy_elastic_demand] = @constraint(mod, [jh in JH],
            D[jh] + ENS[jh] == maximum([0, (λ_EOM[jh] - λ_EOM_0[jh]) / E_EOM[jh]])
        )
      
        # Simple dCM demand constraint (removed scarcity price logic)
        mod.ext[:constraints][:dCM_demand] = @constraint(mod, [jh in JH],
        D_dCM >= 0
        )

        #RO_comp[jh] <= max(0, λ_EOM[jh] - λ_RO_strike) * min(D_dCM, D[jh] + D_EOM_cap[jh])

        # Reliability option constraint
        mod.ext[:constraints][:reliability_option1] = @constraint(mod, [jh in JH],
        RO_comp[jh] <= maximum([0, λ_EOM[jh] - λ_RO_strike]) *D_dCM
        )

         # Reliability option constraint
         mod.ext[:constraints][:reliability_option2] = @constraint(mod, [jh in JH],
         RO_comp[jh] <= maximum([0, λ_EOM[jh] - λ_RO_strike]) *D[jh]
         )
                
        
    elseif scen_number == 7 # Scenario 7: CVaR + EOM + Reliability Options
        # Extract reliability option parameters
        λ_RO_strike = λ_EOM_scr  # Strike price for reliability option
        
        # Define total consumer benefit including reliability options
        utility = @expression(mod, [jh in JH],
            0.5 * (λ_EOM_cap - λ_EOM[jh]) * (D[jh] + D_EOM_cap[jh]) + λ_EOM[jh] * PV_curt[jh] + RO_comp[jh]
        )
        
        # Store utility expression (includes RO_comp already)
        mod.ext[:expressions][:utility] = utility
        
        # Consumer minimizes: -(γ×utility) - (1-γ)×CVaR + premium_paid
        mod.ext[:objective] = @objective(mod, Min,
            λ_dCM * D_dCM - γ_co * sum(W[jh] * utility[jh] for jh in JH)
            + sum(ρ_EOM/2 * (g[jh] - g_bar[jh])^2 for jh in JH)
            + ρ_dCM/2 * (C_dCM - C_dCM_bar)^2
            - (1-γ_co) * (α_co - 1/β_co * sum(W[jh] * u_co[jh] for jh in JH))
        )

        # Delete and redefine constraints
        for jh in JH
            delete(mod, mod.ext[:constraints][:energy_elastic_demand][jh])
            delete(mod, mod.ext[:constraints][:CVAR_constr][jh])
            delete(mod, mod.ext[:constraints][:dCM_demand][jh])
            delete(mod, mod.ext[:constraints][:reliability_option1][jh])
            #delete(mod, mod.ext[:constraints][:reliability_option2][jh])
        end

        # Energy elastic demand with safeguard against negative demand
        mod.ext[:constraints][:energy_elastic_demand] = @constraint(mod, [jh in JH],
            D[jh] + ENS[jh] == maximum([0, (λ_EOM[jh] - λ_EOM_0[jh]) / E_EOM[jh]])
        )

        # CVaR constraint - fixed the weighting issue
        mod.ext[:constraints][:CVAR_constr] = @constraint(mod, [jh in JH],
            u_co[jh] >= -utility[jh] + α_co
        )
        
        # dCM demand constraint 
        mod.ext[:constraints][:dCM_demand] = @constraint(mod, [jh in JH],
            D_dCM >= 0
        )

        # Reliability option constraints
        mod.ext[:constraints][:reliability_option1] = @constraint(mod, [jh in JH],
            RO_comp[jh] <=  maximum([0, λ_EOM[jh] - λ_RO_strike]) *D_dCM
        )

        #mod.ext[:constraints][:reliability_option2] = @constraint(mod, [jh in JH],
        #    RO_comp[jh] <=  maximum([0, λ_EOM[jh] - λ_RO_strike]) *(D[jh] + D_EOM_cap[jh])
        #)
  
    end

    #=if scen_number == 3
        println("--- Consumer Agent: Key Diagnostics ---")
        println("λ_EOM: [$(round(minimum(λ_EOM), digits=2)), $(round(maximum(λ_EOM), digits=2))] | ρ_EOM: $(round(ρ_EOM, digits=2))")
        println("VaR bounds: [$(has_lower_bound(α_co) ? lower_bound(α_co) : "-inf"), $(has_upper_bound(α_co) ? upper_bound(α_co) : "inf")]")
        println("Risk params: β=$(β_co), γ=$(γ_co)")
    end =#
    
    optimize!(mod)

    #= # Simplified post-optimization diagnostics
    if scen_number == 3 && termination_status(mod) == MOI.OPTIMAL
        println("VaR value: $(round(value(α_co), digits=2)) | Status: $(termination_status(mod))")
    end =#
      
    return mod
end
