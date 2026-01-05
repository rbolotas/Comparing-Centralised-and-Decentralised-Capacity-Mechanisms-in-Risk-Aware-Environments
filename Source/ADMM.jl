# ADMM 
function ADMM!(results::Dict, ADMM::Dict, EOM::Dict, cCM::Dict, dCM::Dict, mdict::Dict, agents::Dict, scenario_overview_row::DataFrameRow, data::Dict, TO::TimerOutput)
    convergence = 0
    iterations = ProgressBar(1:data["ADMM"]["max_iter"])
    scen_number = scenario_overview_row["scen_number"]  # Extract scenario number



    # 1. deactivate decentralised CM in scenarios 4 and 5
    if scen_number == 4 || scen_number == 5
        agents[:dCM] = String[]          
        dCM["C_dCM"] = 0.0
    elseif scen_number == 6 || scen_number == 7
        agents[:cCM] = String[]          
        cCM["C_cCM"] = 0.0
    end

    for iter in iterations
        if convergence == 0
            # RETURN TO Multi-threaded version
            #for m in agents[:all] 
                # Created subroutine to allow multi-threading to solve agents' decision problems
               # ADMM_subroutine!(m, results, ADMM, EOM, cCM, dCM, mdict[m], agents, scen_number, TO)
            #end

            @sync for m in agents[:all] 
                # created subroutine to allow multi-treading to solve agents' decision problems
                @spawn ADMM_subroutine!(m, results, ADMM, EOM, cCM, dCM, mdict[m], agents, scen_number, TO)
            end

            # 2. compute market imbalances          
            @timeit TO "Compute imbalances" begin
                push!(ADMM["Imbalances"]["EOM"],
                      sum(results["g"][m][end] for m in agents[:eom]) - EOM["HV_LOAD"][:])

                push!(ADMM["Imbalances"]["cCM"],
                      sum((results["C_cCM"][m][end] for m in agents[:Gen]) ) - cCM["C_cCM"])

                # dCM imbalance: Generator supply - Total demand (active + passive)
                # Generators: C_dCM > 0 (capacity supplied)
                # Active consumers: C_dCM < 0 (via C_dCM = -D_dCM)
                # Passive load: C_dCM_inflexible > 0 (HV_LOAD, not an agent)
                # Equilibrium: sum(C_dCM[Gen]) + sum(C_dCM[Cons]) - C_dCM_inflexible = 0
                push!(ADMM["Imbalances"]["dCM"],
                      sum((results["C_dCM"][m][end] for m in agents[:Gen])) + sum((results["C_dCM"][m][end] for m in agents[:Cons])) - dCM["C_dCM_inflexible"])
                
            end

            # 3. zero capacity-market terms when they are not part of the scenario
            if scen_number ≤ 3
                ADMM["Imbalances"]["cCM"][end] = 0.0
                ADMM["Imbalances"]["dCM"][end] = 0.0
                ADMM["Residuals"]["Primal"]["cCM"][end] = 0.0
                ADMM["Residuals"]["Dual"]["cCM"][end]   = 0.0
                ADMM["Residuals"]["Primal"]["dCM"][end] = 0.0
                ADMM["Residuals"]["Dual"]["dCM"][end]   = 0.0
            elseif scen_number == 4 || scen_number == 5
                ADMM["Imbalances"]["dCM"][end] = 0.0
                ADMM["Residuals"]["Primal"]["dCM"][end] = 0.0
                ADMM["Residuals"]["Dual"]["dCM"][end]   = 0.0
            elseif scen_number == 6 || scen_number == 7
                ADMM["Imbalances"]["cCM"][end] = 0.0
                ADMM["Residuals"]["Primal"]["cCM"][end] = 0.0
                ADMM["Residuals"]["Dual"]["cCM"][end]   = 0.0
            end

            # Primal residuals 
            @timeit TO "Compute primal residuals" begin
                push!(ADMM["Residuals"]["Primal"]["EOM"], sqrt(sum(ADMM["Imbalances"]["EOM"][end].^2)))
                push!(ADMM["Residuals"]["Primal"]["cCM"], abs(ADMM["Imbalances"]["cCM"][end]))
                push!(ADMM["Residuals"]["Primal"]["dCM"], abs(ADMM["Imbalances"]["dCM"][end]))
            end

            # Dual residuals
            @timeit TO "Compute dual residuals" begin 
                if iter > 1
                    push!(ADMM["Residuals"]["Dual"]["EOM"], sqrt(sum(sum((ADMM["ρ"]["EOM"][end]*((results["g"][m][end]-sum(results["g"][mstar][end] for mstar in agents[:eom])./(EOM["nAgents"]+1)) - (results["g"][m][end-1]-sum(results["g"][mstar][end-1] for mstar in agents[:eom])./(EOM["nAgents"]+1)))).^2 for m in agents[:eom]))))
                    push!(ADMM["Residuals"]["Dual"]["cCM"], sqrt(sum((ADMM["ρ"]["cCM"][end] *
                                                                  (results["C_cCM"][m][end] - results["C_cCM"][m][end-1]))^2
                                                                 for m in agents[:cCM]; init = 0.0)))
                    push!(ADMM["Residuals"]["Dual"]["dCM"], sqrt(sum((ADMM["ρ"]["dCM"][end] *
                                                                  (results["C_dCM"][m][end] - results["C_dCM"][m][end-1]))^2
                                                                 for m in agents[:dCM]; init = 0.0)))
                end
            end

            # Price updates //add a relaxed update constant α? α * ADMM["ρ"]["EOM"][end] * ADMM["Imbalances"]["EOM"][end])
            @timeit TO "Update prices" begin
                    α_EOM = data["ADMM"]["α_EOM"] 
                    α_cCM = data["ADMM"]["α_cCM"] 
                    α_dCM = data["ADMM"]["α_dCM"] 
                    push!(results["λ"]["EOM"],
                          #minimum.([results["λ"]["EOM"][end] - α_EOM * ADMM["ρ"]["EOM"][end] .* ADMM["Imbalances"]["EOM"][end],0.069*ones(length(results["λ"]["EOM"][end]))])) #price cap

                          results["λ"]["EOM"][end] - α_EOM * ADMM["ρ"]["EOM"][end] .* ADMM["Imbalances"]["EOM"][end])
                          # for i in 1:length(results["λ"]["EOM"][end])
                          #     if results["λ"]["EOM"][end][i] > 0.1
                          #         results["λ"]["EOM"][end][i] = 0.1
                          #     end
                          # end
                    push!(results["λ"]["cCM"],
                          results["λ"]["cCM"][end] - α_cCM * ADMM["ρ"]["cCM"][end] * ADMM["Imbalances"]["cCM"][end])
    
                    push!(results["λ"]["dCM"],
                          results["λ"]["dCM"][end] - α_dCM * ADMM["ρ"]["dCM"][end] * ADMM["Imbalances"]["dCM"][end])

            end
               

            # Update ρ-values
            #@timeit TO "Update ρ" begin
               # update_rho!(ADMM, iter)
            #end

            # Progress bar with conditional display based on scenario
            @timeit TO "Progress bar" begin
                # Prepare progress message based on scenario
                if scen_number <= 3
                    progress_msg = @sprintf("EOM: %.3f/%.3f (ρ=%.2f)",
                                           ADMM["Residuals"]["Primal"]["EOM"][end],
                                           iter > 1 ? ADMM["Residuals"]["Dual"]["EOM"][end] : 0.0,
                                           ADMM["ρ"]["EOM"][end])
                elseif scen_number == 4 || scen_number == 5
                    progress_msg = @sprintf("EOM: %.3f/%.3f (ρ=%.2f)  cCM: %.3f/%.3f (ρ=%.3f)",
                                           ADMM["Residuals"]["Primal"]["EOM"][end],
                                           iter > 1 ? ADMM["Residuals"]["Dual"]["EOM"][end] : 0.0,
                                           ADMM["ρ"]["EOM"][end],
                                           ADMM["Residuals"]["Primal"]["cCM"][end],
                                           iter > 1 ? ADMM["Residuals"]["Dual"]["cCM"][end] : 0.0,
                                           ADMM["ρ"]["cCM"][end])
                elseif scen_number == 6 || scen_number == 7
                    progress_msg = @sprintf("EOM: %.3f/%.3f (ρ=%.2f) | dCM: %.3f/%.3f (ρ=%.3f)",
                                           ADMM["Residuals"]["Primal"]["EOM"][end],
                                           iter > 1 ? ADMM["Residuals"]["Dual"]["EOM"][end] : 0.0,
                                           ADMM["ρ"]["EOM"][end],
                                           ADMM["Residuals"]["Primal"]["dCM"][end],
                                           iter > 1 ? ADMM["Residuals"]["Dual"]["dCM"][end] : 0.0,
                                           ADMM["ρ"]["dCM"][end])
                end
                # Force clear and update
                set_description(iterations, progress_msg)
            end

            # Check convergence based on scenario number
            if scen_number in [1, 2, 3]
                if ADMM["Residuals"]["Primal"]["EOM"][end] <= ADMM["Tolerance"]["EOM"] && 
                   ADMM["Residuals"]["Dual"]["EOM"][end] <= ADMM["Tolerance"]["EOM"]
                    convergence = 1
                end
            elseif scen_number == 4 || scen_number == 5
                if ADMM["Residuals"]["Primal"]["cCM"][end] <= ADMM["Tolerance"]["cCM"] && 
                   ADMM["Residuals"]["Dual"]["cCM"][end] <= ADMM["Tolerance"]["cCM"] &&
                   ADMM["Residuals"]["Primal"]["EOM"][end] <= ADMM["Tolerance"]["EOM"] && 
                   ADMM["Residuals"]["Dual"]["EOM"][end] <= ADMM["Tolerance"]["EOM"]
                    convergence = 1
                end
            elseif scen_number == 6 || scen_number == 7
                if ADMM["Residuals"]["Primal"]["dCM"][end] <= ADMM["Tolerance"]["dCM"] && 
                   ADMM["Residuals"]["Dual"]["dCM"][end] <= ADMM["Tolerance"]["dCM"]  &&
                   ADMM["Residuals"]["Primal"]["EOM"][end] <= ADMM["Tolerance"]["EOM"] && 
                   ADMM["Residuals"]["Dual"]["EOM"][end] <= ADMM["Tolerance"]["EOM"]
                    convergence = 1
                end
            end

            # Store number of iterations
            ADMM["n_iter"] = copy(iter)
        end
    end
end
