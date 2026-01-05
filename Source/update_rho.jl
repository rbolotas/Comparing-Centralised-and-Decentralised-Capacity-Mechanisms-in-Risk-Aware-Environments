function update_rho!(ADMM::Dict, iter::Int64)
    if mod(iter,1) == 0
        # ρ-updates following Boyd et al. (2011) (make more responsive 2->1.2, 1.5->1.2)
        if ADMM["Residuals"]["Primal"]["EOM"][end] > 2*ADMM["Residuals"]["Dual"]["EOM"][end]
            push!(ADMM["ρ"]["EOM"], minimum([1000,1.5*ADMM["ρ"]["EOM"][end]]))
        elseif ADMM["Residuals"]["Dual"]["EOM"][end] > 2*ADMM["Residuals"]["Primal"]["EOM"][end]
            push!(ADMM["ρ"]["EOM"], 1/1.5*ADMM["ρ"]["EOM"][end])
        end
        # Update for cCM
        if ADMM["Residuals"]["Primal"]["cCM"][end] > 2 * ADMM["Residuals"]["Dual"]["cCM"][end]
            push!(ADMM["ρ"]["cCM"], minimum([1000, 1.5 * ADMM["ρ"]["cCM"][end]]))
        elseif ADMM["Residuals"]["Dual"]["cCM"][end] > 2 * ADMM["Residuals"]["Primal"]["cCM"][end]
            push!(ADMM["ρ"]["cCM"], 1 / 1.5 * ADMM["ρ"]["cCM"][end])
        end

        # Update for dCM
        if ADMM["Residuals"]["Primal"]["dCM"][end] > 2 * ADMM["Residuals"]["Dual"]["dCM"][end]
            push!(ADMM["ρ"]["dCM"], minimum([1000, 1.5 * ADMM["ρ"]["dCM"][end]]))
        elseif ADMM["Residuals"]["Dual"]["dCM"][end] > 2 * ADMM["Residuals"]["Primal"]["dCM"][end]
            push!(ADMM["ρ"]["dCM"], 1 / 1.5 * ADMM["ρ"]["dCM"][end])
        end
    end
end