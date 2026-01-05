function diagnose_all_components(mod::Model, scen_number::Int)
    println("\n=== Condensed Model Diagnostics ===")
    
    # Most critical parameters to check
    if haskey(mod.ext[:variables], :α_gc)
        println("GEN VaR: $(round(value(mod.ext[:variables][:α_gc]), digits=2))")
    elseif haskey(mod.ext[:variables], :α_co)
        println("CONS VaR: $(round(value(mod.ext[:variables][:α_co]), digits=2))")
    end
    
    # Most important ranges - profit and prices
    if haskey(mod.ext, :expressions) && haskey(mod.ext[:expressions], :profit)
        profit_vals = value.(mod.ext[:expressions][:profit])
        println("Profit: [$(round(minimum(profit_vals), digits=2)), $(round(maximum(profit_vals), digits=2))]")
    elseif haskey(mod.ext, :expressions) && haskey(mod.ext[:expressions], :utility)
        utility_vals = value.(mod.ext[:expressions][:utility])
        println("Utility: [$(round(minimum(utility_vals), digits=2)), $(round(maximum(utility_vals), digits=2))]")
    end
    
    # Check ADMM penalty
    if haskey(mod.ext[:parameters], :ρ_EOM) && haskey(mod.ext[:variables], :g)
        g = mod.ext[:variables][:g]
        g_bar = mod.ext[:parameters][:g_bar]
        ρ_EOM = mod.ext[:parameters][:ρ_EOM]
        max_gap = maximum(abs.(value.(g) - g_bar))
        max_penalty = maximum(ρ_EOM/2 * (value.(g) - g_bar).^2)
        println("Max |g-g_bar|: $(round(max_gap, digits=2)) | Max penalty: $(round(max_penalty, digits=2))")
    end
    
    # Condition number (most important numerical health indicator)
    try
        backend = unsafe_backend(mod)
        if typeof(backend.inner) <: Gurobi.Optimizer
            println("Kappa: $(round(backend.inner.kappa, digits=2)) | Status: $(termination_status(mod))")
        end
    catch e
        println("Status: $(termination_status(mod))")
    end
end 