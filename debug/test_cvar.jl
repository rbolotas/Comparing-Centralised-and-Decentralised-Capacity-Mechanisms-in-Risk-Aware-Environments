using JuMP
using Gurobi
using Test
using Statistics  # Add Statistics package for mean function

function test_simple_cvar()
    # Create a simple test case with known outcomes
    profits = [-10.0, 0.0, 10.0, 20.0]  # Simple profit scenarios
    probabilities = [0.25, 0.25, 0.25, 0.25]  # Equal probabilities
    β = 0.1  # Confidence level (same as in config)
    γ = 0.5  # Weight between expectation and CVaR (same as in config)
    
    # Create test model
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    
    # Variables
    @variable(model, α)  # VaR
    @variable(model, u[1:length(profits)] >= 0)  # CVaR auxiliary variables
    
    # Objective: minimize -γ*E[profit] - (1-γ)*(α - 1/β * E[u])
    # Note: We use negative profit because we're minimizing
    @objective(model, Min,
        -γ * sum(probabilities[i] * profits[i] for i in eachindex(profits)) -
        (1-γ) * (α - 1/β * sum(probabilities[i] * u[i] for i in eachindex(profits)))
    )
    
    # CVaR constraints
    @constraint(model, [i=eachindex(profits)],
        u[i] >= -profits[i] + α
    )
    
    # Solve
    optimize!(model)
    
    # Get results
    var_value = value(α)
    cvar_value = var_value - 1/β * sum(probabilities[i] * value(u[i]) for i in eachindex(profits))
    expected_value = sum(probabilities[i] * profits[i] for i in eachindex(profits))
    
    # Theoretical values
    # VaR should be approximately -10 (the worst case in our example)
    # CVaR should be -10 (since all scenarios <= VaR are equally weighted)
    # Expected value should be 5 (average of profits)
    
    println("Test Results:")
    println("VaR: ", var_value)
    println("CVaR: ", cvar_value)
    println("Expected Value: ", expected_value)
    println("Objective Value: ", objective_value(model))
    
    # Test assertions
    @test isapprox(var_value, -10.0, atol=1e-6)
    @test isapprox(cvar_value, -10.0, atol=1e-6)
    @test isapprox(expected_value, 5.0, atol=1e-6)
end

function test_generator_cvar()
    # Create a simplified version of generator model
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    
    # Parameters (similar to actual model)
    nTimesteps = 24
    W = fill(1/nTimesteps, nTimesteps)
    β_gc = 0.1
    γ_gc = 0.5
    
    # Simple price scenarios
    λ_EOM = [50.0 * (1 + sin(2π*t/nTimesteps)) for t in 1:nTimesteps]
    
    # Variables
    @variable(model, g[1:nTimesteps] >= 0)  # generation
    @variable(model, C >= 0)  # capacity
    @variable(model, α_gc)  # VaR
    @variable(model, u_gc[1:nTimesteps] >= 0)  # CVaR auxiliary
    
    # Simple profit expression (simplified version of your model)
    A = 0.008  # From config
    B = 0
    INV = 500000 * (0.05 / (1 - (1 + 0.05)^(-40)))/8760  # Similar to model
    
    profit = @expression(model, [t=1:nTimesteps],
        λ_EOM[t] * g[t] - (A/2 * g[t]^2 + B * g[t]) - INV * C
    )
    
    # Objective with CVaR
    @objective(model, Min,
        -γ_gc * sum(W[t] * profit[t] for t in eachindex(W)) -
        (1-γ_gc) * (α_gc - 1/β_gc * sum(W[t] * u_gc[t] for t in eachindex(W)))
    )
    
    # CVaR constraint
    @constraint(model, [t=eachindex(W)],
        u_gc[t] >= -profit[t] + α_gc
    )
    
    # Capacity constraint
    @constraint(model, [t=1:nTimesteps],
        g[t] <= C
    )
    
    # Solve
    optimize!(model)
    
    # Results
    var_value = value(α_gc)
    cvar_value = var_value - 1/β_gc * sum(W[t] * value(u_gc[t]) for t in eachindex(W))
    expected_profit = sum(W[t] * value(profit[t]) for t in eachindex(W))
    
    println("\nGenerator Model Results:")
    println("VaR: ", var_value)
    println("CVaR: ", cvar_value)
    println("Expected Profit: ", expected_profit)
    println("Optimal Capacity: ", value(C))
    println("Average Generation: ", mean(value.(g)))
    
    # The CVaR should be lower than VaR, which should be lower than expected profit
    @test cvar_value <= var_value
    @test var_value <= expected_profit
end

function test_numerical_stability()
    println("\n=== Testing Numerical Stability of CVaR Implementation ===")
    
    # Create model with potential numerical issues
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    
    # Parameters that mimic your real model
    nTimesteps = 24
    W = fill(1/nTimesteps, nTimesteps)
    β_gc = 0.1
    γ_gc = 0.5
    
    # Price scenarios with high values (similar to your error message)
    λ_EOM = [600.0 * (0.8 + 0.4*sin(2π*t/nTimesteps)) for t in 1:nTimesteps]
    
    # Variables
    @variable(model, g[1:nTimesteps] >= 0)  # generation
    @variable(model, C >= 0)  # capacity
    @variable(model, α_gc)  # VaR
    @variable(model, u_gc[1:nTimesteps] >= 0)  # CVaR auxiliary
    
    # ADMM penalty parameters
    ρ_EOM = 2.85  # Similar to error message
    g_bar = [1740.0 + 10*sin(t) for t in 1:nTimesteps]  # Similar to error message
    
    # Define profit expression with scaling issues
    A = 0.008
    B = 10.0
    INV = 500000 * (0.05 / (1 - (1 + 0.05)^(-40)))/8760
    
    profit = @expression(model, [t=1:nTimesteps],
        λ_EOM[t] * g[t] - (A/2 * g[t]^2 + B * g[t]) - INV * C
    )
    
    # Objective with CVaR and ADMM penalty
    @objective(model, Min,
        -γ_gc * sum(W[t] * profit[t] for t in 1:nTimesteps) +
        sum(ρ_EOM/2 * (g[t] - g_bar[t])^2 for t in 1:nTimesteps) -
        (1-γ_gc) * (α_gc - 1/β_gc * sum(W[t] * u_gc[t] for t in 1:nTimesteps))
    )
    
    # CVaR constraint
    @constraint(model, [t=1:nTimesteps],
        u_gc[t] >= -profit[t] + α_gc
    )
    
    # Capacity constraint
    @constraint(model, [t=1:nTimesteps],
        g[t] <= C
    )
    
    # Try different numerical focus settings to improve stability
    println("\nTesting with original parameter values...")
    
    # Set parameters for numerical stability
    set_optimizer_attribute(model, "NumericFocus", 0)
    
    # Solve
    optimize!(model)
    status1 = termination_status(model)
    
    println("Status: ", status1)
    if status1 == MOI.OPTIMAL
        println("Kappa: ", unsafe_backend(model).inner.kappa)
        println("Matrix range: ", unsafe_backend(model).inner.getAttr("MaxCoeff") / 
                                  unsafe_backend(model).inner.getAttr("MinCoeff"))
    end
    
    # Try with scaled prices
    println("\nTesting with price scaling (prices divided by 100)...")
    
    model2 = copy(model)
    set_silent(model2)
    
    # Scale down prices (decrease coefficient range)
    λ_EOM_scaled = λ_EOM ./ 100.0
    
    # Update the profit expression with scaled prices
    new_profit = @expression(model2, [t=1:nTimesteps],
        λ_EOM_scaled[t] * g[t] - (A/2 * g[t]^2 + B * g[t]) - INV * C
    )
    
    # Update objective with scaled profit
    @objective(model2, Min,
        -γ_gc * sum(W[t] * new_profit[t] for t in 1:nTimesteps) +
        sum(ρ_EOM/2 * (g[t] - g_bar[t])^2 for t in 1:nTimesteps) -
        (1-γ_gc) * (α_gc - 1/β_gc * sum(W[t] * u_gc[t] for t in 1:nTimesteps))
    )
    
    # Update constraints with scaled profit
    for t in 1:nTimesteps
        delete(model2, constraint_by_name(model2, "c$t"))
    end
    
    @constraint(model2, [t=1:nTimesteps],
        u_gc[t] >= -new_profit[t] + α_gc
    )
    
    # Set parameters for numerical stability
    set_optimizer_attribute(model2, "NumericFocus", 0)
    
    # Solve
    optimize!(model2)
    status2 = termination_status(model2)
    
    println("Status: ", status2)
    if status2 == MOI.OPTIMAL
        println("Kappa: ", unsafe_backend(model2).inner.kappa)
        println("Matrix range: ", unsafe_backend(model2).inner.getAttr("MaxCoeff") / 
                                   unsafe_backend(model2).inner.getAttr("MinCoeff"))
    end
    
    # Try with NumericFocus
    println("\nTesting with NumericFocus = 3...")
    
    model3 = copy(model)
    set_silent(model3)
    
    # Set parameters for numerical stability
    set_optimizer_attribute(model3, "NumericFocus", 3)
    
    # Solve
    optimize!(model3)
    status3 = termination_status(model3)
    
    println("Status: ", status3)
    if status3 == MOI.OPTIMAL
        println("Kappa: ", unsafe_backend(model3).inner.kappa)
        println("Matrix range: ", unsafe_backend(model3).inner.getAttr("MaxCoeff") / 
                                   unsafe_backend(model3).inner.getAttr("MinCoeff"))
    end
    
    # Return recommendations
    println("\n=== Recommendations for Fixing Numerical Errors ===")
    println("1. Price Scaling: Scale your prices (λ_EOM) by dividing by 100 to reduce coefficient range")
    println("2. ADMM Parameters: Limit your ρ_EOM values to prevent excessive growth")
    println("3. Add bounds: Set reasonable upper bounds on all variables")
    println("4. Solver Options: Add NumericFocus=3 to your Gurobi parameters")
    println("5. Presolve: Try setting Presolve=0 if the issue persists")
    
    # Print specific implementation guidance
    println("\n=== Implementation Guidance ===")
    println("Add these lines in your model script or scenario 3 handling:")
    println("```julia")
    println("# Scale down prices for numerical stability")
    println("λ_EOM_scale_factor = 100.0")
    println("λ_EOM_scaled = λ_EOM ./ λ_EOM_scale_factor")
    println()
    println("# Use scaled prices in profit expression")
    println("profit = @expression(mod, [jh in JH],")
    println("    λ_EOM_scaled[jh] * g[jh] - (A/2 * g[jh]^2 + B * g[jh]) - INV * C")
    println(")")
    println()
    println("# Don't forget to scale back when reporting results")
    println("```")
    println()
    println("For your ADMM implementation:")
    println("```julia")
    println("# Limit ρ growth in update_rho.jl")
    println("push!(ADMM[\"ρ\"][\"EOM\"], minimum([100, 1.1*ADMM[\"ρ\"][\"EOM\"][end]]))")
    println("```")
end

function check_numerical_issues(model)
    println("\n=== Numerical Diagnostics ===")
    
    # 1. Check Model Statistics
    println("\nModel Statistics:")
    println("Number of variables: ", num_variables(model))
    println("Number of constraints: ", num_constraints(model, count_variable_in_set_constraints=true))
    
    # 2. Check Variable Bounds
    println("\nVariable Bounds:")
    for var in all_variables(model)
        name = name(var)
        lb = has_lower_bound(var) ? lower_bound(var) : "-inf"
        ub = has_upper_bound(var) ? upper_bound(var) : "inf"
        println("$name: [$lb, $ub]")
    end
    
    # 3. Check Coefficient Ranges
    println("\nCoefficient Ranges:")
    obj = objective_function(model)
    if obj isa GenericQuadExpr
        quad_coeffs = [coef for (_, coef) in quad_terms(obj)]
        lin_coeffs = [coef for (_, coef) in linear_terms(obj)]
        
        if !isempty(quad_coeffs)
            println("Quadratic terms: [$(minimum(quad_coeffs)), $(maximum(quad_coeffs))]")
        end
        if !isempty(lin_coeffs)
            println("Linear terms: [$(minimum(lin_coeffs)), $(maximum(lin_coeffs))]")
        end
    end
    
    # 4. Solution Quality Check
    if termination_status(model) == MOI.OPTIMAL
        println("\nSolution Quality:")
        println("Objective value: ", objective_value(model))
        println("Dual objective value: ", dual_objective_value(model))
        println("Relative gap: ", relative_gap(model))
        println("Solver time: ", solve_time(model))
    end
    
    # 5. Gurobi-specific diagnostics
    if solver_name(model) == "Gurobi"
        println("\nGurobi Diagnostics:")
        println("Kappa (condition number): ", 
                get_optimizer_attribute(model, "Kappa"))
        println("Maximum coefficient range: ", 
                get_optimizer_attribute(model, "MaxCoeff") / 
                get_optimizer_attribute(model, "MinCoeff"))
    end
end

function test_model_stability()
    println("\n=== Testing Model Stability ===")
    
    # Create a simple test model with potential numerical issues
    m = Model(Gurobi.Optimizer)
    
    # Test different numerical focus settings
    for num_focus in 0:3
        set_optimizer_attribute(m, "NumericFocus", num_focus)
        
        # Solve and check results
        optimize!(m)
        
        println("\nWith NumericFocus = $num_focus:")
        println("Status: ", termination_status(m))
        if termination_status(m) == MOI.OPTIMAL
            check_numerical_issues(m)
        end
    end
    
    # Recommendations based on results
    println("\nRecommendations:")
    println("1. If Kappa > 1e7, consider scaling your coefficients")
    println("2. If MaxCoeff/MinCoeff > 1e6, normalize your variables")
    println("3. If seeing numerical errors, try NumericFocus=3")
end

# Function to test with different scaling factors
function test_with_scaling(λ_EOM, scale_factors=[1.0, 10.0, 100.0, 1000.0])
    println("\n=== Testing Different Scaling Factors ===")
    
    for scale in scale_factors
        println("\nTesting with scale factor: $scale")
        
        # Scale the prices
        λ_scaled = λ_EOM ./ scale
        
        # Create and solve a test model
        m = Model(Gurobi.Optimizer)
        # ... your model setup with scaled prices ...
        
        optimize!(m)
        
        println("Maximum coefficient: ", maximum(λ_scaled))
        println("Minimum coefficient: ", minimum(λ_scaled))
        println("Status: ", termination_status(m))
    end
end

# Run tests
println("Running simple CVaR test...")
test_simple_cvar()

println("\nRunning generator CVaR test...")
test_generator_cvar()

println("\nRunning numerical stability test...")
test_numerical_stability()

println("\nRunning model stability test...")
test_model_stability()

# End with recommendation
println("\nTo run this test: include(\"test_cvar.jl\")")
println("Then apply the recommended changes to fix the numerical errors")

