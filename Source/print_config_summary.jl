using YAML

function print_solar_capacity_summary(data::Dict)
    println("=== SOLAR CAPACITY SUMMARY ===")
    total_consumers = data["General"]["totConsumers"]
    total_solar_gw = 0.0

    for consumer_type in keys(data["Consumers"])
        share = data["Consumers"][consumer_type]["Share"]
        pv_cap_per_consumer = data["Consumers"][consumer_type]["PV_cap"]  # in GWp
        
        # Calculate solar capacity for this consumer type
        consumers_of_this_type = total_consumers * share
        solar_gw_this_type = consumers_of_this_type * pv_cap_per_consumer
        total_solar_gw += solar_gw_this_type
        
        # Print details for each consumer type
        @printf "  %-12s: %8.0f consumers × %.6f GWp = %.3f GW\n" consumer_type consumers_of_this_type pv_cap_per_consumer solar_gw_this_type
    end

    println("  " * repeat("=", 50))
    @printf "  %-12s: %8.0f consumers total    = %.3f GW\n" "TOTAL" total_consumers total_solar_gw
    println("===============================")
    println("")
    
    return total_solar_gw
end 

function print_generator_merit_order(mdict::Dict, agents::Dict)
    println("=== GENERATOR COST SUMMARY ===")
    generator_costs = []
    
    for m in agents[:Gen]
        a_cost = mdict[m].ext[:parameters][:A]  # Quadratic term coefficient
        b_cost = mdict[m].ext[:parameters][:B]  # Linear term coefficient
        inv_hourly_cost = mdict[m].ext[:parameters][:INV_h]  # Investment cost per hour (M€/GW/h)
        
        # Calculate costs for 1 GWh and 5 GWh generation
        generation_levels = [1.0, 5.0]  # GWh
        costs = []
        
        for gen in generation_levels
            # Marginal cost (M€/GWh)
            marginal_cost = a_cost * gen + b_cost
            
            # Investment cost per GWh (M€/GWh)
            investment_cost = inv_hourly_cost
            
            # Total cost per GWh (M€/GWh)
            total_cost = marginal_cost + investment_cost
            
            push!(costs, (Generation=gen, MC=marginal_cost, INV_h=investment_cost, Total=total_cost))
        end
        
        push!(generator_costs, (
            Generator=m,
            A=a_cost,
            B=b_cost,
            INV_h=inv_hourly_cost,
            Costs=costs
        ))
    end

    # Sort by total cost at 1 GWh
    sort!(generator_costs, by = x -> x.Costs[1].Total)
    
    println("  Generator Cost Summary (M€/GWh)")
    println("  " * repeat("-", 80))
    println("  Generator    INV_h     MC(1GWh)  Total(1GWh)  MC(5GWh)  Total(5GWh)")
    println("  " * repeat("-", 80))
    
    for item in generator_costs
        cost_1 = item.Costs[1]
        cost_5 = item.Costs[2]
        @printf "  %-12s %8.3f   %8.3f   %10.3f   %8.3f   %10.3f\n" item.Generator item.INV_h cost_1.MC cost_1.Total cost_5.MC cost_5.Total
    end
        
    println(repeat("=", 80))
    println("")
    
    return generator_costs
end 