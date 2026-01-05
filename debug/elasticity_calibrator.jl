using YAML, CSV, DataFrames
using Printf
using Statistics

# ====================================================================
# VoLL-Based Demand Elasticity Calibrator
# ====================================================================
# Greek Regulatory VoLL Data (ACER study):
#   - Domestic consumers:     4.240 M€/GWh (4,240 €/MWh)
#   - Non-domestic consumers: 0.410-2.380 M€/GWh (industrial/commercial)
#   - System average:         6.838 M€/GWh
#
# New Calibration Logic:
#   INPUT:  λ_EOM_0 (VoLL), D_cap_ratio
#   CALC:   E_EOM = (λ_ref - λ_0) / D_ref  [vector per timestep]
#   CALC:   λ_cap = λ_0 + D_cap_ratio * (λ_ref - λ_0)  [vector]
# ====================================================================

println("=" ^ 80)
println("VoLL-BASED DEMAND ELASTICITY CALIBRATOR")
println("=" ^ 80)
println()

# --- Target VoLL values from Greek regulatory data ---
VOLL_DOMESTIC = 4.240      # M€/GWh for residential (LV_LOW, LV_MED)
VOLL_NONDOMESTIC_LOW = 0.410   # M€/GWh industrial lower bound
VOLL_NONDOMESTIC_HIGH = 2.380  # M€/GWh industrial upper bound
VOLL_NONDOMESTIC_MID = (VOLL_NONDOMESTIC_LOW + VOLL_NONDOMESTIC_HIGH) / 2  # 1.395

println("📊 Greek Regulatory VoLL (ACER Study):")
println("   Domestic (residential):       $(VOLL_DOMESTIC) M€/GWh (4,240 €/MWh)")
println("   Non-domestic (industrial):    $(VOLL_NONDOMESTIC_LOW)-$(VOLL_NONDOMESTIC_HIGH) M€/GWh (410-2,380 €/MWh)")
println("   Non-domestic (midpoint):      $(round(VOLL_NONDOMESTIC_MID, digits=3)) M€/GWh")
println()

# --- Load scenario 1 results for λ_EOM_ref ---
df_scen1 = CSV.read("Results/Scenario_1_ref.csv", DataFrame; delim=';')

# --- Load config and timeseries ---
config = YAML.load_file("Input/config.yaml")
tot_consumers = config["General"]["totConsumers"]
consumers = config["Consumers"]

demand_df = CSV.read("Input/ts_demand_12d.csv", DataFrame)

println("=" ^ 80)
println("CURRENT CONFIG ANALYSIS")
println("=" ^ 80)
println()

for (agent, cons) in consumers
    println("\n" * "─" ^ 80)
    println("AGENT: $agent")
    println("─" ^ 80)
    
    # --- Load agent data ---
    share = cons["Share"]
    n_agents = share * tot_consumers
    demand_col = cons["D"]
    demand_ts = demand_df[!, demand_col]
    demand_ts_scaled = n_agents .* demand_ts
    
    # Read current config values
    λ_EOM_cap_config = get(cons, "λ_EOM_cap", nothing)
    D_cap_ratio_config = get(cons, "D_cap_ratio", 0.8)
    λ_EOM_0_config = get(cons, "λ_EOM_0", nothing)
    
    # Compute λ_EOM_ref from scenario 1 results
    demand_col_scen1 = "D_" * agent
    if hasproperty(df_scen1, Symbol(demand_col_scen1))
        agent_demand = df_scen1[!, Symbol(demand_col_scen1)]
    else
        println("⚠️  Column $(demand_col_scen1) not found in Scenario_1_ref.csv")
        println("   Run Scenario 1 first, or this is a new agent.")
        continue
    end
    numerator = sum(df_scen1.Price .* agent_demand)
    denominator = sum(agent_demand)
    λ_EOM_ref = numerator / denominator
    
    println("\n📈 Reference Price (from Scenario 1):")
    @printf("   λ_EOM_ref = %.4f M€/GWh (%.1f €/MWh)\n", λ_EOM_ref, λ_EOM_ref * 1000)
    
    println("\n⚙️  Current Config:")
    if λ_EOM_0_config !== nothing
        @printf("   λ_EOM_0 (VoLL)    = %.4f M€/GWh (%.1f €/MWh)\n", λ_EOM_0_config, λ_EOM_0_config * 1000)
    else
        println("   λ_EOM_0 (VoLL)    = NOT SET")
    end
    @printf("   D_cap_ratio       = %.2f (can reduce to %.0f%% of reference)\n", D_cap_ratio_config, D_cap_ratio_config * 100)
    if λ_EOM_cap_config !== nothing
        @printf("   λ_EOM_cap         = %.4f M€/GWh (%.1f €/MWh)\n", λ_EOM_cap_config, λ_EOM_cap_config * 1000)
    else
        println("   λ_EOM_cap         = NOT SET")
    end
    
    # --- Calculate with NEW logic (VoLL-based) ---
    if λ_EOM_0_config !== nothing
        λ_0 = λ_EOM_0_config
        D_cap_ratio = D_cap_ratio_config
        
        # E_EOM = (λ_ref - λ_0) / D_ref [vector]
        E_EOM_vec = (λ_EOM_ref .- λ_0) ./ demand_ts_scaled
        
        # λ_cap = λ_0 + D_cap_ratio * (λ_ref - λ_0) [vector]
        λ_cap_vec = λ_0 .+ D_cap_ratio .* (λ_EOM_ref .- λ_0)
        
        println("\n📊 Calculated Parameters (VoLL-based):")
        @printf("   E_EOM:    min = %9.4f, max = %9.4f, mean = %9.4f\n",
                minimum(E_EOM_vec), maximum(E_EOM_vec), mean(E_EOM_vec))
        @printf("   λ_cap:    min = %9.4f, max = %9.4f, mean = %9.4f M€/GWh\n",
                minimum(λ_cap_vec), maximum(λ_cap_vec), mean(λ_cap_vec))
        @printf("             (%.1f - %.1f €/MWh, avg %.1f €/MWh)\n",
                minimum(λ_cap_vec)*1000, maximum(λ_cap_vec)*1000, mean(λ_cap_vec)*1000)
        
        # Check sign
        if any(E_EOM_vec .>= 0)
            println("   ⚠️  WARNING: Some E_EOM ≥ 0! Expected negative (downward-sloping demand).")
            println("       This means λ_0 < λ_ref, which is incorrect (VoLL should be > ref price)")
        else
            println("   ✅ All E_EOM < 0 (correct negative slope)")
        end
    end
    
    # --- SUGGESTIONS for target VoLL ---
    println("\n" * "─" ^ 60)
    println("💡 SUGGESTIONS TO MATCH GREEK REGULATORY VoLL")
    println("─" ^ 60)
    
    # Determine target VoLL based on agent type
    if agent in ["LV_LOW", "LV_MED"]
        target_voll = VOLL_DOMESTIC
        consumer_type = "Domestic (residential)"
    else  # LV_HIGH, MV_LOAD
        target_voll = VOLL_NONDOMESTIC_MID
        consumer_type = "Non-domestic (industrial/commercial)"
    end
    
    println("Consumer Type: $consumer_type")
    @printf("Target VoLL:   %.3f M€/GWh (%.0f €/MWh)\n", target_voll, target_voll * 1000)
    println()
    
    # Test different D_cap_ratio values
    test_ratios = [0.7, 0.75, 0.8, 0.85, 0.9]
    
    println("Suggested config.yaml values:")
    println()
    for ratio in test_ratios
        E_test = (λ_EOM_ref - target_voll) / mean(demand_ts_scaled)
        λ_cap_test = target_voll + ratio * (λ_EOM_ref - target_voll)
        
        @printf("   D_cap_ratio: %.2f (%.0f%% reduction) → λ_cap_avg ≈ %.4f M€/GWh (%.0f €/MWh)\n",
                ratio, (1-ratio)*100, λ_cap_test, λ_cap_test * 1000)
    end
    
    println()
    println("Recommended config snippet:")
    println("   $agent:")
    @printf("     λ_EOM_0: %.3f      # VoLL (%s)\n", target_voll, consumer_type)
    println("     D_cap_ratio: 0.80  # Can reduce to 80% (20% voluntary curtailment)")
    println()
end

println("\n" * "=" ^ 80)
println("END OF CALIBRATION REPORT")
println("=" ^ 80)
