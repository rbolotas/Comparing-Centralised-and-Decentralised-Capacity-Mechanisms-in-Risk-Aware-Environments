#Metrics Module
import pandas as pd
import numpy as np
from pathlib import Path
from config_utils import get_ntimesteps

# Get the project root directory (2 levels up from the Analysis directory)
PROJECT_ROOT = Path(__file__).parent.parent

# Constants for time scaling - read from config file
N_TIMESTEPS = get_ntimesteps()  # Number of timesteps from config.yaml
# NO SCALING - just use absolute values for simulation period

def load_scenario_data(scenario_num, variant="ref"):
    """Load data for a specific scenario."""
    file_path = Path(f"../Results/Scenario_{scenario_num}_{variant}.csv")
    try:
        return pd.read_csv(file_path, delimiter=";")
    except Exception:
        # Fallbacks for common cases
        fallback_variants = []
        if scenario_num in [3, 5, 7]:
            fallback_variants = ["beta_1.0", "beta_0.9"]
        elif scenario_num in [1, 2]:
            fallback_variants = ["ref"]
        for fv in fallback_variants:
            fp = Path(f"../Results/Scenario_{scenario_num}_{fv}.csv")
            if fp.exists():
                return pd.read_csv(fp, delimiter=";")
        # If still missing, re-raise to be handled by caller
        raise

def calculate_price_metrics(df):
    """Price statistics incl. CVaR on the 5 % most expensive hours."""
    prices = df["Price"]
    alpha  = 0.95                             # confidence level
    var95  = prices.quantile(alpha)
    var05  = prices.quantile(0.05)            # 5% quantile
    cvar95 = prices[prices >= var95].mean()   # Conditional VaR (Tail-Mean)

    return {
        "mean_price":       prices.mean(),
        "median_price":     prices.median(),
        "price_volatility": prices.std(),
        "max_price":        prices.max(),
        "min_price":        prices.min(),
        "price_p95":        var95,
        "price_p05":        var05,
        "price_cvar95":     cvar95,
        "price_data":       prices.tolist(),  # Raw price data for boxplot
    }

def calculate_capacity_metrics(df, scenario_num):
    """Capacity-market related indicators (zero for EOM scenarios)."""
    
    # Calculate installed dispatchable capacity (max instantaneous production)
    dispatchable_gens = ["Baseload", "MidMerit", "Peak"]
    total_dispatchable_capacity = 0
    dispatchable_capacity_by_type = {}
    
    for gen in dispatchable_gens:
        if f"G_{gen}" in df.columns:
            gen_capacity = df[f"G_{gen}"].max()  # Max instantaneous production
            dispatchable_capacity_by_type[gen] = gen_capacity
            total_dispatchable_capacity += gen_capacity
        else:
            dispatchable_capacity_by_type[gen] = 0
    
    # For energy-only scenarios (1-3), calculate total capacity from individual generators
    if scenario_num <= 3:
        # Calculate total capacity from generation columns (including renewables)
        gen_columns = [col for col in df.columns if col.startswith("G_") and not any(c in col for c in ["LV_MED", "MV_LOAD", "LV_HIGH", "LV_LOW"])]
        # Use maximum generation across all hours as proxy for capacity
        total_capacity = sum(df[col].max() for col in gen_columns)
    else:
        # For capacity market scenarios, use the explicit capacity column
        total_capacity = df["C_system_total"].iloc[0] if "C_system_total" in df.columns else 0.0
    
    metrics = {
        "total_capacity": total_capacity,
        "total_dispatchable_capacity": total_dispatchable_capacity,
        "dispatchable_capacity_by_type": dispatchable_capacity_by_type,
        "capacity_price": 0.0,
        "capacity_volume": 0.0,
        "market_type": "Energy-Only Market"
    }

    if scenario_num == 4 or scenario_num == 5:  # Central CM (with or without CVaR)
        # Raw capacity price from data (no scaling)
        metrics.update({
            "capacity_price": df["λ_cCM"].iloc[0],  # Raw value from data
            "capacity_volume": df["C_cCM_vol"].iloc[0],
            "market_type": "Centralized Capacity Market"
        })
    elif scenario_num == 6 or scenario_num == 7:  # Decentral CM
        # Raw capacity price from data (no scaling)
        metrics.update({
            "capacity_price": df["λ_dCM"].iloc[0],  # Raw value from data
            "capacity_volume": df["C_dCM_vol"].iloc[0],
            "market_type": "Decentralized Capacity Market"
        })
    return metrics

def calculate_reliability_metrics(df):
    """Calculate reliability metrics including LOLE and ENS."""
    # Identify all ENS columns
    ens_columns = [col for col in df.columns if col.startswith("ENS_")]
    
    # Calculate total ENS per hour
    df_copy = df.copy()  # Avoid modifying original dataframe
    df_copy["total_ENS"] = df_copy[ens_columns].sum(axis=1)
    
    # Get number of timesteps for scaling
    n_timesteps = get_ntimesteps()
    annual_scaling_factor = 8760 / n_timesteps
    
    # LOLE: count hours with any ENS, scaled to annual hours
    hours_with_ens = (df_copy["total_ENS"] > 0).sum()  # Hours with ENS in simulation period
    lole = hours_with_ens * annual_scaling_factor  # Scale to annual hours
    
    # Total ENS for 12-day period
    total_ens = df_copy["total_ENS"].sum()
    
    # 95th percentile of hourly ENS (when ENS > 0)
    ens_95 = df_copy.loc[df_copy["total_ENS"] > 0, "total_ENS"].quantile(0.95) if (df_copy["total_ENS"] > 0).any() else 0
    
    return {
        "LOLE": lole,  # Hours with load shedding (annualized)
        "total_ENS": total_ens,  # Total ENS in GWh for 12-day period
        "ENS_95": ens_95
    }

def calculate_generator_metrics(df, scenario_num):
    """Calculate generator profitability and cost recovery."""
    # Identify generator columns
    gen_columns = [col for col in df.columns if col.startswith("G_") and not any(c in col for c in ["LV_MED", "MV_LOAD", "LV_HIGH", "LV_LOW"])]
    generators = [col[2:] for col in gen_columns]  # Remove "G_" prefix
    
    results = {}
    
    for gen in generators:
        # Energy market revenue for 12-day period (in €)
        energy_revenue = (df["Price"] * df[f"G_{gen}"]).sum()
        
        # Capacity market revenue (if applicable) - raw values for 12-day period
        cap_revenue = 0
        if (scenario_num == 4 or scenario_num == 5) and f"C_cCM_{gen}" in df.columns:
            # Raw capacity revenue for 12-day period
            cap_revenue = df["λ_cCM"].iloc[0] * df[f"C_cCM_{gen}"].iloc[0] * N_TIMESTEPS
        elif (scenario_num == 6 or scenario_num == 7) and f"C_dCM_{gen}" in df.columns:
            cap_revenue = df["λ_dCM"].iloc[0] * df[f"C_dCM_{gen}"].iloc[0] * N_TIMESTEPS
        
        results[gen] = {
            "energy_revenue": energy_revenue,
            "capacity_revenue": cap_revenue,
            "total_revenue": energy_revenue + cap_revenue
        }
    
    return results

def calculate_system_cost_metrics(df, scenario_num):
    """Calculate system costs from generation perspective: generation * price + capacity * capacity_price."""
    
    # Identify all generation columns (excluding consumers)
    gen_columns = [col for col in df.columns if col.startswith("G_") and not any(c in col for c in ["LV_MED", "MV_LOAD", "LV_HIGH", "LV_LOW"])]
    generators = [col[2:] for col in gen_columns]  # Remove "G_" prefix
    
    # Calculate energy costs (generation * price) for each generator (12-day period)
    energy_costs = {}
    total_energy_cost = 0
    
    #TODO add EOM cost for boxplot

    for gen in generators:
        gen_energy_cost = (df["Price"] * df[f"G_{gen}" ]).sum()  # €
        energy_costs[gen] = gen_energy_cost
        total_energy_cost += gen_energy_cost
    
    # Calculate capacity costs (capacity * capacity_price) for 12-day period
    capacity_costs = {}
    total_capacity_cost = 0
    
    if scenario_num >= 4:  # Capacity market scenarios
        if scenario_num == 4 or scenario_num == 5:  # Central CM
            capacity_price = df["λ_cCM"].iloc[0]  # M€/GW/Ntimesteps
            for gen in generators:
                if f"C_cCM_{gen}" in df.columns:
                    # Convert capacity to GW if needed (assume it's in GW)
                    gen_capacity_cost = capacity_price * df[f"C_cCM_{gen}"].iloc[0]  # M€
                    capacity_costs[gen] = gen_capacity_cost
                    total_capacity_cost += gen_capacity_cost
                else:
                    capacity_costs[gen] = 0
        elif scenario_num == 6 or scenario_num == 7:  # Decentral CM
            capacity_price = df["λ_dCM"].iloc[0]  # M€/GW/Ntimesteps
            for gen in generators:
                if f"C_dCM_{gen}" in df.columns:
                    gen_capacity_cost = capacity_price * df[f"C_dCM_{gen}"].iloc[0]  # M€
                    capacity_costs[gen] = gen_capacity_cost
                    total_capacity_cost += gen_capacity_cost
                else:
                    capacity_costs[gen] = 0
    else:
        # Energy-only scenarios - no capacity costs
        capacity_costs = {gen: 0 for gen in generators}
    
    # Calculate total system costs
    system_costs_by_generator = {}
    for gen in generators:
        system_costs_by_generator[gen] = {
            "energy_cost": energy_costs[gen],  # M€
            "capacity_cost": capacity_costs[gen],  # M€
            "total_cost": (energy_costs[gen] + capacity_costs[gen])  # M€
            #TODO add EOM cost for boxplot
        }
    
    # Calculate total demand for cost per MWh calculation (12-day period)
    total_demand_gwh = df["Total_Demand"].sum() if "Total_Demand" in df.columns else df["HV_LOAD"].sum()  # GWh
    total_demand_mwh = abs(total_demand_gwh) * 1e3  # MWh

    # Summary metrics
    total_system_cost = total_energy_cost + total_capacity_cost  # €

    # Calculate per-timestep system cost (for boxplot)
    if "Total_Demand" in df.columns:
        per_timestep_cost_eur = (df["Price"] * df["Total_Demand"]).values  # € per hour
        per_timestep_demand_mwh = df["Total_Demand"].values * 1e3  # MWh per hour
    else:
        per_timestep_cost_eur = (df["Price"] * df["HV_LOAD"]).values
        per_timestep_demand_mwh = df["HV_LOAD"].values * 1e3
    per_timestep_cost_per_mwh = np.divide(per_timestep_cost_eur, per_timestep_demand_mwh, out=np.zeros_like(per_timestep_cost_eur), where=per_timestep_demand_mwh!=0)  # €/MWh per hour

    return {
        "by_generator": system_costs_by_generator,  # M€
        "total_energy_cost": total_energy_cost,  # M€
        "total_capacity_cost": total_capacity_cost,  # M€
        "total_system_cost": total_system_cost,  # M€
        "system_cost_per_mwh": (total_system_cost / total_demand_mwh) if total_demand_mwh != 0 else 0,  # €/MWh
        "energy_cost_per_mwh": (total_energy_cost / total_demand_mwh) if total_demand_mwh != 0 else 0,  # €/MWh
        "capacity_cost_per_mwh": (total_capacity_cost / total_demand_mwh) if total_demand_mwh != 0 else 0,  # €/MWh
        "per_timestep_cost_per_mwh": per_timestep_cost_per_mwh,  # €/MWh per hour
        "per_timestep_cost_eur": per_timestep_cost_eur,  # M€ per hour
    }

def calculate_generation_mix_metrics(df):
    """Calculate generation mix including actual PV generation for consumers."""
    
    # Conventional generators
    conventional_gens = ["WindOnshore", "WindOffshore", "Baseload", "MidMerit", "Peak"]
    consumer_types = ["LV_MED", "MV_LOAD", "LV_HIGH", "LV_LOW"]
    
    generation_mix = {}
    
    # Calculate total generation from conventional sources (in GWh for 12-day period)
    for gen in conventional_gens:
        if f"G_{gen}" in df.columns:
            generation_mix[gen] = df[f"G_{gen}"].sum()  # Sum over N_TIMESTEPS hours = GWh for simulation period
        else:
            generation_mix[gen] = 0
    
    # Calculate actual PV generation for each consumer type (in GWh for 12-day period)
    for consumer in consumer_types:
        if f"PV_{consumer}" in df.columns:
            # Use actual PV generation data
            generation_mix[f"Solar_{consumer}"] = df[f"PV_{consumer}"].sum()  # GWh for 12 days
        else:
            generation_mix[f"Solar_{consumer}"] = 0
    
    # Calculate total generation
    total_generation = sum(generation_mix.values())
    
    # Calculate percentages
    generation_percentages = {}
    if total_generation > 0:
        for source, amount in generation_mix.items():
            generation_percentages[source] = (amount / total_generation) * 100
    else:
        generation_percentages = {source: 0 for source in generation_mix.keys()}
    
    return {
        "absolute": generation_mix,
        "percentages": generation_percentages,
        "total_generation": total_generation
    }

def calculate_enhanced_consumer_metrics(df, scenario_num):
    """Calculate enhanced consumer metrics including industrial demand breakdown."""
    # Consumer types - treating them as different demand categories
    # Use only those consumer types that actually exist in the dataset
    consumer_candidates = ["LV_MED", "MV_LOAD", "LV_HIGH", "LV_LOW"]
    consumer_types = [c for c in consumer_candidates if f"G_{c}" in df.columns]
    
    # Calculate energy costs for each consumer type (12-day period in €)
    energy_costs = {}
    for consumer in consumer_types:
        # Net energy cost = price * net consumption (G_Type is negative for net consumption)
        energy_costs[consumer] = -(df["Price"] * df[f"G_{consumer}"]).sum()
    
    # Calculate capacity costs (if applicable) - for 12-day period
    capacity_costs = {}
    if scenario_num == 6 or scenario_num == 7:  # Decentralized CM
        for consumer in consumer_types:
            if f"C_dCM_{consumer}" in df.columns:
                # Raw capacity cost for 12-day period
                capacity_costs[consumer] = df["λ_dCM"].iloc[0] * df[f"C_dCM_{consumer}"].iloc[0] * N_TIMESTEPS
            else:
                capacity_costs[consumer] = 0
    else:
        # For centralized CM, allocate costs proportionally to consumption
        total_consumption = sum([-df[f"G_{consumer}"].sum() for consumer in consumer_types])
        if (scenario_num == 4 or scenario_num == 5) and total_consumption > 0:
            total_cap_cost = df["λ_cCM"].iloc[0] * df["C_cCM_vol"].iloc[0] * N_TIMESTEPS
            for consumer in consumer_types:
                consumption_share = -df[f"G_{consumer}"].sum() / total_consumption
                capacity_costs[consumer] = total_cap_cost * consumption_share
        else:
            capacity_costs = {consumer: 0 for consumer in consumer_types}
    
    # Calculate industrial demand (inflexible) - in GWh for 12-day period
    industrial_demand = {}
    behind_meter_solar = {}
    for consumer in consumer_types:
        if f"D_{consumer}" in df.columns:
            # Total demand for this consumer type (GWh)
            industrial_demand[consumer] = df[f"D_{consumer}"].sum()
        else:
            industrial_demand[consumer] = 0
            
        if f"PV_{consumer}" in df.columns:
            # Use actual PV generation data
            behind_meter_solar[consumer] = df[f"PV_{consumer}"].sum()
        else:
            behind_meter_solar[consumer] = 0
    
    # Combine metrics
    results = {}
    for consumer in consumer_types:
        results[consumer] = {
            "energy_cost": energy_costs[consumer],
            "capacity_cost": capacity_costs.get(consumer, 0),
            "total_cost": energy_costs[consumer] + capacity_costs.get(consumer, 0),
            "industrial_demand": industrial_demand[consumer],
            "behind_meter_solar": behind_meter_solar[consumer],
            "net_consumption": -df[f"G_{consumer}"].sum(),  # Net consumption from grid (GWh)
        }
    
    # Add total metrics
    results["total"] = {
        "energy_cost": sum(energy_costs.values()),
        "capacity_cost": sum(capacity_costs.values()),
        "total_cost": sum(energy_costs.values()) + sum(capacity_costs.values()),
        "industrial_demand": sum(industrial_demand.values()),
        "behind_meter_solar": sum(behind_meter_solar.values()),
        "net_consumption": sum([-df[f"G_{consumer}"].sum() for consumer in consumer_types]) if consumer_types else 0,
    }
    
    return results

def calculate_consumer_metrics(df, scenario_num):
    """Calculate consumer costs and benefits (legacy function for backward compatibility)."""
    # Consumer types
    consumer_candidates = ["LV_MED", "MV_LOAD", "LV_HIGH", "LV_LOW"]
    consumer_types = [c for c in consumer_candidates if f"G_{c}" in df.columns]
    
    # Calculate energy costs (12-day period)
    energy_costs = {}
    for consumer in consumer_types:
        energy_costs[consumer] = -(df["Price"] * df[f"G_{consumer}"]).sum()
    
    # Calculate capacity costs (if applicable) - for 12-day period
    capacity_costs = {}
    if scenario_num == 6 or scenario_num == 7:  # Decentralized CM
        for consumer in consumer_types:
            if f"C_dCM_{consumer}" in df.columns:
                capacity_costs[consumer] = df["λ_dCM"].iloc[0] * df[f"C_dCM_{consumer}"].iloc[0] * N_TIMESTEPS
            else:
                capacity_costs[consumer] = 0
    else:
        # For centralized CM, allocate costs proportionally to consumption
        total_consumption = sum([-df[f"G_{consumer}"].sum() for consumer in consumer_types])
        if (scenario_num == 4 or scenario_num == 5) and total_consumption > 0:
            total_cap_cost = df["λ_cCM"].iloc[0] * df["C_cCM_vol"].iloc[0] * N_TIMESTEPS
            for consumer in consumer_types:
                consumption_share = -df[f"G_{consumer}"].sum() / total_consumption
                capacity_costs[consumer] = total_cap_cost * consumption_share
        else:
            capacity_costs = {consumer: 0 for consumer in consumer_types}
    
    # Combine metrics
    results = {}
    for consumer in consumer_types:
        results[consumer] = {
            "energy_cost": energy_costs[consumer],
            "capacity_cost": capacity_costs.get(consumer, 0),
            "total_cost": energy_costs[consumer] + capacity_costs.get(consumer, 0)
        }
    
    # Add total costs
    results["total"] = {
        "energy_cost": sum(energy_costs.values()),
        "capacity_cost": sum(capacity_costs.values()),
        "total_cost": sum(energy_costs.values()) + sum(capacity_costs.values())
    }
    
    return results

def calculate_crm_cost_metrics(df, scenario_num):
    """Total and specific CRM cost seen by the system demand."""
    if scenario_num < 4:
        return {"crm_cost_total": 0.0, "crm_cost_per_mwh": 0.0}

    # Calculate demand for 12-day period
    demand_mwh = abs(df["HV_LOAD"].sum()) if "HV_LOAD" in df.columns else abs(df["Total_Demand"].sum())

    if scenario_num == 4 or scenario_num == 5:  # Centralized CM
        crm_total = df["λ_cCM"].iloc[0] * df["C_cCM_vol"].iloc[0] * N_TIMESTEPS
    elif scenario_num == 6 or scenario_num == 7:  # Decentralized CM
        crm_total = df["λ_dCM"].iloc[0] * df["C_dCM_vol"].iloc[0] * N_TIMESTEPS

    return {
        "crm_cost_total":   crm_total,             # M€
        "crm_cost_per_mwh": crm_total / demand_mwh if demand_mwh else 0.0
    }

def calculate_all_metrics(scenario_num, variant="ref"):
    """Aggregate all metric groups into one dictionary."""
    df = load_scenario_data(scenario_num, variant)

    results = {
        "scenario":            scenario_num,
        "variant":             variant,
        "price_metrics":       calculate_price_metrics(df),
        "reliability_metrics": calculate_reliability_metrics(df),
        "capacity_metrics":    calculate_capacity_metrics(df, scenario_num),
        "generator_metrics":   calculate_generator_metrics(df, scenario_num),
        "system_cost_metrics": calculate_system_cost_metrics(df, scenario_num),
        "generation_mix_metrics": calculate_generation_mix_metrics(df),
        "consumer_metrics":    calculate_consumer_metrics(df, scenario_num),
        "enhanced_consumer_metrics": calculate_enhanced_consumer_metrics(df, scenario_num),
        "crm_cost_metrics":    calculate_crm_cost_metrics(df, scenario_num),
    }

    # Derived aggregates: €/MWh costs from different perspectives
    results["system_cost_total_mwh"] = results["system_cost_metrics"]["system_cost_per_mwh"]
    
    # Legacy consumer cost per MWh calculation for backward compatibility
    annual_demand = abs(df.filter(regex="^G_(LV_MED|MV_LOAD|LV_HIGH|LV_LOW)").sum(axis=1)).sum()
    total_cons_cost = results["consumer_metrics"]["total"]["total_cost"]
    results["consumer_cost_total_mwh"] = total_cons_cost / annual_demand if annual_demand else np.nan
    
    return results

def compare_scenarios(scenarios=[1, 2, 3, 4, 5, 6, 7], variant="ref"):
    """Compare metrics across multiple scenarios."""
    results = {}
    for scen in scenarios:
        try:
            results[scen] = calculate_all_metrics(scen, variant)
        except Exception as e:
            print(f"Error processing Scenario {scen}: {e}")
    
    return results
