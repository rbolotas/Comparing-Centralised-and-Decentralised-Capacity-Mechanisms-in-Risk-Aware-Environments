#Market Comparison Module
import pandas as pd
import numpy as np
from pathlib import Path
import metrics
import json
import yaml

# Get the project root directory (2 levels up from the Analysis directory)
PROJECT_ROOT = Path(__file__).parent.parent

def get_agent_names_from_config(config_file="../Input/config.yaml"):
    """Get actual agent names from config."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        consumers = list(config.get('Consumers', {}).keys())
        generators = list(config.get('Generators', {}).keys())
        
        return {'consumers': consumers, 'generators': generators}
        
    except Exception as e:
        print(f"⚠️  Could not load agent names from config: {e}")
        return {'consumers': [], 'generators': []}

def scenario_names():
    """Return mapping of scenario numbers to descriptive names."""
    return {
        1: "1. EOM Inelastic",
        2: "2.EOM Elastic", 
        3: "3.EOM El.+ CVAR",
        4: "4.EOM El.+ CM",
        5: "5.EOM El.+ CM+ CVAR",
        6: "6.EOM El.+ RO",
        7: "7.EOM El.+ RO+ CVAR"
    }

def create_scenario_summary_table(metrics_dict):
    """One-row-per-scenario overview with key indicators."""
    scenarios = sorted(metrics_dict.keys())
    summary_data = []

    for scen in scenarios:
        m = metrics_dict[scen]

        row = {
            "Scenario":                       scenario_names().get(scen, f"S{scen}"),
            "Avg Price (€/MWh)":              round(m["price_metrics"]["mean_price"], 2),
            "σ Price":                        round(m["price_metrics"]["price_volatility"], 2),
            "CVaR95 (€/MWh)":                 round(m["price_metrics"]["price_cvar95"], 2),
            "LOLE (h/yr)":                    round(m["reliability_metrics"]["LOLE"], 1),
            "Total ENS (MWh)":                round(m["reliability_metrics"]["total_ENS"], 1),
            "System Cost (€/MWh)":            round(m["system_cost_metrics"]["system_cost_per_mwh"], 2),
            "Energy Cost (€/MWh)":            round(m["system_cost_metrics"]["energy_cost_per_mwh"], 2),
            "Capacity Cost (€/MWh)":          round(m["system_cost_metrics"]["capacity_cost_per_mwh"], 2),
            "Consumer Cost (€/MWh)":          round(m["consumer_cost_total_mwh"], 2) if "consumer_cost_total_mwh" in m else "n/a",
            "CRM Cost (€/MWh)":               round(m["crm_cost_metrics"]["crm_cost_per_mwh"], 2),
        }

        # capacity-specific columns (scenarios 4-7)
        if scen >= 4:
            row.update({
                "Capacity Price (€/kW-yr)": round(m["capacity_metrics"]["capacity_price"], 1),
                "Capacity Volume (GW)":     round(m["capacity_metrics"]["capacity_volume"], 2),
                "Total Dispatchable (GW)": round(m["capacity_metrics"]["total_dispatchable_capacity"], 2),
            })
        else:
            row.update({
                "Capacity Price (€/kW-yr)": "n/a",
                "Capacity Volume (GW)":     "n/a",
                "Total Dispatchable (GW)": round(m["capacity_metrics"]["total_dispatchable_capacity"], 2),
            })

        summary_data.append(row)

    return pd.DataFrame(summary_data)

def create_dispatchable_capacity_comparison(metrics_dict):
    """Compare dispatchable capacity by technology type across scenarios."""
    scenarios = sorted(list(metrics_dict.keys()))
    
    # Focus on dispatchable technologies
    dispatchable_types = ["Baseload", "MidMerit", "Peak"]
    
    # Prepare comparison data
    comparison_data = []
    
    for gen_type in dispatchable_types:
        row = {"Technology": gen_type}
        
        for scen in scenarios:
            scen_name = scenario_names().get(scen, f"Scenario {scen}")
            capacity_by_type = metrics_dict[scen]["capacity_metrics"]["dispatchable_capacity_by_type"]
            
            row.update({
                f"{scen_name} - Capacity (GW)": round(capacity_by_type.get(gen_type, 0), 2),
            })
        
        comparison_data.append(row)
    
    # Add total row
    total_row = {"Technology": "TOTAL DISPATCHABLE"}
    for scen in scenarios:
        scen_name = scenario_names().get(scen, f"Scenario {scen}")
        total_disp = metrics_dict[scen]["capacity_metrics"]["total_dispatchable_capacity"]
        
        total_row.update({
            f"{scen_name} - Capacity (GW)": round(total_disp, 2),
        })
    
    comparison_data.append(total_row)
    
    return pd.DataFrame(comparison_data)

def create_welfare_comparison(metrics_dict):
    """Compare welfare distribution across all agents and scenarios."""
    scenarios = sorted(list(metrics_dict.keys()))
    if not scenarios:
        return pd.DataFrame([])

    # Get all agents from available scenarios
    # Generators: union across scenarios
    all_generators = set()
    for s in scenarios:
        all_generators.update(metrics_dict[s].get("generator_metrics", {}).keys())
    generators = sorted(all_generators)

    # Consumers: derive from any scenario's consumer_metrics keys
    all_consumers = set()
    for s in scenarios:
        all_consumers.update(metrics_dict[s].get("consumer_metrics", {}).keys())
    # remove aggregate 'total' if present
    all_consumers.discard("total")
    consumers = sorted(all_consumers) if all_consumers else ["LV_LOW", "LV_MED", "LV_HIGH", "MV_LOAD"]
    
    # Prepare welfare comparison data
    welfare_data = []
    
    # Generator welfare (positive - revenue)
    for gen in generators:
        row = {"Agent": f"{gen} (Generator)", "Type": "Generator"}
        for s in scenarios:
            scen_name = scenario_names().get(s, f"Scenario {s}")
            total_revenue = metrics_dict[s]["generator_metrics"].get(gen, {}).get("total_revenue", 0)
            row[f"{scen_name} - Welfare (M€)"] = round(total_revenue, 1)
        welfare_data.append(row)
    
    # Consumer welfare (negative - costs)
    for consumer in consumers:
        row = {"Agent": f"{consumer} (Consumer)", "Type": "Consumer"}
        for s in scenarios:
            scen_name = scenario_names().get(s, f"Scenario {s}")
            total_cost = metrics_dict[s]["consumer_metrics"].get(consumer, {}).get("total_cost", 0)
            row[f"{scen_name} - Welfare (M€)"] = round(-total_cost, 1)  # Negative for cost burden
        welfare_data.append(row)
    
    # Add summary rows
    # Total generator welfare
    gen_total_row = {"Agent": "TOTAL GENERATORS", "Type": "Generator"}
    for s in scenarios:
        scen_name = scenario_names().get(s, f"Scenario {s}")
        total_gen_welfare = sum(metrics_dict[s]["generator_metrics"].get(gen, {}).get("total_revenue", 0) for gen in generators)
        gen_total_row[f"{scen_name} - Welfare (M€)"] = round(total_gen_welfare, 1)
    welfare_data.append(gen_total_row)
    
    # Total consumer welfare
    con_total_row = {"Agent": "TOTAL CONSUMERS", "Type": "Consumer"}
    for s in scenarios:
        scen_name = scenario_names().get(s, f"Scenario {s}")
        total_con_welfare = -sum(metrics_dict[s]["consumer_metrics"].get(consumer, {}).get("total_cost", 0) for consumer in consumers)
        con_total_row[f"{scen_name} - Welfare (M€)"] = round(total_con_welfare, 1)
    welfare_data.append(con_total_row)
    
    # Net welfare
    net_welfare_row = {"Agent": "NET WELFARE", "Type": "System"}
    for s in scenarios:
        scen_name = scenario_names().get(s, f"Scenario {s}")
        total_gen_welfare = sum(metrics_dict[s]["generator_metrics"].get(gen, {}).get("total_revenue", 0) for gen in generators)
        total_con_welfare = -sum(metrics_dict[s]["consumer_metrics"].get(consumer, {}).get("total_cost", 0) for consumer in consumers)
        net_welfare = total_gen_welfare + total_con_welfare
        net_welfare_row[f"{scen_name} - Welfare (M€)"] = round(net_welfare, 1)
    welfare_data.append(net_welfare_row)
    
    return pd.DataFrame(welfare_data)

def create_generator_comparison(metrics_dict, scenarios=[4, 5, 6, 7]):
    """Compare generator revenues between scenarios (typically for CM comparison)."""
    # Extract relevant scenarios
    scenario_data = {s: metrics_dict[s] for s in scenarios if s in metrics_dict}
    
    # Get all generator types
    all_generators = set()
    for s, data in scenario_data.items():
        all_generators.update(data["generator_metrics"].keys())
    
    all_generators = sorted(all_generators)
    
    # Prepare comparison data
    comparison_data = []
    
    for gen in all_generators:
        row = {"Generator": gen}
        
        for s in scenarios:
            if s in scenario_data:
                scen_name = scenario_names().get(s, f"Scenario {s}")
                gen_data = scenario_data[s]["generator_metrics"].get(gen, {})
                
                row.update({
                    f"{scen_name} - Energy Revenue (M€)": round(gen_data.get("energy_revenue", 0), 1),
                    f"{scen_name} - Capacity Revenue (M€)": round(gen_data.get("capacity_revenue", 0), 1),
                    f"{scen_name} - Total Revenue (M€)": round(gen_data.get("total_revenue", 0), 1)
                })
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def create_consumer_comparison(metrics_dict, config_file="../Input/config.yaml"):
    """Compare consumer costs across all scenarios."""
    scenarios = sorted(list(metrics_dict.keys()))
    
    # Get consumer types from config
    agent_config = get_agent_names_from_config(config_file)
    consumer_types = agent_config['consumers'] + ["total"]
    
    # Prepare comparison data
    comparison_data = []
    
    for consumer in consumer_types:
        row = {"Consumer": "System Total" if consumer == "total" else consumer}
        
        for scen in scenarios:
            scen_name = scenario_names().get(scen, f"Scenario {scen}")
            consumer_data = metrics_dict[scen]["consumer_metrics"].get(consumer, {})
            
            row.update({
                f"{scen_name} - Energy Cost (M€)": round(consumer_data.get("energy_cost", 0), 1),
                f"{scen_name} - Capacity Cost (M€)": round(consumer_data.get("capacity_cost", 0), 1),
                f"{scen_name} - Total Cost (M€)": round(consumer_data.get("total_cost", 0), 1)
            })
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def create_system_cost_comparison(metrics_dict):
    """Compare system costs across all scenarios from generation perspective."""
    scenarios = sorted(list(metrics_dict.keys()))
    
    # Get all generator types from first scenario
    all_generators = set()
    for s, data in metrics_dict.items():
        all_generators.update(data["system_cost_metrics"]["by_generator"].keys())
    
    all_generators = sorted(all_generators)
    
    # Prepare comparison data
    comparison_data = []
    
    for gen in all_generators:
        row = {"Generator": gen}
        
        for scen in scenarios:
            scen_name = scenario_names().get(scen, f"Scenario {scen}")
            gen_data = metrics_dict[scen]["system_cost_metrics"]["by_generator"].get(gen, {})
            
            row.update({
                f"{scen_name} - Energy Cost (M€)": round(gen_data.get("energy_cost", 0), 1),
                f"{scen_name} - Capacity Cost (M€)": round(gen_data.get("capacity_cost", 0), 1),
                f"{scen_name} - Total System Cost (M€)": round(gen_data.get("total_cost", 0), 1)
            })
        
        comparison_data.append(row)
    
    # Add system totals row
    total_row = {"Generator": "SYSTEM TOTAL"}
    for scen in scenarios:
        scen_name = scenario_names().get(scen, f"Scenario {scen}")
        system_data = metrics_dict[scen]["system_cost_metrics"]
        
        total_row.update({
            f"{scen_name} - Energy Cost (M€)": round(system_data.get("total_energy_cost", 0), 1),
            f"{scen_name} - Capacity Cost (M€)": round(system_data.get("total_capacity_cost", 0), 1),
            f"{scen_name} - Total System Cost (M€)": round(system_data.get("total_system_cost", 0), 1)
        })
    
    comparison_data.append(total_row)
    
    return pd.DataFrame(comparison_data)

def create_enhanced_consumer_comparison(metrics_dict):
    """Compare enhanced consumer metrics including industrial demand across all scenarios."""
    scenarios = sorted(list(metrics_dict.keys()))
    consumer_types = ["LV_MED", "MV_LOAD", "LV_HIGH", "LV_LOW", "total"]
    
    # Prepare comparison data
    comparison_data = []
    
    for consumer in consumer_types:
        row = {"Consumer": "SYSTEM TOTAL" if consumer == "total" else consumer}
        
        for scen in scenarios:
            scen_name = scenario_names().get(scen, f"Scenario {scen}")
            consumer_data = metrics_dict[scen]["enhanced_consumer_metrics"].get(consumer, {})
            
            row.update({
                f"{scen_name} - Energy Cost (M€)": round(consumer_data.get("energy_cost", 0), 1),
                f"{scen_name} - Capacity Cost (M€)": round(consumer_data.get("capacity_cost", 0), 1),
                f"{scen_name} - Total Cost (M€)": round(consumer_data.get("total_cost", 0), 1),
                f"{scen_name} - Industrial Demand (GWh)": round(consumer_data.get("industrial_demand", 0), 1),
                f"{scen_name} - Behind-Meter Solar (GWh)": round(consumer_data.get("behind_meter_solar", 0), 1),
                f"{scen_name} - Net Consumption (GWh)": round(consumer_data.get("net_consumption", 0), 1),
            })
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def create_generation_mix_comparison(metrics_dict):
    """Compare generation mix across all scenarios."""
    scenarios = sorted(list(metrics_dict.keys()))
    
    # Get all generation sources from all scenarios
    all_sources = set()
    for s, data in metrics_dict.items():
        all_sources.update(data["generation_mix_metrics"]["absolute"].keys())
    
    all_sources = sorted(all_sources)
    
    # Prepare comparison data
    comparison_data = []
    
    for source in all_sources:
        row = {"Generation Source": source}
        
        for scen in scenarios:
            scen_name = scenario_names().get(scen, f"Scenario {scen}")
            gen_data = metrics_dict[scen]["generation_mix_metrics"]["absolute"]
            percentage_data = metrics_dict[scen]["generation_mix_metrics"]["percentages"]
            
            row.update({
                f"{scen_name} - Generation (GWh)": round(gen_data.get(source, 0), 1),
                f"{scen_name} - Share (%)": round(percentage_data.get(source, 0), 1),
            })
        
        comparison_data.append(row)
    
    # Add total row
    total_row = {"Generation Source": "TOTAL"}
    for scen in scenarios:
        scen_name = scenario_names().get(scen, f"Scenario {scen}")
        total_gen = metrics_dict[scen]["generation_mix_metrics"]["total_generation"]
        
        total_row.update({
            f"{scen_name} - Generation (GWh)": round(total_gen, 1),
            f"{scen_name} - Share (%)": 100.0,
        })
    
    comparison_data.append(total_row)
    
    return pd.DataFrame(comparison_data)

def calculate_cvar_metrics(scenario_data):
    """Calculate CVaR for price distributions."""
    results = {}
    
    for scen, df in scenario_data.items():
        prices = df["Price"].values
        
        # Calculate CVaR at different confidence levels
        for alpha in [0.05, 0.1, 0.2]:
            cutoff_idx = int(len(prices) * alpha)
            sorted_prices = np.sort(prices)[::-1]  # Sort in descending order
            worst_prices = sorted_prices[:cutoff_idx]
            results[f"Scenario {scen} - CVaR {alpha*100}%"] = np.mean(worst_prices)
    
    return results

def export_results_to_excel(metrics_dict, file_path):
    """Export all analysis results to a structured Excel file."""
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        # 1. Summary table
        summary_df = create_scenario_summary_table(metrics_dict)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # 2. NEW: Dispatchable capacity comparison
        disp_capacity_df = create_dispatchable_capacity_comparison(metrics_dict)
        disp_capacity_df.to_excel(writer, sheet_name='Dispatchable Capacity', index=False)
        
        # 3. NEW: Welfare comparison
        welfare_df = create_welfare_comparison(metrics_dict)
        welfare_df.to_excel(writer, sheet_name='Welfare Distribution', index=False)
        
        # 4. Generator comparison
        gen_df = create_generator_comparison(metrics_dict)
        gen_df.to_excel(writer, sheet_name='Generator Comparison', index=False)
        
        # 5. Consumer comparison
        consumer_df = create_consumer_comparison(metrics_dict)
        consumer_df.to_excel(writer, sheet_name='Consumer Comparison', index=False)
        
        # 6. System cost comparison
        system_cost_df = create_system_cost_comparison(metrics_dict)
        system_cost_df.to_excel(writer, sheet_name='System Cost Comparison', index=False)
        
        # 7. Enhanced consumer comparison
        enhanced_consumer_df = create_enhanced_consumer_comparison(metrics_dict)
        enhanced_consumer_df.to_excel(writer, sheet_name='Enhanced Consumer Comparison', index=False)
        
        # 8. Generation mix comparison
        generation_mix_df = create_generation_mix_comparison(metrics_dict)
        generation_mix_df.to_excel(writer, sheet_name='Generation Mix Comparison', index=False)
        
        # 9. Detailed metrics for each scenario (with better formatting)
        for scen in metrics_dict.keys():
            scen_name = scenario_names().get(scen, f"Scenario {scen}")
            
            # Price metrics
            price_metrics = pd.DataFrame([metrics_dict[scen]["price_metrics"]])
            price_metrics.to_excel(writer, sheet_name=f'S{scen} Details', startrow=1, startcol=0, index=False)
            
            # Capacity metrics if available
            if "capacity_metrics" in metrics_dict[scen]:
                capacity_metrics = pd.DataFrame([metrics_dict[scen]["capacity_metrics"]])
                capacity_metrics.to_excel(writer, sheet_name=f'S{scen} Details', startrow=5, startcol=0, index=False)
            
            # Reliability metrics
            reliability_metrics = pd.DataFrame([metrics_dict[scen]["reliability_metrics"]])
            reliability_metrics.to_excel(writer, sheet_name=f'S{scen} Details', startrow=9, startcol=0, index=False)
            
            # Generator metrics
            if "generator_metrics" in metrics_dict[scen]:
                gen_metrics = metrics_dict[scen]["generator_metrics"]
                gen_df = pd.DataFrame.from_dict(gen_metrics, orient='index').reset_index().rename(columns={'index': 'Generator'})
                gen_df.to_excel(writer, sheet_name=f'S{scen} Details', startrow=13, startcol=0, index=False)
            
            # Consumer metrics
            if "consumer_metrics" in metrics_dict[scen]:
                consumer_metrics = metrics_dict[scen]["consumer_metrics"]
                # Restructure consumer metrics for Excel
                cons_rows = []
                for consumer, metrics in consumer_metrics.items():
                    row = {'Consumer': consumer}
                    row.update(metrics)
                    cons_rows.append(row)
                cons_df = pd.DataFrame(cons_rows)
                cons_df.to_excel(writer, sheet_name=f'S{scen} Details', 
                                 startrow=15 + (len(gen_df) if 'gen_df' in locals() else 0), 
                                 startcol=0, index=False)
    
    # Using context manager (with statement) automatically closes the file
    return file_path

def export_results_to_json(metrics_dict, file_path):
    """Export all analysis results to a JSON file for further processing."""
    # Convert numpy values to Python native types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                           np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return obj
    
    # Process the metrics dictionary recursively
    json_ready = {}
    for scen, data in metrics_dict.items():
        json_ready[str(scen)] = {}
        for category, values in data.items():
            if isinstance(values, dict):
                json_ready[str(scen)][category] = {}
                for k, v in values.items():
                    if isinstance(v, dict):
                        json_ready[str(scen)][category][k] = {}
                        for k2, v2 in v.items():
                            json_ready[str(scen)][category][k][k2] = convert_for_json(v2)
                    else:
                        json_ready[str(scen)][category][k] = convert_for_json(v)
            else:
                json_ready[str(scen)][category] = convert_for_json(values)
    
    # Save to JSON file
    with open(file_path, 'w') as f:
        json.dump(json_ready, f, indent=4)
    
    return file_path
