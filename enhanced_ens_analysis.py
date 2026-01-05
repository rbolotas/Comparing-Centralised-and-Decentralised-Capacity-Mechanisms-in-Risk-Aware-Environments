# Enhanced ENS Analysis Module
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import metrics
from config_utils import get_ntimesteps

def calculate_enhanced_ens_metrics(df):
    """Calculate detailed ENS metrics by consumer type with percentages, using separated components."""
    
    # Identify all ENS columns (original and separated)
    ens_columns = [col for col in df.columns if col.startswith("ENS_") and not col.startswith("ENS_involuntary_")]
    ens_involuntary_columns = [col for col in df.columns if col.startswith("ENS_involuntary_")]
    d_elastic_columns = [col for col in df.columns if col.startswith("D_elastic_")]
    
    consumer_types = [col.replace("ENS_", "") for col in ens_columns]
    
    # Calculate total demand by consumer type
    demand_columns = [f"D_{consumer}" for consumer in consumer_types]
    
    # ENS by consumer type (12-day totals in GWh) - using separated components
    ens_by_consumer = {}
    ens_involuntary_by_consumer = {}
    d_elastic_by_consumer = {}
    demand_by_consumer = {}
    
    for consumer in consumer_types:
        ens_col = f"ENS_{consumer}"
        ens_involuntary_col = f"ENS_involuntary_{consumer}"
        d_elastic_col = f"D_elastic_{consumer}"
        demand_col = f"D_{consumer}"
        
        if ens_col in df.columns:
            ens_by_consumer[consumer] = df[ens_col].sum()  # Original total ENS
        
        if ens_involuntary_col in df.columns:
            ens_involuntary_by_consumer[consumer] = df[ens_involuntary_col].sum()  # True involuntary ENS
        
        if d_elastic_col in df.columns:
            d_elastic_by_consumer[consumer] = df[d_elastic_col].sum()  # Voluntary elastic response
        
        if demand_col in df.columns:
            demand_by_consumer[consumer] = df[demand_col].sum()  # Total demand over 12 days
    
    # Calculate total system metrics
    total_ens = sum(ens_by_consumer.values())
    total_ens_involuntary = sum(ens_involuntary_by_consumer.values())
    total_d_elastic = sum(d_elastic_by_consumer.values())
    total_demand = sum(demand_by_consumer.values())
    
    # Calculate ENS percentages relative to individual consumer demand
    ens_percentage_by_consumer = {}
    ens_involuntary_percentage_by_consumer = {}
    d_elastic_percentage_by_consumer = {}
    
    for consumer in consumer_types:
        consumer_demand = demand_by_consumer.get(consumer, 0)
        consumer_ens = ens_by_consumer.get(consumer, 0)
        consumer_ens_involuntary = ens_involuntary_by_consumer.get(consumer, 0)
        consumer_d_elastic = d_elastic_by_consumer.get(consumer, 0)
        
        if consumer_demand > 0:
            ens_percentage_by_consumer[consumer] = (consumer_ens / consumer_demand) * 100
            ens_involuntary_percentage_by_consumer[consumer] = (consumer_ens_involuntary / consumer_demand) * 100
            d_elastic_percentage_by_consumer[consumer] = (consumer_d_elastic / consumer_demand) * 100
        else:
            ens_percentage_by_consumer[consumer] = 0
            ens_involuntary_percentage_by_consumer[consumer] = 0
            d_elastic_percentage_by_consumer[consumer] = 0
    
    # Calculate LOLE details using separated components
    df_copy = df.copy()
    df_copy["total_ENS_involuntary"] = df_copy[ens_involuntary_columns].sum(axis=1)  # Sum involuntary ENS across all consumers per hour
    df_copy["total_D_elastic"] = df_copy[d_elastic_columns].sum(axis=1)  # Sum elastic response across all consumers per hour
    
    # Get number of timesteps for scaling
    n_timesteps = get_ntimesteps()
    annual_scaling_factor = 8760 / n_timesteps
    
    # Hours with involuntary ENS by consumer (scaled to annual hours)
    lole_involuntary_by_consumer = {}
    lole_elastic_by_consumer = {}
    
    for consumer in consumer_types:
        ens_involuntary_col = f"ENS_involuntary_{consumer}"
        d_elastic_col = f"D_elastic_{consumer}"
        
        if ens_involuntary_col in df.columns:
            hours_with_involuntary_ens = (df[ens_involuntary_col] > 0).sum()
            lole_involuntary_by_consumer[consumer] = hours_with_involuntary_ens * annual_scaling_factor
        
        if d_elastic_col in df.columns:
            hours_with_elastic_response = (df[d_elastic_col] > 0).sum()
            lole_elastic_by_consumer[consumer] = hours_with_elastic_response * annual_scaling_factor
    
    # Hours with any involuntary ENS (scaled to annual hours)
    total_lole_involuntary = (df_copy["total_ENS_involuntary"] > 0).sum() * annual_scaling_factor
    total_lole_elastic = (df_copy["total_D_elastic"] > 0).sum() * annual_scaling_factor
    
    # Percentiles and statistics for separated components
    ens_involuntary_stats = {}
    d_elastic_stats = {}
    
    for consumer in consumer_types:
        ens_involuntary_col = f"ENS_involuntary_{consumer}"
        d_elastic_col = f"D_elastic_{consumer}"
        
        if ens_involuntary_col in df.columns:
            ens_involuntary_values = df[ens_involuntary_col]
            ens_involuntary_stats[consumer] = {
                "min": ens_involuntary_values.min(),
                "max": ens_involuntary_values.max(),
                "mean": ens_involuntary_values.mean(),
                "median": ens_involuntary_values.median(),
                "p95": ens_involuntary_values.quantile(0.95),
                "p99": ens_involuntary_values.quantile(0.99),
                "hours_with_ens": (ens_involuntary_values > 0).sum() * annual_scaling_factor,
                "hours_without_ens": (ens_involuntary_values == 0).sum() * annual_scaling_factor
            }
        
        if d_elastic_col in df.columns:
            d_elastic_values = df[d_elastic_col]
            d_elastic_stats[consumer] = {
                "min": d_elastic_values.min(),
                "max": d_elastic_values.max(),
                "mean": d_elastic_values.mean(),
                "median": d_elastic_values.median(),
                "p95": d_elastic_values.quantile(0.95),
                "p99": d_elastic_values.quantile(0.99),
                "hours_with_response": (d_elastic_values > 0).sum() * annual_scaling_factor,
                "hours_without_response": (d_elastic_values == 0).sum() * annual_scaling_factor
            }
    
    return {
        "ens_by_consumer": ens_by_consumer,
        "ens_involuntary_by_consumer": ens_involuntary_by_consumer,
        "d_elastic_by_consumer": d_elastic_by_consumer,
        "demand_by_consumer": demand_by_consumer,
        "ens_percentage_by_consumer": ens_percentage_by_consumer,
        "ens_involuntary_percentage_by_consumer": ens_involuntary_percentage_by_consumer,
        "d_elastic_percentage_by_consumer": d_elastic_percentage_by_consumer,
        "total_ens": total_ens,
        "total_ens_involuntary": total_ens_involuntary,
        "total_d_elastic": total_d_elastic,
        "total_demand": total_demand,
        "system_ens_percentage": (total_ens / total_demand) * 100 if total_demand > 0 else 0,
        "system_ens_involuntary_percentage": (total_ens_involuntary / total_demand) * 100 if total_demand > 0 else 0,
        "system_d_elastic_percentage": (total_d_elastic / total_demand) * 100 if total_demand > 0 else 0,
        "lole_involuntary_by_consumer": lole_involuntary_by_consumer,
        "lole_elastic_by_consumer": lole_elastic_by_consumer,
        "total_lole_involuntary": total_lole_involuntary,
        "total_lole_elastic": total_lole_elastic,
        "ens_involuntary_statistics": ens_involuntary_stats,
        "d_elastic_statistics": d_elastic_stats
    }

def analyze_ens_for_all_scenarios(scenarios=[1, 2], variant="ref"):
    """Analyze ENS across multiple scenarios."""
    results = {}
    
    for scenario in scenarios:
        try:
            df = metrics.load_scenario_data(scenario, variant)
            results[scenario] = calculate_enhanced_ens_metrics(df)
        except Exception as e:
            print(f"Error analyzing Scenario {scenario}: {e}")
    
    return results

def create_ens_summary_report(ens_results):
    """Create a comprehensive text summary of ENS analysis."""
    
    print("=" * 80)
    print("🔴 ENHANCED ENS ANALYSIS REPORT")
    print("=" * 80)
    
    for scenario, data in ens_results.items():
        print(f"\n📋 SCENARIO {scenario} ANALYSIS")
        print("-" * 50)
        
        # System-level summary
        print(f"🌐 SYSTEM OVERVIEW:")
        print(f"   • Total ENS (12 days): {data['total_ens']:.1f} GWh")
        print(f"   •   └─ Involuntary ENS: {data['total_ens_involuntary']:.1f} GWh ({data['system_ens_involuntary_percentage']:.2f}%)")
        print(f"   •   └─ Elastic Response: {data['total_d_elastic']:.1f} GWh ({data['system_d_elastic_percentage']:.2f}%)")
        print(f"   • Total Demand (12 days): {data['total_demand']:.1f} GWh")
        print(f"   • System ENS Percentage: {data['system_ens_percentage']:.2f}%")
        print(f"   • LOLE (hours with involuntary ENS): {data['total_lole_involuntary']:.1f} hours/year")
        print(f"   • LOLE Percentage: {(data['total_lole_involuntary']/8760)*100:.1f}% of time")
        print(f"   • Hours with elastic response: {data['total_lole_elastic']:.1f} hours/year")
        print(f"   • Elastic response percentage: {(data['total_lole_elastic']/8760)*100:.1f}% of time")
        
        # Consumer breakdown
        print(f"\n👥 CONSUMER BREAKDOWN:")
        for consumer in data['ens_by_consumer'].keys():
            ens_total = data['ens_by_consumer'][consumer]
            ens_involuntary_total = data['ens_involuntary_by_consumer'].get(consumer, 0)
            d_elastic_total = data['d_elastic_by_consumer'].get(consumer, 0)
            demand_total = data['demand_by_consumer'][consumer]
            ens_pct = data['ens_percentage_by_consumer'][consumer]
            ens_involuntary_pct = data['ens_involuntary_percentage_by_consumer'].get(consumer, 0)
            d_elastic_pct = data['d_elastic_percentage_by_consumer'].get(consumer, 0)
            lole_involuntary_hours = data['lole_involuntary_by_consumer'].get(consumer, 0)
            lole_elastic_hours = data['lole_elastic_by_consumer'].get(consumer, 0)
            
            print(f"   • {consumer}:")
            print(f"     - Total ENS: {ens_total:.1f} GWh ({ens_pct:.2f}% of demand)")
            print(f"       └─ Involuntary: {ens_involuntary_total:.1f} GWh ({ens_involuntary_pct:.2f}%)")
            print(f"       └─ Elastic: {d_elastic_total:.1f} GWh ({d_elastic_pct:.2f}%)")
            print(f"     - Demand: {demand_total:.1f} GWh")
            print(f"     - LOLE (involuntary): {lole_involuntary_hours:.1f} hours/year ({(lole_involuntary_hours/8760)*100:.1f}% of time)")
            print(f"     - Hours with elastic response: {lole_elastic_hours:.1f} hours/year ({(lole_elastic_hours/8760)*100:.1f}% of time)")
        
        # Detailed statistics for MV_LOAD (the problematic consumer)
        if 'MV_LOAD' in data['ens_involuntary_statistics']:
            mv_involuntary_stats = data['ens_involuntary_statistics']['MV_LOAD']
            mv_elastic_stats = data['d_elastic_statistics'].get('MV_LOAD', {})
            print(f"\n🎯 MV_LOAD DETAILED ANALYSIS:")
            print(f"   • Involuntary ENS:")
            print(f"     - Hourly Range: {mv_involuntary_stats['min']:.2f} - {mv_involuntary_stats['max']:.2f} GWh/hour")
            print(f"     - Mean Hourly: {mv_involuntary_stats['mean']:.2f} GWh/hour")
            print(f"     - 95th Percentile: {mv_involuntary_stats['p95']:.2f} GWh/hour")
            print(f"     - Hours with involuntary ENS: {mv_involuntary_stats['hours_with_ens']:.1f} hours/year")
            
            if mv_elastic_stats:
                print(f"   • Elastic Response:")
                print(f"     - Hourly Range: {mv_elastic_stats['min']:.2f} - {mv_elastic_stats['max']:.2f} GWh/hour")
                print(f"     - Mean Hourly: {mv_elastic_stats['mean']:.2f} GWh/hour")
                print(f"     - 95th Percentile: {mv_elastic_stats['p95']:.2f} GWh/hour")
                print(f"     - Hours with elastic response: {mv_elastic_stats['hours_with_response']:.1f} hours/year")

# Note: plot_enhanced_ens_visualization is now in visualize.py to avoid duplication

# End of Selection

def plot_consumer_demand_breakdown(scenarios_data, consumer_type="MV_LOAD", save_path=None):
    """Plot reference demand, actual demand, grid interaction, and solar effects."""
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    for scenario, df in scenarios_data.items():
        hours = np.arange(len(df))
        
        # Extract data for the consumer type
        if f"D_{consumer_type}" in df.columns:
            reference_demand = df[f"D_{consumer_type}"]  # Reference demand
            actual_demand = reference_demand - df[f"ENS_{consumer_type}"]  # Served demand
            grid_interaction = df[f"G_{consumer_type}"]  # Net grid consumption (negative = export)
            ens = df[f"ENS_{consumer_type}"]
            
            # Plot 1: Demand components
            axes[0,0].plot(hours[:96], reference_demand[:96], label=f'S{scenario} - Reference Demand', linewidth=2)
            axes[0,0].plot(hours[:96], actual_demand[:96], label=f'S{scenario} - Served Demand', linewidth=2, linestyle='--')
            
            # Plot 2: Grid interaction and ENS
            axes[0,1].plot(hours[:96], grid_interaction[:96], label=f'S{scenario} - Grid Net Consumption', linewidth=2)
            axes[0,1].plot(hours[:96], ens[:96], label=f'S{scenario} - ENS', linewidth=2, color='red')
            
            # Plot 3: Behind-meter solar effect
            # Solar generation = Reference demand - Net grid consumption  
            behind_meter_solar = reference_demand - (-grid_interaction)  # Approximate
            axes[1,0].plot(hours[:96], behind_meter_solar[:96], label=f'S{scenario} - Behind-meter Solar', linewidth=2)
            
            # Plot 4: Summary comparison
            axes[1,1].bar([f'S{scenario}'], [ens.sum()], alpha=0.7, label='Total ENS')
    
    # Format plots
    axes[0,0].set_title(f'{consumer_type} - Demand Analysis', fontsize=14)
    axes[0,0].set_ylabel('Energy (GWh/hour)', fontsize=12)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].set_title(f'{consumer_type} - Grid Interaction & ENS', fontsize=14)
    axes[0,1].set_ylabel('Energy (GWh/hour)', fontsize=12)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].set_title(f'{consumer_type} - Solar Generation Effects', fontsize=14)
    axes[1,0].set_ylabel('Energy (GWh/hour)', fontsize=12)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].set_title(f'{consumer_type} - Total ENS Comparison', fontsize=14)
    axes[1,1].set_ylabel('Total ENS (GWh/12 days)', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.3)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.3)
        
    return fig

def analyze_capacity_market_effectiveness(results, scenarios_data):
    """Compare ENS reduction across market designs."""
    
    energy_only_scenarios = [1, 2, 3]
    capacity_market_scenarios = [4, 5, 6, 7]
    
    print("🔍 CAPACITY MARKET EFFECTIVENESS ANALYSIS")
    print("=" * 60)
    
    # ENS comparison
    for group_name, scenarios in [("Energy-Only", energy_only_scenarios), 
                                  ("Capacity Markets", capacity_market_scenarios)]:
        print(f"\n{group_name} Scenarios:")
        for s in scenarios:
            if s in results:
                total_ens = results[s]["reliability_metrics"]["total_ENS"]
                print(f"  S{s}: {total_ens:.1f} GWh ENS")
    
    # Capacity installation comparison
    print(f"\n📊 CAPACITY INSTALLATION COMPARISON:")
    for s in results:
        total_cap = results[s]["capacity_metrics"]["total_capacity"]
        disp_cap = results[s]["capacity_metrics"]["total_dispatchable_capacity"]
        print(f"  S{s}: {total_cap:.1f} GW total, {disp_cap:.1f} GW dispatchable")

# Main execution function
def run_enhanced_ens_analysis(scenarios=[1, 2], save_plots=True):
    """Run complete enhanced ENS analysis with separated components."""
    
    print("🚀 Starting Enhanced ENS Analysis with Separated Components...")
    
    # Calculate enhanced metrics
    ens_results = analyze_ens_for_all_scenarios(scenarios)
    
    # Create summary report
    create_ens_summary_report(ens_results)
    
    print("✅ Enhanced ENS Analysis with Separated Components Complete!")
    return ens_results

if __name__ == "__main__":
    # Run analysis for scenarios 1 and 2
    results = run_enhanced_ens_analysis([1, 2])
