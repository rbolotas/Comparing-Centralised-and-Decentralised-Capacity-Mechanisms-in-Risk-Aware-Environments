#  Visualization Module
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from config_utils import get_ntimesteps
from matplotlib.gridspec import GridSpec
import os

# Get the project root directory (2 levels up from the Analysis directory)
PROJECT_ROOT = Path(__file__).parent.parent

# Set style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('ggplot')

# Scenario name mapping
SCENARIO_NAMES = {
    1: "1. EOM Inelastic",
    2: "2.EOM Elastic", 
    3: "3.EOM El.+ CVAR",
    4: "4.EOM El.+ CM",
    5: "5.EOM El.+ CM+ CVAR",
    6: "6.EOM El.+ RO",
    7: "7.EOM El.+ RO+ CVAR"
}

# CONSUMER TYPE MAPPING - consistent naming
CONSUMER_TYPE_MAPPING = {
    "LV_MED": "LV Medium",
    "MV_LOAD": "MV Industrial", 
    "LV_HIGH": "LV High",
    "LV_LOW": "LV Low"
}

# IMPROVED COLOR SCHEMES
GENERATION_COLORS = {
    # Conventional plants - earth tones
    'WindOnshore': '#2E8B57',      # Sea Green
    'WindOffshore': '#4682B4',     # Steel Blue  
    'Baseload': '#8B4513',         # Saddle Brown (coal/nuclear)
    'MidMerit': '#FF8C00',         # Dark Orange (gas)
    'Peak': '#DC143C',             # Crimson (peaking plants)
    # Solar - yellow/orange family
    'Solar_LV_MED': '#FFA500',     # Orange
    'Solar_MV_LOAD': '#FF4500',    # Orange Red
    'Solar_LV_HIGH': '#FF6347',    # Tomato
    'Solar_LV_LOW': '#FFD700',     # Gold
}

CONSUMER_COLORS = {
    'LV Low': '#87CEEB',      # Sky Blue
    'LV Medium': '#4169E1',   # Royal Blue
    'LV High': '#191970',     # Midnight Blue
    'MV Industrial': '#8B0000' # Dark Red
}

# Constants for time scaling - read from config file
N_TIMESTEPS = get_ntimesteps()  # Number of timesteps from config.yaml
# NO SCALING - just use absolute values for simulation period

def get_agent_names_from_config(config_file="../Input/config.yaml"):
    """Get actual agent names and their display labels from config."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        consumers = {}
        generators = {}
        
        # Get consumer agents and their display names
        if 'Consumers' in config:
            for agent_name, agent_config in config['Consumers'].items():
                display_name = agent_config.get('display_name', agent_name.replace('_', ' ').title())
                consumers[agent_name] = display_name
                
        # Get generator agents
        if 'Generators' in config:
            for agent_name, agent_config in config['Generators'].items():
                display_name = agent_config.get('display_name', agent_name.replace('_', ' ').title())
                generators[agent_name] = display_name
                
        return {'consumers': consumers, 'generators': generators}
        
    except Exception as e:
        print(f"⚠️  Could not load agent names from config: {e}")
        return {'consumers': {}, 'generators': {}}

def get_consumer_colors():
    """Get color mapping for consumers based on their display names."""
    return {
        'LV Low': '#87CEEB',      # Sky Blue
        'LV Medium': '#4169E1',   # Royal Blue
        'LV High': '#191970',     # Midnight Blue
        'MV Industrial': '#8B0000' # Dark Red
    }

def get_consumer_colors_direct():
    """Get color mapping for consumers based on their raw agent names."""
    return {
        'LV_LOW': '#87CEEB',      # Sky Blue
        'LV_MED': '#4169E1',      # Royal Blue
        'LV_HIGH': '#191970',     # Midnight Blue
        'MV_LOAD': '#8B0000'      # Dark Red
    }

def get_generation_colors():
    """Get color mapping for generation sources."""
    # Get consumer colors for consistent solar colors
    consumer_colors = get_consumer_colors_direct()
    
    return {
        # Conventional plants - earth tones
        'WindOnshore': '#2E8B57',      # Sea Green
        'WindOffshore': '#4682B4',     # Steel Blue  
        'Baseload': '#8B4513',         # Saddle Brown (coal/nuclear)
        'MidMerit': '#FF8C00',         # Dark Orange (gas)
        'Peak': '#DC143C',             # Crimson (peaking plants)
        
        # Solar - use same colors as consumer types for consistency
        'Solar_LV_LOW': consumer_colors['LV_LOW'],      # Sky Blue
        'Solar_LV_MED': consumer_colors['LV_MED'],      # Royal Blue
        'Solar_LV_HIGH': consumer_colors['LV_HIGH'],    # Midnight Blue
        'Solar_MV_LOAD': consumer_colors['MV_LOAD'],    # Dark Red
    }

def load_all_scenarios(scenarios=[1, 2, 3, 4, 5, 6, 7], variant="ref"):
    """Load all scenario data into a dictionary."""
    data = {}
    for scen in scenarios:
        try:
            file_path = Path(f"../Results/Scenario_{scen}_{variant}.csv")
            try:
                data[scen] = pd.read_csv(file_path, delimiter=";")
            except Exception:
                # Fallbacks: for EOM/CM/RO scenarios, try beta_1.0 then beta_0.9
                fallback_variants = []
                if scen in [3, 5, 7]:
                    fallback_variants = ["beta_1.0", "beta_0.9"]
                elif scen in [1, 2]:
                    fallback_variants = ["ref"]
                loaded = False
                for fv in fallback_variants:
                    fp = Path(f"../Results/Scenario_{scen}_{fv}.csv")
                    if fp.exists():
                        data[scen] = pd.read_csv(fp, delimiter=";")
                        loaded = True
                        break
                if not loaded:
                    raise
            try:
                data[scen] = pd.read_csv(file_path, delimiter=";")
            except Exception:
                # Fallbacks: for EOM/CM/RO scenarios, try beta_1.0 then beta_0.9
                fallback_variants = []
                if scen in [3, 5, 7]:
                    fallback_variants = ["beta_1.0", "beta_0.9"]
                elif scen in [1, 2]:
                    fallback_variants = ["ref"]
                loaded = False
                for fv in fallback_variants:
                    fp = Path(f"../Results/Scenario_{scen}_{fv}.csv")
                    if fp.exists():
                        data[scen] = pd.read_csv(fp, delimiter=";")
                        loaded = True
                        break
                if not loaded:
                    raise
        except Exception as e:
            print(f"Could not load Scenario {scen}: {e}")
    return data

def plot_price_duration_curves(scenarios_data, save_path=None):
    """Plot price duration curves for all scenarios in M€/GWh."""
    plt.figure(figsize=(14, 8))
    for scen, df in scenarios_data.items():
        prices = df["Price"].sort_values(ascending=False).values  # M€/GWh
        plt.plot(range(len(prices)), prices, linewidth=2.5, label=SCENARIO_NAMES.get(scen, f"Scenario {scen}"))
    plt.title("Price Duration Curves", fontsize=18, pad=20)
    plt.xlabel("Hours (sorted by price)", fontsize=14)
    plt.ylabel("Price (M€/GWh)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Format y-axis to show more decimal places
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.2)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)
    return plt.gcf()

def plot_capacity_comparison(metrics_dict, save_path=None):
    """Plot capacity metrics comparison across scenarios including scenario 3 as benchmark."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))  # Wider figure
    
    # Include scenario 3 as benchmark for capacity markets (scenarios 3-7)
    cm_scenarios = [s for s in metrics_dict.keys() if s >= 3]
    
    if not cm_scenarios:
        print("No scenarios found for capacity comparison.")
        return None
        
    scenario_names = [SCENARIO_NAMES.get(s, f"Scenario {s}") for s in cm_scenarios]
    
    # Total dispatchable capacity by scenario
    try:
        # Calculate dispatchable capacity (Baseload + MidMerit + Peak)
        dispatchable_capacities = []
        for s in cm_scenarios:
            try:
                capacity_by_type = metrics_dict[s]["capacity_metrics"]["dispatchable_capacity_by_type"]
                dispatchable_cap = (capacity_by_type.get('Baseload', 0) + 
                                  capacity_by_type.get('MidMerit', 0) + 
                                  capacity_by_type.get('Peak', 0))
                dispatchable_capacities.append(dispatchable_cap)
            except KeyError:
                dispatchable_capacities.append(0)
        
        bars1 = ax1.bar(scenario_names, dispatchable_capacities, color='steelblue', alpha=0.7)
        ax1.set_title("Total Dispatchable Capacity", fontsize=16)
        ax1.set_ylabel("Capacity (GW)", fontsize=14)
        ax1.tick_params(axis='x', rotation=45, labelsize=12)
        
        for bar, val in zip(bars1, dispatchable_capacities):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(dispatchable_capacities)*0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=11, weight='bold')
    except:
        ax1.text(0.5, 0.5, "No dispatchable capacity data available", ha='center', va='center', 
                transform=ax1.transAxes, fontsize=14)
    
    # Capacity prices - Only show for scenarios 4-7 (not scenario 3)
    try:
        # Filter scenarios to only include 4-7 for capacity market prices
        cm_price_scenarios = [s for s in cm_scenarios if s >= 4]
        cm_price_names = [SCENARIO_NAMES.get(s, f"Scenario {s}") for s in cm_price_scenarios]
        
        if cm_price_scenarios:
            prices = [metrics_dict[s]["capacity_metrics"].get("capacity_price", 0) for s in cm_price_scenarios]
            volumes = [metrics_dict[s]["capacity_metrics"].get("capacity_volume", 0) for s in cm_price_scenarios]
            market_types = [metrics_dict[s]["capacity_metrics"].get("market_type", "") for s in cm_price_scenarios]
            
            bars2 = ax2.bar(cm_price_names, prices, color='coral', alpha=0.7)
            ax2.set_title("Capacity Market Clearing Price", fontsize=16)
            ax2.set_ylabel("Price (€/kW/year)", fontsize=14)
            ax2.tick_params(axis='x', rotation=45, labelsize=12)
            
            for bar, price, vol, market_type, scen in zip(bars2, prices, volumes, market_types, cm_price_scenarios):
                if price > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(prices)*0.02,
                            f'{price:.1f}€/kW/yr\n{vol:.2f}GW\n{market_type}', 
                            ha='center', va='bottom', fontsize=9)
        else:
            # Empty plot if no capacity market scenarios
            ax2.set_title("Capacity Market Clearing Price", fontsize=16)
            ax2.set_ylabel("Price (€/kW/year)", fontsize=14)
            ax2.text(0.5, 0.5, "No capacity market scenarios", ha='center', va='center',
                    transform=ax2.transAxes, fontsize=14)
    except:
        ax2.text(0.5, 0.5, "No capacity price data available", ha='center', va='center',
                transform=ax2.transAxes, fontsize=14)
    
    plt.tight_layout(pad=3.0)
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.2)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)
        
    return fig

def plot_dispatchable_capacity_comparison(metrics_dict, save_path=None):
    """Plot installed dispatchable capacity (Baseload, MidMerit, Peak) and wind capacity by type across all scenarios."""
    fig = plt.figure(figsize=(16, 10))  # Reduced height
    
    # Create grid layout: 2 rows, 2 columns
    # Top row: ax1 (left), ax2 (right)
    # Bottom row: ax3 (spans full width)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])  # Top left
    ax2 = fig.add_subplot(gs[0, 1])  # Top right
    ax3 = fig.add_subplot(gs[1, :])  # Bottom row, spans both columns
    
    # Include all scenarios for comparison
    scenarios = sorted(list(metrics_dict.keys()))
    scenario_names = [SCENARIO_NAMES.get(s, f"Scenario {s}") for s in scenarios]
    
    # Dispatchable generator types
    dispatchable_types = ['Baseload', 'MidMerit', 'Peak']
    colors = ['#8B4513', '#FF8C00', '#DC143C']  # Brown, Orange, Red
    
    # Top left plot: Stacked bar chart of dispatchable capacity by type
    capacity_data = {}
    for gen_type in dispatchable_types:
        capacity_data[gen_type] = []
        for s in scenarios:
            try:
                capacity_by_type = metrics_dict[s]["capacity_metrics"]["dispatchable_capacity_by_type"]
                capacity_data[gen_type].append(capacity_by_type.get(gen_type, 0))
            except KeyError:
                capacity_data[gen_type].append(0)
    
    index = np.arange(len(scenarios))
    bar_width = 0.6
    bottom = np.zeros(len(scenarios))
    
    for i, (gen_type, color) in enumerate(zip(dispatchable_types, colors)):
        values = capacity_data[gen_type]
        bars = ax1.bar(index, values, bar_width, bottom=bottom, label=gen_type,
                      color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
        bottom += np.array(values)
        
        # Add individual capacity labels for each type
        for j, (bar, val) in enumerate(zip(bars, values)):
            if val > 0.1:  # Only show labels for significant values
                ax1.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}', ha='center', va='center', fontsize=10, weight='bold')
    
    # Add total capacity labels on top
    total_capacities = [sum(capacity_data[gen_type][i] for gen_type in dispatchable_types) 
                       for i in range(len(scenarios))]
    for i, total in enumerate(total_capacities):
        if total > 0:
            ax1.text(i, total + max(total_capacities)*0.02, f'{total:.1f}GW',
                    ha='center', va='bottom', fontsize=11, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax1.set_xlabel('Market Design', fontsize=14)
    ax1.set_ylabel('Installed Capacity (GW)', fontsize=14)
    ax1.set_title('Installed Dispatchable Capacity by Technology', fontsize=16)
    ax1.set_xticks(index)
    ax1.set_xticklabels(scenario_names, rotation=45, fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Top right plot: Total system capacity including wind and solar
    # Get wind capacity from config
    try:
        from config_utils import load_config
        config = load_config()
        wind_onshore_capacity = config['Generators']['WindOnshore']['C']
        wind_offshore_capacity = config['Generators']['WindOffshore']['C']
        total_wind_capacity = wind_onshore_capacity + wind_offshore_capacity
    except:
        # Fallback values if config can't be loaded
        wind_onshore_capacity = 5.0
        wind_offshore_capacity = 5.065
        total_wind_capacity = wind_onshore_capacity + wind_offshore_capacity
    
    # Calculate solar capacity using the same method as print_config_summary.jl
    try:
        from config_utils import load_config
        config = load_config()
        total_consumers = config['General']['totConsumers']
        total_solar_gw = 0.0
        
        for consumer_type in config['Consumers']:
            share = config['Consumers'][consumer_type]['Share']
            pv_cap_per_consumer = config['Consumers'][consumer_type]['PV_cap']  # in GWp
            
            # Calculate solar capacity for this consumer type
            consumers_of_this_type = total_consumers * share
            solar_gw_this_type = consumers_of_this_type * pv_cap_per_consumer
            total_solar_gw += solar_gw_this_type
    except:
        # Fallback solar capacity calculation
        total_consumers = 1100000
        total_solar_gw = 0.0
        consumer_shares = {'LV_LOW': 0.22, 'LV_MED': 0.28, 'LV_HIGH': 0.28, 'MV_LOAD': 0.22}
        pv_caps = {'LV_LOW': 0.000001, 'LV_MED': 0.000003, 'LV_HIGH': 0.000007, 'MV_LOAD': 0.000025}
        
        for consumer_type, share in consumer_shares.items():
            consumers_of_this_type = total_consumers * share
            solar_gw_this_type = consumers_of_this_type * pv_caps[consumer_type]
            total_solar_gw += solar_gw_this_type
    
    # Calculate total system capacity for each scenario
    total_system_capacity = []
    for s in scenarios:
        dispatchable_capacity = sum(capacity_data[gen_type][scenarios.index(s)] 
                                  for gen_type in dispatchable_types)
        total_system = dispatchable_capacity + total_wind_capacity + total_solar_gw
        total_system_capacity.append(total_system)
    
    # Create stacked bar chart for total system capacity
    bars1 = ax2.bar(index, total_capacities, bar_width, label='Dispatchable', 
                    color='lightblue', alpha=0.8, edgecolor='white', linewidth=0.5)
    bars2 = ax2.bar(index, [total_wind_capacity] * len(scenarios), bar_width, 
                    bottom=total_capacities, label='Wind', 
                    color='green', alpha=0.8, edgecolor='white', linewidth=0.5)
    bars3 = ax2.bar(index, [total_solar_gw] * len(scenarios), bar_width, 
                    bottom=[t + total_wind_capacity for t in total_capacities], label='Solar', 
                    color='gold', alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Add labels for each segment
    for i, (dispatchable, wind, solar) in enumerate(zip(total_capacities, [total_wind_capacity] * len(scenarios), [total_solar_gw] * len(scenarios))):
        if dispatchable > 0.1:
            ax2.text(i, dispatchable/2, f'{dispatchable:.1f}', ha='center', va='center', fontsize=9, weight='bold')
        if wind > 0.1:
            ax2.text(i, dispatchable + wind/2, f'{wind:.1f}', ha='center', va='center', fontsize=9, weight='bold')
        if solar > 0.001:  # Solar might be small
            ax2.text(i, dispatchable + wind + solar/2, f'{solar:.3f}', ha='center', va='center', fontsize=9, weight='bold')
    
    # Add total capacity labels exactly like dispatchable capacity subplot
    for i, total in enumerate(total_system_capacity):
        if total > 0:
            ax2.text(i, total + max(total_system_capacity)*0.02, f'{total:.1f}GW',
                    ha='center', va='bottom', fontsize=11, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax2.set_xlabel('Market Design', fontsize=14)
    ax2.set_ylabel('Total System Capacity (GW)', fontsize=14)
    ax2.set_title('Total System Capacity (Dispatchable + Wind + Solar)', fontsize=16)
    ax2.set_xticks(index)
    ax2.set_xticklabels(scenario_names, rotation=45, fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Bottom plot: Capacity factor comparison by generator type (spans full width)
    # Calculate capacity factors for each generator type across scenarios
    cf_data = {}
    for gen_type in dispatchable_types:
        cf_data[gen_type] = []
        for s in scenarios:
            try:
                # Get generation and capacity for this generator type
                gen_mix = metrics_dict[s]["generation_mix_metrics"]["absolute"]
                capacity_by_type = metrics_dict[s]["capacity_metrics"]["dispatchable_capacity_by_type"]
                
                generation = gen_mix.get(gen_type, 0)  # GWh over simulation period
                capacity = capacity_by_type.get(gen_type, 0)  # GW
                
                if capacity > 0:
                    # Convert generation to average power over simulation period
                    n_timesteps = get_ntimesteps()
                    avg_power = generation / (n_timesteps / 24)  # GW average
                    capacity_factor = avg_power / capacity
                    cf_data[gen_type].append(capacity_factor)
                else:
                    cf_data[gen_type].append(0)
            except KeyError:
                cf_data[gen_type].append(0)
    
    # Create grouped bar chart for capacity factors
    x = np.arange(len(scenarios))
    width = 0.25
    
    for i, (gen_type, color) in enumerate(zip(dispatchable_types, colors)):
        cf_values = cf_data[gen_type]
        bars = ax3.bar(x + i*width, cf_values, width, label=gen_type,
                      color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Add capacity factor labels
        for j, (bar, cf) in enumerate(zip(bars, cf_values)):
            if cf > 0.01:  # Only show if capacity factor > 1%
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{cf:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax3.set_xlabel('Market Design', fontsize=14)
    ax3.set_ylabel('Capacity Factor', fontsize=14)
    ax3.set_title('Capacity Factor by Generator Type\n(1.0 = Full utilization)', fontsize=16)
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(scenario_names, rotation=45, fontsize=12)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.0)
    
    plt.tight_layout(pad=3.0)
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.2)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)
        
    return fig

# Note: plot_reliability_comparison removed - duplicate of plot_enhanced_ens_visualization

def plot_generator_revenue(metrics_dict, scenario, save_path=None):
    """Plot generator revenue breakdown for a specific scenario in M€."""
    gen_metrics = metrics_dict[scenario]["generator_metrics"]
    generators = list(gen_metrics.keys())
    
    # Extract revenue components (already in M€)
    energy_revenue = [gen_metrics[g]["energy_revenue"] for g in generators]
    capacity_revenue = [gen_metrics[g]["capacity_revenue"] for g in generators]
    
    # Create larger figure
    plt.figure(figsize=(14, 8))
    bar_width = 0.6
    
    index = np.arange(len(generators))
    bars1 = plt.bar(index, energy_revenue, bar_width, label='Energy Revenue',
                    color='lightgreen', alpha=0.8, edgecolor='white', linewidth=0.5)
    bars2 = plt.bar(index, capacity_revenue, bar_width, bottom=energy_revenue, 
                    label='Capacity Revenue', color='gold', alpha=0.8, 
                    edgecolor='white', linewidth=0.5)
    
    plt.xlabel('Generator Type', fontsize=16)
    plt.ylabel('Revenue (Million €)', fontsize=16)
    plt.title(f'Generator Revenue Breakdown\n{SCENARIO_NAMES.get(scenario, f"Scenario {scenario}")}', 
              fontsize=18, pad=20)
    plt.xticks(index, generators, rotation=45, fontsize=12)
    plt.legend(fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add total revenue labels with better precision
    for i, (e, c) in enumerate(zip(energy_revenue, capacity_revenue)):
        total = e + c
        if total > 0.1:  # Only add label if there's meaningful revenue
            plt.text(i, total + (max(energy_revenue) * 0.02), 
                     f"{total:.1f}M€", ha='center', va='bottom', fontsize=11,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout(pad=3.0)
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.2)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)
        
    return plt.gcf()

def plot_consumer_costs(metrics_dict, save_path=None):
    """Plot consumer costs across scenarios in M€ and M€/GWh."""
    scenarios = sorted(list(metrics_dict.keys()))
    scenario_names = [SCENARIO_NAMES.get(s, f"Scenario {s}") for s in scenarios]

    energy_costs = [metrics_dict[s]["consumer_metrics"]["total"]["energy_cost"] for s in scenarios]
    capacity_costs = [metrics_dict[s]["consumer_metrics"]["total"]["capacity_cost"] for s in scenarios]
    cost_per_mwh = [metrics_dict[s].get("consumer_cost_total_mwh", 0) for s in scenarios]

    plt.figure(figsize=(14, 8))
    bar_width = 0.6
    index = np.arange(len(scenarios))
    bars1 = plt.bar(index, energy_costs, bar_width, label='Energy Costs', color='skyblue', alpha=0.8)
    bars2 = plt.bar(index, capacity_costs, bar_width, bottom=energy_costs, label='Capacity Costs', color='lightcoral', alpha=0.8)

    plt.xlabel('Market Design', fontsize=16)
    plt.ylabel('Consumer Cost (M€)', fontsize=16)
    plt.title('Total Consumer Costs by Market Design', fontsize=18, pad=20)
    plt.xticks(index, scenario_names, rotation=45, fontsize=12)
    plt.legend(fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left')

    for i, (e, c, cp) in enumerate(zip(energy_costs, capacity_costs, cost_per_mwh)):
        total = e + c
        if total > 0:
            # Position label inside the bar
            plt.text(i, total * 0.5, f"{total:.1f} M€\n({cp:.2f} M€/GWh)", ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    plt.tight_layout(pad=3.0)
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.2)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)
    return plt.gcf()

def plot_system_costs(metrics_dict, save_path=None):
    """Plot system costs from generation perspective across scenarios in M€ and M€/GWh."""
    scenarios = sorted(list(metrics_dict.keys()))
    scenario_names = [SCENARIO_NAMES.get(s, f"Scenario {s}") for s in scenarios]

    # Extract total system costs (M€)
    energy_costs = [metrics_dict[s]["system_cost_metrics"]["total_energy_cost"] for s in scenarios]
    capacity_costs = [metrics_dict[s]["system_cost_metrics"]["total_capacity_cost"] for s in scenarios]

    plt.figure(figsize=(14, 8))
    bar_width = 0.6
    index = np.arange(len(scenarios))
    bars1 = plt.bar(index, energy_costs, bar_width, label='Energy Costs', color='steelblue', alpha=0.8)
    bars2 = plt.bar(index, capacity_costs, bar_width, bottom=energy_costs, label='Capacity Costs', color='coral', alpha=0.8)

    plt.xlabel('Market Design', fontsize=16)
    plt.ylabel('System Cost (M€)', fontsize=16)
    plt.title('Total System Costs by Market Design', fontsize=18, pad=20)
    plt.xticks(index, scenario_names, rotation=45, fontsize=12)
    plt.legend(fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add M€/GWh label
    for i, (e, c) in enumerate(zip(energy_costs, capacity_costs)):
        total_cost = e + c
        cost_per_mwh = metrics_dict[scenarios[i]]["system_cost_metrics"]["system_cost_per_mwh"]
        plt.text(i, total_cost + (max(energy_costs) * 0.02), f"{total_cost:.1f} M€\n({cost_per_mwh:.2f} M€/GWh)", ha='center', va='bottom', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout(pad=3.0)
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.2)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)
    return plt.gcf()

def plot_enhanced_consumer_costs(metrics_dict, save_path=None, config_file="../Input/config.yaml"):
    """Plot enhanced consumer costs with direct agent names."""
    
    scenarios = sorted(list(metrics_dict.keys()))
    scenario_names = [SCENARIO_NAMES.get(s, f"Scenario {s}") for s in scenarios]
    
    # Get actual consumer types from data
    first_scenario = list(metrics_dict.keys())[0]
    consumer_types = [ct for ct in metrics_dict[first_scenario]["enhanced_consumer_metrics"].keys() 
                     if ct != "total"]
    
    # Use agent names directly (no confusing mapping)
    consumer_labels = consumer_types
    
    
    # Create figure with more space
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Get colors for the agent names
    consumer_colors = get_consumer_colors_direct()
    colors = [consumer_colors.get(label, '#666666') for label in consumer_labels]
    
    # Prepare data for stacked bar chart
    energy_costs_by_type = {label: [] for label in consumer_labels}
    capacity_costs_by_type = {label: [] for label in consumer_labels}
    
    for s in scenarios:
        for consumer_type in consumer_types:
            # Use agent name directly
            consumer_data = metrics_dict[s]["enhanced_consumer_metrics"][consumer_type]
            energy_costs_by_type[consumer_type].append(consumer_data["energy_cost"])
            capacity_costs_by_type[consumer_type].append(consumer_data["capacity_cost"])
    
    # Create stacked bars
    index = np.arange(len(scenarios))
    bar_width = 0.6
    
    bottom_energy = np.zeros(len(scenarios))
    bottom_capacity = np.zeros(len(scenarios))
    
    for i, (agent_name, color) in enumerate(zip(consumer_labels, colors)):
        # Energy costs
        bars_energy = ax1.bar(index, energy_costs_by_type[agent_name], bar_width, 
               bottom=bottom_energy, label=f'{agent_name} - Energy', 
               color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Add labels for energy costs
        for j, (bar, val) in enumerate(zip(bars_energy, energy_costs_by_type[agent_name])):
            if val > 0.1:  # Only show labels for significant values
                ax1.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}', ha='center', va='center', fontsize=8, weight='bold')
        
        bottom_energy += np.array(energy_costs_by_type[agent_name])
        
        # Capacity costs (if any)
        capacity_values = np.array(capacity_costs_by_type[agent_name])
        if capacity_values.sum() > 0:  # Only plot if there are capacity costs
            bars_capacity = ax1.bar(index, capacity_values, bar_width, 
                   bottom=bottom_energy + bottom_capacity, label=f'{agent_name} - Capacity', 
                   color=color, alpha=0.5, hatch='//', edgecolor='white', linewidth=0.5)
            
            # Add labels for capacity costs
            for j, (bar, val) in enumerate(zip(bars_capacity, capacity_values)):
                if val > 0.1:  # Only show labels for significant values
                    ax1.text(bar.get_x() + bar.get_width()/2, 
                            bar.get_y() + bar.get_height()/2,
                            f'{val:.1f}', ha='center', va='center', fontsize=8, weight='bold')
            
            bottom_capacity += capacity_values
    
    ax1.set_xlabel('Market Design', fontsize=14)
    ax1.set_ylabel(f'Cost (M€/{N_TIMESTEPS//24} days)', fontsize=14)
    ax1.set_title('Consumer Costs by Type and Market Design', fontsize=16)
    ax1.set_xticks(index)
    ax1.set_xticklabels(scenario_names, rotation=45, fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Right plot: Industrial demand vs behind-meter solar
    industrial_demand = [metrics_dict[s]["enhanced_consumer_metrics"]["total"]["industrial_demand"] for s in scenarios]
    behind_meter_solar = [metrics_dict[s]["enhanced_consumer_metrics"]["total"]["behind_meter_solar"] for s in scenarios]
    net_consumption = [metrics_dict[s]["enhanced_consumer_metrics"]["total"]["net_consumption"] for s in scenarios]
    
    bars1 = ax2.bar(index, industrial_demand, bar_width, label='Industrial Demand', 
                    color='lightblue', alpha=0.8)
    bars2 = ax2.bar(index, behind_meter_solar, bar_width, bottom=industrial_demand, 
                    label='Behind-Meter Solar', color='gold', alpha=0.8)
    
    # Add line for net consumption
    line = ax2.plot(index, net_consumption, 'ro-', linewidth=3, markersize=8, 
                   label='Net Grid Consumption', color='red')
    
    ax2.set_xlabel('Market Design', fontsize=14)
    ax2.set_ylabel(f'Energy (GWh/{N_TIMESTEPS//24} days)', fontsize=14)
    ax2.set_title('Demand Components by Market Design', fontsize=16)
    ax2.set_xticks(index)
    ax2.set_xticklabels(scenario_names, rotation=45, fontsize=12)
    ax2.legend(fontsize=12)
    
    # Add value labels
    for i, (ind, solar, net) in enumerate(zip(industrial_demand, behind_meter_solar, net_consumption)):
        total = ind + solar
        ax2.text(i, total + max(industrial_demand)*0.02, f'{total:.1f}', 
                ha='center', va='bottom', fontsize=10)
        ax2.text(i, net - max(industrial_demand)*0.05, f'{net:.1f}', 
                ha='center', va='top', fontsize=10, color='red', weight='bold')
    
    plt.tight_layout(pad=3.0)
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.3)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.3)
        
    return fig

def plot_generation_mix(metrics_dict, save_path=None):
    """Plot generation mix with improved colors and consistent naming."""
    scenarios = sorted(list(metrics_dict.keys()))
    scenario_names = [SCENARIO_NAMES.get(s, f"Scenario {s}") for s in scenarios]
    
    # Create figure with more space
    plt.figure(figsize=(16, 10))
    
    # Prepare data
    generation_sources = []
    for s in scenarios:
        gen_mix = metrics_dict[s]["generation_mix_metrics"]["absolute"]
        if not generation_sources:
            generation_sources = list(gen_mix.keys())
    
    # Create data matrix (sources x scenarios)
    data_matrix = []
    labels = []
    colors = []
    
    generation_colors = get_generation_colors()
    
    for source in generation_sources:
        values = [metrics_dict[s]["generation_mix_metrics"]["absolute"].get(source, 0) for s in scenarios]
        data_matrix.append(values)
        labels.append(source.replace('_', ' '))
        colors.append(generation_colors.get(source, '#666666'))
    
    # Create stacked bar chart
    index = np.arange(len(scenarios))
    bar_width = 0.6
    
    bottom = np.zeros(len(scenarios))
    
    for i, (source_data, label, color) in enumerate(zip(data_matrix, labels, colors)):
        bars = plt.bar(index, source_data, bar_width, bottom=bottom, 
               label=label, color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Add labels for significant values
        for j, (bar, val) in enumerate(zip(bars, source_data)):
            if val > 0.1:  # Only show labels for significant values
                plt.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}', ha='center', va='center', fontsize=9, weight='bold')
        
        bottom += np.array(source_data)
    
    plt.xlabel('Market Design', fontsize=16)
    plt.ylabel(f'Generation (GWh/{N_TIMESTEPS//24} days)', fontsize=16)
    plt.title('Generation Mix by Market Design', fontsize=18, pad=20)
    plt.xticks(index, scenario_names, rotation=45, fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    plt.tight_layout(pad=3.0)
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.3)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.3)
        
    return plt.gcf()

def create_summary_dashboard(metrics_dict, save_path=None):
    """Create a comprehensive dashboard with key metrics."""
    
    scenarios = sorted(list(metrics_dict.keys()))
    scenario_names = [SCENARIO_NAMES.get(s, f"S{s}") for s in scenarios]
    
    # Use constrained_layout for better spacing
    fig = plt.figure(figsize=(16, 20), constrained_layout=True)
    gs = fig.add_gridspec(4, 3)
    
    # 1. Price comparison
    ax1 = fig.add_subplot(gs[0, 0])
    mean_prices = [metrics_dict[s]["price_metrics"]["mean_price"] for s in scenarios]
    bars1 = ax1.bar(scenario_names, mean_prices, color='steelblue', alpha=0.7)
    
    # Add labels to price bars
    for bar, val in zip(bars1, mean_prices):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mean_prices)*0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax1.set_title('Mean Price (M€/GWh)', fontsize=14, pad=30)  # Increased pad
    ax1.tick_params(axis='x', rotation=45, labelsize=10, pad=10)  # Added pad
    ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))  # 3 decimal places
    
    # 2. Price volatility
    ax2 = fig.add_subplot(gs[0, 1])
    price_vol = [metrics_dict[s]["price_metrics"]["price_volatility"] for s in scenarios]
    bars2 = ax2.bar(scenario_names, price_vol, color='orange', alpha=0.7)
    
    # Add labels to volatility bars
    for bar, val in zip(bars2, price_vol):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(price_vol)*0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax2.set_title('Price Volatility (σ)', fontsize=14, pad=30)
    ax2.tick_params(axis='x', rotation=45, labelsize=10, pad=10)
    ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))  # 3 decimal places
    
    # 3. Reliability metrics
    ax3 = fig.add_subplot(gs[0, 2])
    lole_values = [metrics_dict[s]["reliability_metrics"]["LOLE"] for s in scenarios]
    bars3 = ax3.bar(scenario_names, lole_values, color='red', alpha=0.7)
    
    # Add labels to LOLE bars
    for bar, val in zip(bars3, lole_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(lole_values)*0.01,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax3.set_title('LOLE (hours/year)', fontsize=14, pad=30)
    ax3.tick_params(axis='x', rotation=45, labelsize=10, pad=10)
    
    # 4. System costs
    ax4 = fig.add_subplot(gs[1, 0])
    energy_costs = [metrics_dict[s]["system_cost_metrics"]["total_energy_cost"] for s in scenarios]
    capacity_costs = [metrics_dict[s]["system_cost_metrics"]["total_capacity_cost"] for s in scenarios]
    bars4a = ax4.bar(scenario_names, energy_costs, label='Energy', color='lightgreen', alpha=0.8)
    bars4b = ax4.bar(scenario_names, capacity_costs, bottom=energy_costs, 
                     label='Capacity', color='gold', alpha=0.8)
    
    # Add labels to system cost bars
    for i, (energy, capacity) in enumerate(zip(energy_costs, capacity_costs)):
        total = energy + capacity
        if total > 0:
            ax4.text(i, total + max([e + c for e, c in zip(energy_costs, capacity_costs)])*0.01,
                    f'{total:.1f}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax4.set_title('System Costs (M€/12 days)', fontsize=14, pad=30)
    ax4.tick_params(axis='x', rotation=45, labelsize=10, pad=10)
    ax4.legend(fontsize=10)
    
    # 5. Consumer costs
    ax5 = fig.add_subplot(gs[1, 1])
    cons_energy = [metrics_dict[s]["consumer_metrics"]["total"]["energy_cost"] for s in scenarios]
    cons_capacity = [metrics_dict[s]["consumer_metrics"]["total"]["capacity_cost"] for s in scenarios]
    bars5a = ax5.bar(scenario_names, cons_energy, label='Energy', color='skyblue', alpha=0.8)
    bars5b = ax5.bar(scenario_names, cons_capacity, bottom=cons_energy, 
                     label='Capacity', color='lightcoral', alpha=0.8)
    
    # Add labels to consumer cost bars
    for i, (energy, capacity) in enumerate(zip(cons_energy, cons_capacity)):
        total = energy + capacity
        if total > 0:
            ax5.text(i, total + max([e + c for e, c in zip(cons_energy, cons_capacity)])*0.01,
                    f'{total:.1f}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax5.set_title('Consumer Costs (M€/12 days)', fontsize=14, pad=30)
    ax5.tick_params(axis='x', rotation=45, labelsize=10, pad=10)
    ax5.legend(fontsize=10)
    
    # 6. Capacity metrics (for scenarios 4+)
    ax6 = fig.add_subplot(gs[1, 2])
    cm_scenarios = [s for s in scenarios if s >= 4]
    if cm_scenarios:
        cm_names = [SCENARIO_NAMES.get(s, f"S{s}") for s in cm_scenarios]
        cap_prices = [metrics_dict[s]["capacity_metrics"].get("capacity_price", 0) for s in cm_scenarios]
        bars6 = ax6.bar(cm_names, cap_prices, color='coral', alpha=0.7)
        
        # Add labels to capacity price bars
        for bar, val in zip(bars6, cap_prices):
            if val > 0:
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cap_prices)*0.01,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        ax6.set_title('Capacity Price (M€/GW/Ntimesteps)', fontsize=14, pad=30)
        ax6.tick_params(axis='x', rotation=45, labelsize=10, pad=10)
    else:
        ax6.text(0.5, 0.5, 'No Capacity\nMarkets', ha='center', va='center', 
                transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Capacity Price', fontsize=14, pad=30)
    
    # 7-8. Generation mix overview - use bottom row
    ax7 = fig.add_subplot(gs[2:4, :])
    
    # Get generation data but filter out WindOffshore if not active
    gen_sources = []
    gen_data = {}
    
    for s in scenarios:
        gen_mix = metrics_dict[s]["generation_mix_metrics"]["absolute"]
        for source in gen_mix.keys():
            # FIXED: Skip WindOffshore if it has zero output across all scenarios
            if source == "WindOffshore":
                total_output = sum(gen_mix[source] for s in scenarios if source in metrics_dict[s]["generation_mix_metrics"]["absolute"])
                if total_output == 0:
                    continue
            
            if source not in gen_sources:
                gen_sources.append(source)
            if source not in gen_data:
                gen_data[source] = []
            gen_data[source].append(gen_mix[source])  # Raw values in GWh for 12 days
    
    # Create stacked bar chart for generation mix
    index = np.arange(len(scenarios))
    bar_width = 0.6
    
    bottom = np.zeros(len(scenarios))
    generation_colors = get_generation_colors()
    
    for source in gen_sources:
        values = gen_data.get(source, [0] * len(scenarios))
        color = generation_colors.get(source, '#666666')
        label = source.replace('_', ' ')
        
        bars = ax7.bar(index, values, bar_width, bottom=bottom, 
               label=label, color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Add labels for significant values
        for j, (bar, val) in enumerate(zip(bars, values)):
            if val > 0.1:  # Only show labels for significant values
                ax7.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}', ha='center', va='center', fontsize=8, weight='bold')
        
        bottom += np.array(values)
    
    ax7.set_xlabel('Market Design', fontsize=16)
    ax7.set_ylabel(f'Generation (GWh/{N_TIMESTEPS//24} days)', fontsize=16)
    ax7.set_title('Generation Mix by Market Design', fontsize=16, pad=30)  # FIXED: More padding
    ax7.set_xticks(index)
    ax7.set_xticklabels(scenario_names, rotation=45, fontsize=12)
    ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.suptitle('Market Design Summary Dashboard', fontsize=20, y=1.02)  # Move main title up
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.3)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.3)
    
    return fig

def create_generator_revenue_summary(metrics_dict, save_path=None):
    """Create summary plots for generator revenues across all scenarios."""
    
    scenarios = [s for s in metrics_dict.keys() if s >= 4]  # Focus on capacity market scenarios
    if not scenarios:
        print("No capacity market scenarios found for generator revenue comparison.")
        return None
        
    scenario_names = [SCENARIO_NAMES.get(s, f"Scenario {s}") for s in scenarios]
    
    # Get all generators
    generators = list(metrics_dict[scenarios[0]]["generator_metrics"].keys())
    
    # Create subplot for each generator - FIXED: Force 2x2 square layout for 4 agents
    n_gens = len(generators)
    if n_gens == 4:
        rows, cols = 2, 2
        fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    else:
        cols = min(3, n_gens)
    rows = (n_gens + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    if n_gens == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, gen in enumerate(generators):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            continue
            
        # Extract revenue data for this generator across scenarios
        energy_revenues = [metrics_dict[s]["generator_metrics"][gen]["energy_revenue"] for s in scenarios]
        capacity_revenues = [metrics_dict[s]["generator_metrics"][gen]["capacity_revenue"] for s in scenarios]
        
        # Create stacked bar chart
        index = np.arange(len(scenarios))
        bar_width = 0.6
        
        bars1 = ax.bar(index, energy_revenues, bar_width, label='Energy Revenue',
                      color='lightgreen', alpha=0.8, edgecolor='white', linewidth=0.5)
        bars2 = ax.bar(index, capacity_revenues, bar_width, bottom=energy_revenues, 
                      label='Capacity Revenue', color='gold', alpha=0.8, 
                      edgecolor='white', linewidth=0.5)
        
        ax.set_title(f'{gen} Revenue', fontsize=14)
        ax.set_ylabel('Revenue (M€)', fontsize=12)
        ax.set_xticks(index)
        ax.set_xticklabels(scenario_names, rotation=45, fontsize=10)
        
        if i == 0:  # Add legend to first subplot only
            ax.legend(fontsize=10)
    
    # Hide unused subplots
    for i in range(n_gens, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Generator Revenue Comparison Across Capacity Market Scenarios', fontsize=16, y=0.98)
    plt.tight_layout(pad=3.0)
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.3)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.3)
        
    return fig

def plot_load_duration_curve(scenarios_data, save_path=None):
    """Plot load duration curves for all scenarios using total system demand in GW."""
    plt.figure(figsize=(14, 8))
    for scen, df in scenarios_data.items():
        total_load = df["Total_Demand"]  #GW
        load_sorted = total_load.sort_values(ascending=False).values
        plt.plot(range(len(load_sorted)), load_sorted, linewidth=2.5, label=SCENARIO_NAMES.get(scen, f"Scenario {scen}"))
    plt.title("Load Duration Curves (Total System Load)", fontsize=18, pad=20)
    plt.xlabel("Hours (sorted by load)", fontsize=14)
    plt.ylabel("Load (GW)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Format y-axis to show more decimal places
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.2)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)
    return plt.gcf()

def plot_average_hourly_profiles(scenarios_data, save_path=None):
    """Plot average price and load profiles for each hour of the day using total system demand in GW and price in M€/GWh."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    hours = range(24)
    for scen, df in scenarios_data.items():
        df_copy = df.copy()
        df_copy['hour'] = df_copy.index % 24
        avg_prices = df_copy.groupby('hour')['Price'].mean()
        ax1.plot(hours, avg_prices.values, linewidth=2.5, marker='o', label=SCENARIO_NAMES.get(scen, f"Scenario {scen}"))
        avg_load = df_copy.groupby('hour')['Total_Demand'].mean()  # GW
        ax2.plot(hours, avg_load.values, linewidth=2.5, marker='s', label=SCENARIO_NAMES.get(scen, f"Scenario {scen}"))
    ax1.set_title("Average Hourly Price Profile", fontsize=16, pad=15)
    ax1.set_ylabel("Price (M€/GWh)", fontsize=14)
    ax1.set_xticks(hours[::2])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Format y-axis to show more decimal places for price chart
    ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    
    ax2.set_title("Average Hourly Load Profile (Total System)", fontsize=16, pad=15)
    ax2.set_xlabel("Hour of Day", fontsize=14)
    ax2.set_ylabel("Load (GW)", fontsize=14)
    ax2.set_xticks(hours[::2])
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.2)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)
    return fig

def plot_summary_dashboard(metrics_dict, save_path=None):
    """Create a comprehensive summary dashboard with improved layout."""
    # Define scenarios and scenario_names that were missing
    scenarios = sorted(list(metrics_dict.keys()))
    scenario_names = [SCENARIO_NAMES.get(s, f"S{s}") for s in scenarios]
    
    # Make the figure portrait instead of square and give more space
    fig = plt.figure(figsize=(20, 28))  # Taller and wider for portrait layout
    
    # Create a grid with more vertical space between plots
    gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3, 
                         top=0.95, bottom=0.05, left=0.08, right=0.95)
    
    # 1. Price comparison
    ax1 = fig.add_subplot(gs[0, 0])
    mean_prices = [metrics_dict[s]["price_metrics"]["mean_price"] for s in scenarios]
    bars1 = ax1.bar(scenario_names, mean_prices, color='steelblue', alpha=0.7)
    
    # Add labels to price bars
    for bar, val in zip(bars1, mean_prices):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mean_prices)*0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax1.set_title('Mean Price (M€/GWh)', fontsize=14, pad=15)
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))  # 3 decimal places
    
    # 2. Price volatility
    ax2 = fig.add_subplot(gs[0, 1])
    price_vol = [metrics_dict[s]["price_metrics"]["price_volatility"] for s in scenarios]
    bars2 = ax2.bar(scenario_names, price_vol, color='orange', alpha=0.7)
    
    # Add labels to volatility bars
    for bar, val in zip(bars2, price_vol):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(price_vol)*0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax2.set_title('Price Volatility (σ)', fontsize=14, pad=15)
    ax2.tick_params(axis='x', rotation=45, labelsize=10)
    ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))  # 3 decimal places
    
    # 3. Reliability metrics
    ax3 = fig.add_subplot(gs[1, 0])
    lole_values = [metrics_dict[s]["reliability_metrics"]["LOLE"] for s in scenarios]
    bars3 = ax3.bar(scenario_names, lole_values, color='red', alpha=0.7)
    
    # Add labels to LOLE bars
    for bar, val in zip(bars3, lole_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(lole_values)*0.01,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax3.set_title('LOLE (hours/year)', fontsize=14, pad=30)
    ax3.tick_params(axis='x', rotation=45, labelsize=10)
    
    # 4. System costs
    ax4 = fig.add_subplot(gs[1, 1])
    energy_costs = [metrics_dict[s]["system_cost_metrics"]["total_energy_cost"] for s in scenarios]
    capacity_costs = [metrics_dict[s]["system_cost_metrics"]["total_capacity_cost"] for s in scenarios]
    
    bars4a = ax4.bar(scenario_names, energy_costs, label='Energy', color='lightgreen', alpha=0.8)
    bars4b = ax4.bar(scenario_names, capacity_costs, bottom=energy_costs, 
                     label='Capacity', color='gold', alpha=0.8)
    
    # Add labels to system cost bars
    for i, (energy, capacity) in enumerate(zip(energy_costs, capacity_costs)):
        total = energy + capacity
        if total > 0:
            ax4.text(i, total + max([e + c for e, c in zip(energy_costs, capacity_costs)])*0.01,
                    f'{total:.1f}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax4.set_title('System Costs (M€/12 days)', fontsize=14, pad=15)
    ax4.tick_params(axis='x', rotation=45, labelsize=10)
    ax4.legend(fontsize=10)
    
    # 5. Consumer costs
    ax5 = fig.add_subplot(gs[2, 0])
    cons_energy = [metrics_dict[s]["consumer_metrics"]["total"]["energy_cost"] for s in scenarios]
    cons_capacity = [metrics_dict[s]["consumer_metrics"]["total"]["capacity_cost"] for s in scenarios]
    
    bars5a = ax5.bar(scenario_names, cons_energy, label='Energy', color='skyblue', alpha=0.8)
    bars5b = ax5.bar(scenario_names, cons_capacity, bottom=cons_energy, 
                     label='Capacity', color='lightcoral', alpha=0.8)
    
    # Add labels to consumer cost bars
    for i, (energy, capacity) in enumerate(zip(cons_energy, cons_capacity)):
        total = energy + capacity
        if total > 0:
            ax5.text(i, total + max([e + c for e, c in zip(cons_energy, cons_capacity)])*0.01,
                    f'{total:.1f}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax5.set_title('Consumer Costs (M€/12 days)', fontsize=14, pad=15)
    ax5.tick_params(axis='x', rotation=45, labelsize=10)
    ax5.legend(fontsize=10)
    
    # 6. Capacity metrics (for scenarios 4+)
    ax6 = fig.add_subplot(gs[2, 1])
    cm_scenarios = [s for s in scenarios if s >= 4]
    if cm_scenarios:
        cm_names = [SCENARIO_NAMES.get(s, f"S{s}") for s in cm_scenarios]
        cap_prices = [metrics_dict[s]["capacity_metrics"].get("capacity_price", 0) for s in cm_scenarios]
        bars6 = ax6.bar(cm_names, cap_prices, color='coral', alpha=0.7)
        
        # Add labels to capacity price bars
        for bar, val in zip(bars6, cap_prices):
            if val > 0:
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cap_prices)*0.01,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        ax6.set_title('Capacity Price (Raw Data Units)', fontsize=14, pad=15)
        ax6.tick_params(axis='x', rotation=45, labelsize=10)
    else:
        ax6.text(0.5, 0.5, 'No Capacity\nMarkets', ha='center', va='center', 
                transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Capacity Price', fontsize=14, pad=15)
    
    # 7-8. Generation mix overview
    ax7 = fig.add_subplot(gs[3:5, :])
    
    # Get generation data
    gen_sources = []
    gen_data = {}
    
    for s in scenarios:
        gen_mix = metrics_dict[s]["generation_mix_metrics"]["absolute"]
        for source in gen_mix.keys():
            if source not in gen_sources:
                gen_sources.append(source)
            if source not in gen_data:
                gen_data[source] = []
            gen_data[source].append(gen_mix[source])  # Raw values in GWh for 12 days
    
    # Create stacked bar chart for generation mix
    index = np.arange(len(scenarios))
    bar_width = 0.6
    
    bottom = np.zeros(len(scenarios))
    generation_colors = get_generation_colors()
    
    for source in gen_sources:
        values = gen_data.get(source, [0] * len(scenarios))
        color = generation_colors.get(source, '#666666')
        label = source.replace('_', ' ')
        
        bars = ax7.bar(index, values, bar_width, bottom=bottom, 
               label=label, color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Add labels for significant values
        for j, (bar, val) in enumerate(zip(bars, values)):
            if val > 0.1:  # Only show labels for significant values
                ax7.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}', ha='center', va='center', fontsize=8, weight='bold')
        
        bottom += np.array(values)
    
    ax7.set_xlabel('Market Design', fontsize=16)
    ax7.set_ylabel(f'Generation (GWh/{N_TIMESTEPS//24} days)', fontsize=16)
    ax7.set_title('Generation Mix by Market Design', fontsize=16, pad=20)
    ax7.set_xticks(index)
    ax7.set_xticklabels(scenario_names, rotation=45, fontsize=12)
    ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.suptitle('Market Design Comparison Dashboard', fontsize=20, y=0.98)
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.3)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.3)
        
    return fig

def plot_price_boxplot(metrics_dict, save_path=None):
    """Boxplot of EOM prices across scenarios in M€/GWh."""
    scenarios = sorted(list(metrics_dict.keys()))
    scenario_names = [SCENARIO_NAMES.get(s, f"Scenario {s}") for s in scenarios]

    # Gather price data for each scenario (M€/GWh per hour)
    data = [metrics_dict[s]["price_metrics"]["price_data"] for s in scenarios]

    plt.figure(figsize=(14, 8))
    plt.boxplot(data, labels=scenario_names, patch_artist=True, showmeans=True)
    plt.xlabel('Market Design', fontsize=16)
    plt.ylabel('Electricity Price (M€/GWh)', fontsize=16)
    plt.title('Distribution of Hourly Electricity Prices by Market Design', fontsize=18, pad=20)
    plt.grid(True, alpha=0.3)
    
    # Format y-axis to show more decimal places
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))

    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.2)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)

    return plt.gcf()

def plot_enhanced_ens_visualization(ens_results, save_path=None):
    """Create enhanced ENS visualization with proper context."""
    scenarios = sorted(ens_results.keys())
    scenario_names = [SCENARIO_NAMES.get(s, f"Scenario {s}") for s in scenarios]
    
    # Create comprehensive figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Total ENS by Consumer Type
    consumer_types = list(ens_results[scenarios[0]]['ens_by_consumer'].keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    
    x = np.arange(len(scenarios))
    width = 0.2
    
    for i, consumer in enumerate(consumer_types):
        ens_values = [ens_results[s]['ens_by_consumer'][consumer] for s in scenarios]
        ax1.bar(x + i*width, ens_values, width, label=consumer, color=colors[i % len(colors)], alpha=0.8)
    
    ax1.set_xlabel('Scenario', fontsize=14)
    ax1.set_ylabel('Total ENS (GWh/12 days)', fontsize=14)
    ax1.set_title('Total Energy Not Served by Consumer Type\n(Involuntary + Elastic Response over 12-day period)', fontsize=16, pad=20)
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(scenario_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, consumer in enumerate(consumer_types):
        ens_values = [ens_results[s]['ens_by_consumer'][consumer] for s in scenarios]
        for j, val in enumerate(ens_values):
            if val > 0.1:  # Only show if significant
                ax1.text(j + i*width, val + max(ens_values)*0.01, f'{val:.1f}', 
                        ha='center', va='bottom', fontsize=10, rotation=90)
    
    # 2. ENS Percentage relative to Consumer Demand
    for i, consumer in enumerate(consumer_types):
        pct_values = [ens_results[s]['ens_percentage_by_consumer'][consumer] for s in scenarios]
        ax2.bar(x + i*width, pct_values, width, label=consumer, color=colors[i % len(colors)], alpha=0.8)
    
    ax2.set_xlabel('Scenario', fontsize=14)
    ax2.set_ylabel('ENS as % of Consumer Demand', fontsize=14)
    ax2.set_title('Energy Not Served as Percentage of Total Demand\n(Per Consumer Type)', fontsize=16, pad=20)
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(scenario_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add percentage labels
    for i, consumer in enumerate(consumer_types):
        pct_values = [ens_results[s]['ens_percentage_by_consumer'][consumer] for s in scenarios]
        for j, val in enumerate(pct_values):
            if val > 0.1:  # Only show if significant
                ax2.text(j + i*width, val + max(pct_values)*0.01, f'{val:.1f}%', 
                        ha='center', va='bottom', fontsize=10, rotation=90)
    
    # 3. LOLE Analysis - Hours with Involuntary ENS (True Loss of Load)
    for i, consumer in enumerate(consumer_types):
        lole_values = [ens_results[s]['lole_involuntary_by_consumer'].get(consumer, 0) for s in scenarios]
        ax3.bar(x + i*width, lole_values, width, label=consumer, color=colors[i % len(colors)], alpha=0.8)
    
    ax3.set_xlabel('Scenario', fontsize=14)
    ax3.set_ylabel('Hours with Involuntary ENS (hours/year)', fontsize=14)
    ax3.set_title('Loss of Load Expectation (LOLE) by Consumer\n(Only Involuntary Load Shedding - Annualized)', fontsize=16, pad=20)
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels(scenario_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=288, color='red', linestyle='--', alpha=0.7, label='All Hours (288)')
    
    # 4. System-Level ENS Summary
    total_ens = [ens_results[s]['total_ens'] for s in scenarios]
    total_demand = [ens_results[s]['total_demand'] for s in scenarios]
    system_pct = [ens_results[s]['system_ens_percentage'] for s in scenarios]
    
    ax4_twin = ax4.twinx()
    
    bars = ax4.bar(x, total_ens, color='red', alpha=0.7, label='Total ENS (GWh)')
    line = ax4_twin.plot(x, system_pct, 'bo-', linewidth=3, markersize=8, label='ENS %')
    
    ax4.set_xlabel('Scenario', fontsize=14)
    ax4.set_ylabel('Total ENS (GWh/12 days)', fontsize=14, color='red')
    ax4_twin.set_ylabel('ENS as % of Total System Demand', fontsize=14, color='blue')
    ax4.set_title('System-Level Energy Not Served\n(Total and Percentage)', fontsize=16, pad=20)
    ax4.set_xticks(x)
    ax4.set_xticklabels(scenario_names)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (ens, pct, demand) in enumerate(zip(total_ens, system_pct, total_demand)):
        ax4.text(i, ens + max(total_ens)*0.02, f'{ens:.1f} GWh', 
                ha='center', va='bottom', fontsize=11, weight='bold')
        ax4_twin.text(i, pct + max(system_pct)*0.02, f'{pct:.2f}%', 
                     ha='center', va='bottom', fontsize=11, weight='bold', color='blue')
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout(pad=3.0)
    
    # Add main title
    fig.suptitle('🔴 ENHANCED ENS ANALYSIS DASHBOARD\n' + 
                'Involuntary Load Shedding vs. Voluntary Elastic Response', 
                fontsize=20, y=0.98)
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.3, dpi=300)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.3)
    
    return fig

def plot_consumer_timeseries_summary(scenarios_data, config_file="../Input/config.yaml", save_dir="../Analysis/output/plots/"):
    """
    For each consumer, create a portrait summary image with landscape plots for each scenario:
    - Scaled ts_demand (as used in the model)
    - D_EOM_cap (D_EOM_cap_<consumer>)
    - D (D_<consumer>)
    - D_elastic (D_elastic_<consumer>) - voluntary elastic response
    - ENS_involuntary (ENS_involuntary_<consumer>) - true involuntary curtailment (LOLE)
    """
    import os
    if not scenarios_data:
        print("No scenario data provided to plot_consumer_timeseries_summary; skipping.")
        return None
    os.makedirs(save_dir, exist_ok=True)
    # Get consumer names from config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    consumers = list(config['Consumers'].keys())
    tot_consumers = config['General']['totConsumers']
    # For each consumer
    for consumer in consumers:
        fig, axes = plt.subplots(len(scenarios_data), 1, figsize=(16, 4*len(scenarios_data)), sharex=True)
        if len(scenarios_data) == 1:
            axes = [axes]
        for i, (scen, df) in enumerate(sorted(scenarios_data.items())):
            ax = axes[i]
            # Scaled ts_demand
            share = config['Consumers'][consumer]['Share']
            demand_col = config['Consumers'][consumer]['D']
            ts_demand =  df[f"Input_D_{consumer}"]
            ax.plot(ts_demand.values, label="ts_demand", color="black", linewidth=1.5)
            # D_EOM_cap
            if f"D_EOM_cap_{consumer}" in df.columns:
                ax.plot(df[f"D_EOM_cap_{consumer}"].values, label="D_EOM_cap", color="orange", linestyle="--")
            # D (actual demand)
            if f"D_{consumer}" in df.columns:
                ax.plot(df[f"D_{consumer}"].values, label="D (actual)", color="blue")
            # D_elastic (voluntary elastic response)
            if f"D_elastic_{consumer}" in df.columns:
                ax.plot(df[f"D_elastic_{consumer}"].values, label="D_elastic (voluntary)", color="green", linewidth=2)
            # ENS_involuntary (true involuntary curtailment - LOLE)
            if f"ENS_involuntary_{consumer}" in df.columns:
                ax.plot(df[f"ENS_involuntary_{consumer}"].values, label="ENS_involuntary (LOLE)", color="darkred", linewidth=2)
            ax.set_title(f"Scenario {scen}")
            ax.set_ylabel("GW")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Timestep (hour)")
        fig.suptitle(f"Consumer: {consumer}", fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        save_path = os.path.join(save_dir, f"consumer_timeseries_{consumer}.svg")
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.2)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)

def plot_price_timeseries_summary(scenarios_data, save_dir="../Analysis/output/plots/"):
    """
    Create a price timeseries summary image with landscape plots for each scenario:
    - Price (M€/GWh) for every timestep
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with subplots for each scenario
    fig, axes = plt.subplots(len(scenarios_data), 1, figsize=(16, 4*len(scenarios_data)), sharex=True)
    if len(scenarios_data) == 1:
        axes = [axes]
    
    for i, (scen, df) in enumerate(sorted(scenarios_data.items())):
        ax = axes[i]
        
        # Plot price timeseries
        if "Price" in df.columns:
            ax.plot(df["Price"].values, label="Price", color="purple", linewidth=1.5)
        
        ax.set_title(f"Scenario {scen} - {SCENARIO_NAMES.get(scen, f'Scenario {scen}')}")
        ax.set_ylabel("Price (M€/GWh)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Timestep (hour)")
    fig.suptitle("Price Timeseries by Scenario", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = os.path.join(save_dir, "price_timeseries_summary.svg")
    plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.2)
    png_path = str(save_path).replace('.svg', '.png')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

def plot_generator_timeseries_per_scenario(scenarios_data, config_file="../Input/config.yaml", save_dir="../Analysis/output/plots/"):
    """
    For each scenario, create a generation timeseries summary with:
    - Price subplot on top
    - Stacked generation subplot below (all generators stacked together)
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Define the order we want to display generators (Baseload, MidMerit, Peak, Solar, Wind)
    generator_order = ['Baseload', 'MidMerit', 'Peak', 'Solar', 'WindOnshore']
    colors = ['#8B4513', '#FF8C00', '#DC143C', '#FFD700', '#2E8B57']  # Brown, Orange, Red, Gold, Green
    
    # For each scenario
    for scen, df in sorted(scenarios_data.items()):
        # Create figure with 2 subplots (1 price + 1 stacked generation)
        fig, (ax_price, ax_gen) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
        
        # Top subplot: Price
        if "Price" in df.columns:
            ax_price.plot(df.index, df["Price"].values, label="Electricity Price", 
                        color="purple", linewidth=1.5)
        
        ax_price.set_title(f"Electricity Price - Scenario {scen}", fontsize=14)
        ax_price.set_ylabel("Price (M€/GWh)", fontsize=12)
        ax_price.legend(loc="upper right", fontsize=10)
        ax_price.grid(True, alpha=0.3)
        ax_price.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
        
        # Bottom subplot: Stacked generation
        hours = np.arange(len(df))
        bottom = np.zeros(len(df))
        
        # Get actual solar generation from PV_Total column
        solar_generation = np.zeros(len(df))
        if "PV_Total" in df.columns:
            solar_generation = df["PV_Total"].values
        else:
            # Fallback: calculate from individual consumer PV columns
            consumer_types = ['LV_LOW', 'LV_MED', 'LV_HIGH', 'MV_LOAD']
            for consumer_type in consumer_types:
                pv_col = f"PV_{consumer_type}"
                if pv_col in df.columns:
                    solar_generation += df[pv_col].values
        
        # Add solar generation FIRST (at the bottom of the stack)
        if np.any(solar_generation > 0):
            ax_gen.fill_between(hours, bottom, bottom + solar_generation, 
                              color=colors[3], alpha=0.7, label='Solar')
            bottom += solar_generation
        
        # Plot dispatchable generators on top of solar
        for i, gen_type in enumerate(['Baseload', 'MidMerit', 'Peak']):
            gen_col = f"G_{gen_type}"
            if gen_col in df.columns:
                ax_gen.fill_between(hours, bottom, bottom + df[gen_col].values, 
                                  color=colors[i], alpha=0.7, label=gen_type)
                bottom += df[gen_col].values
        
        # Add wind generation
        wind_col = "G_WindOnshore"
        if wind_col in df.columns:
            ax_gen.fill_between(hours, bottom, bottom + df[wind_col].values, 
                              color=colors[4], alpha=0.7, label='WindOnshore')
            bottom += df[wind_col].values
        
        # Calculate total generation for statistics
        total_generation = np.zeros(len(df))
        
        # Add dispatchable generation
        for gen_type in ['Baseload', 'MidMerit', 'Peak']:
            gen_col = f"G_{gen_type}"
            if gen_col in df.columns:
                total_generation += df[gen_col].values
        
        # Add solar generation (actual PV generation)
        total_generation += solar_generation
        
        # Add wind generation
        wind_col = "G_WindOnshore"
        if wind_col in df.columns:
            total_generation += df[wind_col].values
        
        max_val = total_generation.max()
        min_val = total_generation.min()
        avg_val = total_generation.mean()
        
        stats_text = f'Max: {max_val:.1f} GW\nMin: {min_val:.1f} GW\nAvg: {avg_val:.1f} GW'
        ax_gen.text(0.02, 0.98, stats_text, transform=ax_gen.transAxes, 
                   fontsize=10, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Formatting for generation subplot
        ax_gen.set_ylabel('Power (GW)', fontsize=12)
        ax_gen.set_title(f'Scenario {scen} - {SCENARIO_NAMES.get(scen, f"Scenario {scen}")}\nStacked Generation', 
                        fontsize=14, pad=10)
        ax_gen.grid(True, alpha=0.3)
        ax_gen.legend()
        
        # Set x-axis limits
        ax_gen.set_xlim(0, len(df)-1)
        
        # Set x-label for bottom subplot
        ax_gen.set_xlabel("Timestep (hour)", fontsize=12)
        
        fig.suptitle(f"Generation and Price Timeseries - Scenario {scen} ({SCENARIO_NAMES.get(scen, f'Scenario {scen}')})", 
                    fontsize=18, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        save_path = os.path.join(save_dir, f"generator_timeseries_per_scenario_{scen}.svg")
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.2)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)

def plot_generator_timeseries_per_generator(scenarios_data, config_file="../Input/config.yaml", save_dir="../Analysis/output/plots/"):
    """
    For each generator, create a timeseries summary with different scenarios stacked on top of each other:
    - Separate subplots for each generator (WindOnshore, Baseload, MidMerit, Peak)
    - Each subplot shows all scenarios for that generator
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Get generator names from config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    generators = list(config['Generators'].keys())
    
    # Define the order we want to display generators
    generator_order = ['WindOnshore', 'Baseload', 'MidMerit', 'Peak']
    colors = ['#2E8B57', '#8B4513', '#FF8C00', '#DC143C']
    
    # For each generator
    for gen_type in generator_order:
        # Create figure with subplots for each scenario
        scenarios = sorted(list(scenarios_data.keys()))
        fig, axes = plt.subplots(len(scenarios), 1, figsize=(16, 4*len(scenarios)), sharex=True)
        if len(scenarios) == 1:
            axes = [axes]
        
        for i, (scen, df) in enumerate(sorted(scenarios_data.items())):
            ax = axes[i]
            
            # Plot generation for this generator type
            gen_col = f"G_{gen_type}"
            if gen_col in df.columns:
                ax.plot(df.index, df[gen_col].values, label=f"Generation ({gen_type})", 
                       color=colors[generator_order.index(gen_type)], linewidth=1.5)
            
            ax.set_title(f"Scenario {scen} - {SCENARIO_NAMES.get(scen, f'Scenario {scen}')}", fontsize=14)
            ax.set_ylabel("Generation (GW)", fontsize=12)
            ax.legend(loc="upper right", fontsize=10)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel("Timestep (hour)", fontsize=12)
        fig.suptitle(f"Generation Timeseries by Generator: {gen_type}", fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        save_path = os.path.join(save_dir, f"generator_timeseries_per_generator_{gen_type}.svg")
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.2)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)

def plot_residual_load_duration_curves(scenarios_data, save_path=None):
    """Plot residual load duration curves showing sorted dispatchable generation (Baseload + MidMerit + Peak) for all scenarios on one graph."""
    
    # Colors for different scenarios (same as other duration curves)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    plt.figure(figsize=(14, 8))
    
    for i, (scen, df) in enumerate(scenarios_data.items()):
        # Calculate dispatchable generation (Baseload + MidMerit + Peak)
        baseload = df['G_Baseload'].values
        midmerit = df['G_MidMerit'].values
        peak = df['G_Peak'].values
        
        # Sum of dispatchable generation
        dispatchable_total = baseload + midmerit + peak
        
        # Sort the dispatchable generation in descending order
        sorted_dispatchable = np.sort(dispatchable_total)[::-1]
        
        # Create hours array
        hours = np.arange(len(sorted_dispatchable))
        
        # Plot the sorted dispatchable generation as a single line
        plt.plot(hours, sorted_dispatchable, linewidth=2.5, 
                color=colors[i % len(colors)], 
                label=f'{SCENARIO_NAMES.get(scen, f"Scenario {scen}")}')
    
    plt.title("Dispatchable Generation Duration Curves (All Scenarios)", fontsize=18, pad=20)
    plt.xlabel("Hours (sorted by dispatchable generation)", fontsize=14)
    plt.ylabel("Power (GW)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.2)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)
    
    return plt.gcf()

def plot_combined_load_duration_curves(scenarios_data, save_path=None):
    """Plot combined load duration curves for all scenarios."""
    if not scenarios_data:
        print("No scenario data provided to plot_combined_load_duration_curves; skipping.")
        return None
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Dynamically detect available generators
    sample_df = next(iter(scenarios_data.values()))  # Use first available scenario to detect columns
    sample_df = next(iter(scenarios_data.values()))  # Use first available scenario to detect columns
    generator_columns = [col for col in sample_df.columns if col.startswith('G_') and not any(c in col for c in ["LV_MED", "MV_LOAD", "LV_HIGH", "LV_LOW"])]
    generators = [col[2:] for col in generator_columns]  # Remove "G_" prefix
    
    for i, (scenario_num, df) in enumerate(scenarios_data.items()):
        if i >= 4:  # Only plot first 4 scenarios
            break
            
        ax = axes[i]
        
        # Get total demand
        total_demand = df['Total_Demand'].values
        
        # Get generation for available generators
        generation_data = {}
        for gen in generators:
            if f'G_{gen}' in df.columns:
                generation_data[gen] = df[f'G_{gen}'].values
        
        # Sort by total demand for load duration curve
        sorted_indices = np.argsort(total_demand)[::-1]
        sorted_demand = total_demand[sorted_indices]
        
        # Plot demand
        ax.plot(sorted_demand, label='Total Demand', color='black', linewidth=2)
        
        # Plot generation for each available generator
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for j, (gen_name, gen_data) in enumerate(generation_data.items()):
            sorted_gen = gen_data[sorted_indices]
            ax.plot(sorted_gen, label=f'{gen_name}', color=colors[j % len(colors)], linewidth=1.5)
        
        ax.set_xlabel('Hours')
        ax.set_ylabel('Power (GW)')
        ax.set_title(f'Scenario {scenario_num} - Load Duration Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.2)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)
    
    plt.close(fig)
    return fig

# ----------------------- Beta Sweep Plots (Scenarios 3,5,7) -----------------------

def _beta_list():
    import numpy as np
    # Include 1.0 down to 0.1
    return [round(b, 1) for b in np.arange(1.0, 0.0, -0.1)]

def _scenario_label(scen: int) -> str:
    """Human-friendly market design label for known scenarios."""
    if scen == 3:
        return "EOM"
    if scen == 5:
        return "Centralized CM"
    if scen == 7:
        return "Decentralized RO"
    return f"S{scen}"

def _load_variant_df_for_beta(scenario: int, beta: float) -> pd.DataFrame:
    file_path = PROJECT_ROOT / f"Results/Scenario_{scenario}_beta_{beta:.1f}.csv"
    return pd.read_csv(file_path, delimiter=";")

def _capacity_mix_from_df(df: pd.DataFrame) -> pd.Series:
    # Only include installed capacity for generation technologies; ignore CM bookkeeping columns
    cap_cols = [
        c for c in df.columns
        if c.startswith("C_")
        and c != "C_system_total"
        and not c.startswith("C_cCM")
        and not c.startswith("C_dCM")
    ]
    caps = {}
    allowed = {"Baseload", "MidMerit", "Peak", "Wind", "Solar"}
    for c in cap_cols:
        gen = c[2:]
        # Skip any remaining CM-related or non-tech identifiers
        if ("cCM" in gen) or ("dCM" in gen):
            continue
            
        # Map to tech buckets
        tech = None
        if gen in ("Baseload", "MidMerit", "Peak"):
            tech = gen
        elif "Wind" in gen:
            tech = "Wind"
        elif ("PV" in gen) or ("Solar" in gen):
            tech = "Solar"
        
        if tech:
            caps[tech] = caps.get(tech, 0.0) + float(df[c].iloc[0])
    
    # Get solar capacity from config file (same method as print_config_summary.jl)
    try:
        from config_utils import load_config
        config = load_config()
        total_consumers = config['General']['totConsumers']
        total_solar_gw = 0.0
        
        for consumer_type in config['Consumers']:
            share = config['Consumers'][consumer_type]['Share']
            pv_cap_per_consumer = config['Consumers'][consumer_type]['PV_cap']  # in GWp
            
            # Calculate solar capacity for this consumer type
            consumers_of_this_type = total_consumers * share
            solar_gw_this_type = consumers_of_this_type * pv_cap_per_consumer
            total_solar_gw += solar_gw_this_type
        
        caps["Solar"] = total_solar_gw
    except Exception as e:
        print(f"Warning: Could not load solar capacity from config: {e}")
        # Solar capacity will remain 0 if we can't read from config
    
    # Ensure consistent ordering of returned series
    caps_series = pd.Series({k: caps.get(k, 0.0) for k in ["Solar", "Wind", "Baseload", "MidMerit", "Peak"]})
    # Drop zeros-only if all are zero to avoid empty legend issues
    if (caps_series.sum() == 0) and caps:
        return pd.Series(caps)
    return caps_series

def _energy_mix_from_df(df: pd.DataFrame) -> pd.Series:
    """Get dispatchable energy mix only (Baseload, MidMerit, Peak)."""
    gen_cols = [c for c in df.columns if c.startswith("G_") and not any(x in c for x in ["LV_MED", "MV_LOAD", "LV_HIGH", "LV_LOW"])]
    energy = {}
    for c in gen_cols:
        gen = c[2:]
        tech = None  # Initialize to avoid UnboundLocalError
        # Only include dispatchable generators
        if gen in ("Baseload", "MidMerit", "Peak"):
            tech = gen
        
        if tech:
            energy[tech] = energy.get(tech, 0.0) + float(df[c].sum())
    return pd.Series(energy)

def plot_beta_sweep_consumer_cost_delta_combined(scenarios=[3,5,7], betas=None, save_dir="../Analysis/output/plots"):
    """
    Combined plot of consumer cost analysis:
    - 3 Subplots (one per scenario): Absolute Stacked Energy + Capacity Cost.
    - 1 Subplot: RO vs CM Comparison (Delta).
    """
    import matplotlib.pyplot as plt
    from config_utils import get_ntimesteps
    
    os.makedirs(save_dir, exist_ok=True)
    if betas is None:
        betas = _beta_list()
    
    n_timesteps = get_ntimesteps()
    
    # Create 2x2 Figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = axes.flatten()
    
    x = np.arange(len(betas))
    width = 0.6
    
    # Store data to determine common Y-axis limit
    scenario_data = {}
    global_max_y = 0
    
    plot_scenarios = [3, 5, 7]
    
    # 1. Collect Data
    for scen in plot_scenarios:
        energy_costs = []
        capacity_costs = []
        total_costs = []
        
        for beta in betas:
            try:
                df = _load_variant_df_for_beta(scen, beta)
                
                # 1. Calculate Energy Cost (Absolute)
                if "Total_Demand" in df.columns:
                    energy_cost = (df["Price"] * df["Total_Demand"]).sum()
                else:
                    cols = [c for c in df.columns if c.startswith("D_")]
                    if "HV_LOAD" in df.columns: cols.append("HV_LOAD")
                    cols = [c for c in cols if "elastic" not in c and "EOM_cap" not in c]
                    total_d = df[cols].sum(axis=1)
                    energy_cost = (df["Price"] * total_d).sum()
                
                # 2. Calculate Capacity Cost (Absolute)
                cap_cost = 0
                if scen == 5 and "λ_cCM" in df.columns: # Central CM
                    cap_cost = df["λ_cCM"].iloc[0] * df["C_cCM_vol"].iloc[0] * n_timesteps
                elif scen == 7 and "λ_dCM" in df.columns: # Decentral RO
                    cap_cost = df["λ_dCM"].iloc[0] * df["C_dCM_vol"].iloc[0] * n_timesteps
                
                energy_costs.append(energy_cost)
                capacity_costs.append(cap_cost)
                total = energy_cost + cap_cost
                total_costs.append(total)
                
                if total > global_max_y: global_max_y = total
                
            except Exception as e:
                energy_costs.append(0)
                capacity_costs.append(0)
                total_costs.append(0)
        
        scenario_data[scen] = {
            "energy": energy_costs,
            "capacity": capacity_costs,
            "total": total_costs
        }

    # 2. Plot Scenarios
    for i, scen in enumerate(plot_scenarios):
        ax = axes_flat[i]
        data = scenario_data[scen]
        
        # Stacked Bar Plot
        # Bottom bar: Absolute Energy Cost
        bars1 = ax.bar(x, data["energy"], width, label='Energy Cost', color='skyblue', alpha=0.85)
        # Top bar: Absolute Capacity Cost
        bars2 = ax.bar(x, data["capacity"], width, bottom=data["energy"], label='Capacity Cost', color='orange', alpha=0.85)
        
        # Add Labels inside bars (Energy)
        for j, rect in enumerate(bars1):
            height = rect.get_height()
            if height > global_max_y * 0.05: # Only show if bar is tall enough
                ax.text(rect.get_x() + rect.get_width()/2., height/2,
                        f'{height:.0f}',
                        ha='center', va='center', color='white', fontsize=9, weight='bold', zorder=10)

        # Add Labels inside bars (Capacity)
        for j, rect in enumerate(bars2):
            height = rect.get_height()
            y_pos = rect.get_y() + height/2
            if height > global_max_y * 0.05: # Only show if visible
                ax.text(rect.get_x() + rect.get_width()/2., y_pos,
                        f'{height:.0f}',
                        ha='center', va='center', color='white', fontsize=9, weight='bold', zorder=10)
        
        # Add Total labels on top
        for j, val in enumerate(data["total"]):
            ax.text(j, val + global_max_y*0.02, 
                   f'{val:.0f}', ha='center', va='bottom', fontsize=9, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
            
        ax.set_ylim(0, global_max_y * 1.15) # Common scale
        ax.axhline(0, color='black', linewidth=1)
        ax.set_title(f"Scenario {scen} ({_scenario_label(scen)})\nTotal Consumer Cost (Absolute)")
        ax.set_ylabel("M€")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b:.1f}" for b in betas])
        ax.set_xlabel("β (Risk Aversion)")
        if i == 0: ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    # Plot 4: Comparison S7 vs S5 (or RO vs CM) - This remains a Delta plot
    ax4 = axes_flat[3]
    if 5 in plot_scenarios and 7 in plot_scenarios:
        comparison_deltas = []
        for j in range(len(betas)):
            cost5 = scenario_data[5]["total"][j]
            cost7 = scenario_data[7]["total"][j]
            comparison_deltas.append(cost7 - cost5)
        
        ax4.bar(x, comparison_deltas, width, color=['red' if v > 0 else 'green' for v in comparison_deltas], alpha=0.7)
        ax4.axhline(0, color='black', linewidth=1)
        ax4.set_title("Net Consumer Cost Difference: RO (S7) − CM (S5)\n(Equal Beta Comparison | Positive = RO is more expensive)", fontsize=14)
        ax4.set_ylabel("Δ Cost (M€)", fontsize=12)
        ax4.set_xticks(x)
        ax4.set_xticklabels([f"{b:.1f}" for b in betas])
        ax4.set_xlabel("β (Risk Aversion)", fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        for j, val in enumerate(comparison_deltas):
            ax4.text(j, val, f'{val:.1f}', ha='center', va='bottom' if val>0 else 'top', fontsize=9, weight='bold')
            
    else:
        ax4.axis('off')

    plt.suptitle("Consumer Cost Analysis: Absolute Costs and Mechanism Comparison", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "consumer_cost_delta_combined.png", dpi=200)
    plt.close(fig)

def plot_beta_sweep_consumer_cost_ro_vs_cm(scenarios=[5, 7], betas=None, save_dir="../Analysis/output/plots"):
    """
    Plot consumer cost difference per consumer agent: RO vs CM (Equal Beta).
    Clustered bars.
    Includes HV_LOAD if available.
    """
    import matplotlib.pyplot as plt
    from config_utils import get_ntimesteps
    
    os.makedirs(save_dir, exist_ok=True)
    if betas is None:
        betas = _beta_list()
    n_timesteps = get_ntimesteps()
    
    if 5 not in scenarios or 7 not in scenarios:
        print("Skipping RO vs CM consumer cost plot (requires Scenarios 5 and 7)")
        return

    # Define consumers - strict list
    consumer_types = ["LV_LOW", "LV_MED", "LV_HIGH", "MV_LOAD", "HV_LOAD"]
    consumers = consumer_types

    # Colors
    consumer_colors = get_consumer_colors_direct()
    # Add HV_LOAD color
    consumer_colors["HV_LOAD"] = "#800080" # Purple

    fig, ax = plt.subplots(figsize=(14, 8))
    width = 0.15
    x = np.arange(len(betas))
    
    for i, cons in enumerate(consumers):
        deltas = []
        for beta in betas:
            try:
                # Load CM (5)
                df_cm = _load_variant_df_for_beta(5, beta)
                # Load RO (7)
                df_ro = _load_variant_df_for_beta(7, beta)
                
                # Helper to calculate cost for a consumer in a dataframe
                def get_cons_cost(df, scen_type):
                    # Energy Cost
                    if cons == "HV_LOAD":
                        demand = df["HV_LOAD"] if "HV_LOAD" in df.columns else 0
                    else:
                        demand = df[f"D_{cons}"] if f"D_{cons}" in df.columns else 0
                    
                    energy_cost = (df["Price"] * demand).sum()
                    
                    # Capacity Cost
                    cap_cost = 0
                    if scen_type == "CM":
                        # Allocated by energy share
                        total_demand = df["Total_Demand"].sum()
                        cons_demand_sum = demand.sum()
                        share = cons_demand_sum / total_demand if total_demand > 0 else 0
                        total_cm_cost = df["λ_cCM"].iloc[0] * df["C_cCM_vol"].iloc[0] * n_timesteps
                        cap_cost = total_cm_cost * share
                    elif scen_type == "RO":
                        if cons != "HV_LOAD" and f"C_dCM_{cons}" in df.columns:
                             cap_cost = df["λ_dCM"].iloc[0] * df[f"C_dCM_{cons}"].iloc[0] * n_timesteps
                        else:
                             # Fallback allocation for HV or missing columns
                             total_demand = df["Total_Demand"].sum()
                             cons_demand_sum = demand.sum()
                             share = cons_demand_sum / total_demand if total_demand > 0 else 0
                             total_ro_cost = df["λ_dCM"].iloc[0] * df["C_dCM_vol"].iloc[0] * n_timesteps
                             cap_cost = total_ro_cost * share
                             
                    return energy_cost + cap_cost

                cost_cm = get_cons_cost(df_cm, "CM")
                cost_ro = get_cons_cost(df_ro, "RO")
                
                # % Delta: (RO - CM) / CM
                if cost_cm != 0:
                    pct = ((cost_ro - cost_cm) / cost_cm) * 100
                else:
                    pct = 0
                deltas.append(pct)
            except Exception as e:
                deltas.append(0)
        
        # Plot bar
        offset = (i - len(consumers)/2 + 0.5) * width
        ax.bar(x + offset, deltas, width, label=cons, 
               color=consumer_colors.get(cons, "#333333"), alpha=0.85, edgecolor='white')

    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:.1f}" for b in betas])
    ax.set_xlabel("β (Risk Aversion)", fontsize=12)
    ax.set_ylabel("Δ Cost (%): (RO − CM) / CM", fontsize=12)
    ax.set_title("Consumer Cost % Difference: Decentralized RO vs Centralized CM (Equal Beta)\n(Positive = RO is more expensive)", fontsize=14)
    ax.legend(title="Consumer Group", fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add explanatory note regarding mechanism differences
    note_text = (
        "Note on Mechanism Comparison:\n"
        "• RO (Decentralised): Costs are driven by the explicit Willingness-to-Pay (WTP) of the four elastic agents to hedge against scarcity.\n"
        "• CM (Centralised): Capacity demand is centrally determined by the TSO. In the model, this is TSO income to generators;\n"
        "  here, it is allocated post-hoc to all consumers (including HV_LOAD) based on energy share to illustrate system cost incidence."
    )
    plt.figtext(0.5, 0.01, note_text, wrap=True, horizontalalignment='center', fontsize=12, style='italic',
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', boxstyle='round,pad=0.5'))

    plt.tight_layout(rect=[0, 0.15, 1, 1]) # Adjust layout to make room for text at bottom
    plt.savefig(Path(save_dir) / "consumer_cost_RO_vs_CM.png", dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_beta_sweep_consumer_boxplots(scenarios=[3,5,7], betas=None, save_dir="../Analysis/output/plots"):
    import seaborn as sns
    from config_utils import get_ntimesteps
    os.makedirs(save_dir, exist_ok=True)
    if betas is None:
        betas = _beta_list()
    import metrics as _metrics
    import matplotlib.pyplot as plt
    
    n_timesteps = get_ntimesteps()
    
    # Portrait layout: 3 rows (one per scenario), 1 column
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(12, 6 * len(scenarios)), sharex=True)
    if len(scenarios) == 1:
        axes = [axes]
        
    # Get consumer colors
    consumer_colors = get_consumer_colors_direct()
    consumer_colors["HV_LOAD"] = "#800080" # Purple for HV_LOAD
    
    # Define consumers - strict list
    consumer_types = ["LV_LOW", "LV_MED", "LV_HIGH", "MV_LOAD", "HV_LOAD"]

    for ax, scen in zip(axes, scenarios):
        # Store data for each consumer type across betas
        cost_data = {c: [] for c in consumer_types}
        xlabels = []
        
        for beta in betas:
            df = _load_variant_df_for_beta(scen, beta)
            
            # Calculate cost manually to ensure consistency (and include HV_LOAD)
            for c in consumer_types:
                # Energy Cost
                if c == "HV_LOAD":
                    demand = df["HV_LOAD"] if "HV_LOAD" in df.columns else 0
                else:
                    demand = df[f"D_{c}"] if f"D_{c}" in df.columns else 0
                
                if isinstance(demand, (pd.Series, np.ndarray)) or isinstance(demand, float):
                    energy_cost = (df["Price"] * demand).sum()
                else:
                    energy_cost = 0
                
                # Capacity Cost
                cap_cost = 0
                if scen == 5 and "λ_cCM" in df.columns: # Central CM
                    total_demand = df["Total_Demand"].sum()
                    cons_demand_sum = demand.sum() if isinstance(demand, (pd.Series, np.ndarray)) else demand
                    share = cons_demand_sum / total_demand if total_demand > 0 else 0
                    total_cm_cost = df["λ_cCM"].iloc[0] * df["C_cCM_vol"].iloc[0] * n_timesteps
                    cap_cost = total_cm_cost * share
                elif scen == 7 and "λ_dCM" in df.columns: # Decentral RO
                    # Allocate by share for consistency
                    total_demand = df["Total_Demand"].sum()
                    cons_demand_sum = demand.sum() if isinstance(demand, (pd.Series, np.ndarray)) else demand
                    share = cons_demand_sum / total_demand if total_demand > 0 else 0
                    total_ro_cost = df["λ_dCM"].iloc[0] * df["C_dCM_vol"].iloc[0] * n_timesteps
                    cap_cost = total_ro_cost * share
                
                cost_data[c].append(energy_cost + cap_cost)
                    
            xlabels.append(f"{beta:.1f}")
            
        x = np.arange(len(betas))
        bottom = np.zeros(len(betas))
        
        # Plot stacked bars for each consumer type
        for c in consumer_types:
            vals = cost_data[c]
            color = consumer_colors.get(c, '#666666')
            ax.bar(x, vals, bottom=bottom, label=c, color=color, alpha=0.85, edgecolor='white', linewidth=1)
            bottom += np.array(vals)
        
        # Add total value labels on bars
        totals = bottom
        for i, total in enumerate(totals):
             ax.text(i, total + max(totals) * 0.02,
                   f'{total:.1f}', ha='center', va='bottom', fontsize=9, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))
        
        # Add individual segment labels
        cumulative = np.zeros(len(betas))
        for c in consumer_types:
            vals = cost_data[c]
            for i, val in enumerate(vals):
                if val > max(totals) * 0.05:  # Only label if segment is > 5% of total height
                    ax.text(i, cumulative[i] + val/2, f'{val:.1f}', 
                           ha='center', va='center', fontsize=8, color='white', weight='bold', zorder=10)
            cumulative += np.array(vals)

        ax.set_ylabel("Cost (M€ over 12 days)", fontsize=12)
        
        # Set y-axis limit to provide more room for labels and prevent overlap with title
        if len(totals) > 0:
            ax.set_ylim(0, max(totals) * 1.15)  # Add 15% headroom
        
        ax.set_title(
            f"Scenario {scen} ({_scenario_label(scen)})\n"
            f"Total Consumer Cost by Risk Aversion",
            fontsize=14
        )
        # Place legend in upper left
        ax.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1.05, 1), title="Consumer Type")
        ax.grid(True, alpha=0.3, axis='y')
        
    # Set x-axis labels only on bottom subplot
    axes[-1].set_xticks(np.arange(len(betas)))
    axes[-1].set_xticklabels(xlabels, rotation=45)
    axes[-1].set_xlabel("β (Risk Aversion)", fontsize=12)
    
    # Add explanatory note regarding mechanism differences
    note_text = (
        "Note on Mechanism Comparison:\n"
        "• RO (Decentralised): Costs are driven by the explicit Willingness-to-Pay (WTP) of the four elastic agents to hedge against scarcity.\n"
        "• CM (Centralised): Capacity demand is centrally determined by the TSO. In the model, this is TSO income to generators;\n"
        "  here, it is allocated post-hoc to all consumers (including HV_LOAD) based on energy share to illustrate system cost incidence."
    )
    plt.figtext(0.5, 0.01, note_text, wrap=True, horizontalalignment='center', fontsize=14, style='italic',
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', boxstyle='round,pad=0.5'))

    plt.tight_layout(rect=[0, 0.12, 1, 1]) # Adjust layout to make room for text at bottom
    fig.savefig(Path(save_dir) / f"consumer_costs_stacked_by_beta.png", dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_beta_sweep_capacity_and_energy(scenarios=[3,5,7], betas=None, save_dir="../Analysis/output/plots"):
    os.makedirs(save_dir, exist_ok=True)
    if betas is None:
        betas = _beta_list()
    import matplotlib.pyplot as plt
    import pandas as pd

    # Load all scenarios at once
    scenario_data = {}  # {scen: {'cap': df, 'disp': df, 'energy': df}}
    
    for scen in scenarios:
        cap_df_list = []
        disp_df_list = []
        energy_df_list = []
        for beta in betas:
            df = _load_variant_df_for_beta(scen, beta)
            cmix = _capacity_mix_from_df(df)
            emix = _energy_mix_from_df(df)
            disp = cmix.reindex(["Baseload", "MidMerit", "Peak"]).fillna(0.0)
            cmix.name = f"{beta:.1f}"
            disp.name = f"{beta:.1f}"
            emix.name = f"{beta:.1f}"
            cap_df_list.append(cmix)
            disp_df_list.append(disp)
            energy_df_list.append(emix)
        
        scenario_data[scen] = {
            'cap': pd.DataFrame(cap_df_list).T.fillna(0.0),
            'disp': pd.DataFrame(disp_df_list).T.fillna(0.0),
            'energy': pd.DataFrame(energy_df_list).T.fillna(0.0)
        }

    # Define consistent colors for technologies across all plots
    TECH_COLORS = {
        'Solar': '#FFD700',      # Gold
        'Wind': '#2E8B57',       # Sea Green
        'Baseload': '#8B4513',   # Saddle Brown
        'MidMerit': '#FF8C00',   # Dark Orange
        'Peak': '#DC143C'        # Crimson
    }

    # Plot absolute values for each scenario
    for scen in scenarios:
        cap_df = scenario_data[scen]['cap']
        disp_df = scenario_data[scen]['disp']
        energy_df = scenario_data[scen]['energy']
        
        # Capacity mix with consistent colors and labels
        colors_cap = [TECH_COLORS.get(t, '#666666') for t in cap_df.index]
        fig_c, ax_c = plt.subplots(figsize=(12, 7))
        cap_df.T.plot(kind="bar", stacked=True, ax=ax_c, color=colors_cap, 
                     alpha=0.85, edgecolor='white', linewidth=1)
        
        # Add capacity value labels for each technology segment
        x_positions = np.arange(len(cap_df.columns))
        cumulative = np.zeros(len(cap_df.columns))
        for tech_idx, tech in enumerate(cap_df.index):
            tech_values = cap_df.loc[tech].values
            # Add text labels showing capacity values at the center of each segment
            for i, val in enumerate(tech_values):
                if val > 0.5:  # Only show label if segment is large enough (> 0.5 GW)
                    y_position = cumulative[i] + val / 2
                    ax_c.text(i, y_position, f'{val:.1f}', 
                             ha='center', va='center', fontsize=9, weight='bold', 
                             color='white', zorder=10)
            cumulative += tech_values
        
        # Add total capacity labels
        totals = cap_df.sum(axis=0)
        for i, (beta_val, total) in enumerate(zip(cap_df.columns, totals)):
            ax_c.text(i, total + max(totals) * 0.02, f'{total:.1f}', 
                     ha='center', va='bottom', fontsize=9, weight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        ax_c.set_title(f"Scenario {scen} ({_scenario_label(scen)})\nInstalled Capacity Mix by Risk Aversion", fontsize=14)
        ax_c.set_xlabel("β (Risk Aversion)", fontsize=12)
        ax_c.set_ylabel("Capacity (GW)", fontsize=12)
        ax_c.legend(title="Technology", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax_c.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(Path(save_dir) / f"capacity_mix_scen{scen}.png", dpi=200, bbox_inches='tight')
        plt.close(fig_c)

        # Dispatchable capacity with consistent colors and labels
        colors_disp = [TECH_COLORS[t] for t in ['Baseload', 'MidMerit', 'Peak']]
        fig_d, ax_d = plt.subplots(figsize=(12, 7))
        disp_df.T.plot(kind="bar", stacked=True, ax=ax_d, color=colors_disp,
                      alpha=0.85, edgecolor='white', linewidth=1)
        
        # Add capacity value labels for each technology segment
        x_positions = np.arange(len(disp_df.columns))
        cumulative = np.zeros(len(disp_df.columns))
        for tech_idx, tech in enumerate(disp_df.index):
            tech_values = disp_df.loc[tech].values
            # Add text labels showing capacity values at the center of each segment
            for i, val in enumerate(tech_values):
                if val > 0.5:  # Only show label if segment is large enough (> 0.5 GW)
                    y_position = cumulative[i] + val / 2
                    ax_d.text(i, y_position, f'{val:.1f}', 
                             ha='center', va='center', fontsize=9, weight='bold', 
                             color='white', zorder=10)
            cumulative += tech_values
        
        # Add total dispatchable capacity labels
        totals_disp = disp_df.sum(axis=0)
        for i, (beta_val, total) in enumerate(zip(disp_df.columns, totals_disp)):
            ax_d.text(i, total + max(totals_disp) * 0.02, f'{total:.1f}', 
                     ha='center', va='bottom', fontsize=9, weight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        ax_d.set_title(f"Scenario {scen} ({_scenario_label(scen)})\nDispatchable Capacity by Risk Aversion", fontsize=14)
        ax_d.set_xlabel("β (Risk Aversion)", fontsize=12)
        ax_d.set_ylabel("Capacity (GW)", fontsize=12)
        ax_d.legend(title="Dispatchable", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax_d.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(Path(save_dir) / f"dispatchable_capacity_scen{scen}.png", dpi=200, bbox_inches='tight')
        plt.close(fig_d)

        # Dispatchable energy mix with consistent colors and labels
        colors_energy = [TECH_COLORS[t] for t in ['Baseload', 'MidMerit', 'Peak']]
        fig_e, ax_e = plt.subplots(figsize=(12, 7))
        energy_df.T.plot(kind="bar", stacked=True, ax=ax_e, color=colors_energy,
                        alpha=0.85, edgecolor='white', linewidth=1)
        
        # Add energy value labels for each technology segment
        x_positions = np.arange(len(energy_df.columns))
        cumulative = np.zeros(len(energy_df.columns))
        for tech_idx, tech in enumerate(energy_df.index):
            tech_values = energy_df.loc[tech].values
            # Add text labels showing energy values at the center of each segment
            for i, val in enumerate(tech_values):
                if val > 50:  # Only show label if segment is large enough (> 50 GWh)
                    y_position = cumulative[i] + val / 2
                    ax_e.text(i, y_position, f'{val:.0f}', 
                             ha='center', va='center', fontsize=9, weight='bold', 
                             color='white', zorder=10)
            cumulative += tech_values
        
        # Add total energy labels
        totals_energy = energy_df.sum(axis=0)
        for i, (beta_val, total) in enumerate(zip(energy_df.columns, totals_energy)):
            ax_e.text(i, total + max(totals_energy) * 0.02, f'{total:.1f}', 
                     ha='center', va='bottom', fontsize=9, weight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        ax_e.set_title(f"Scenario {scen} ({_scenario_label(scen)})\nDispatchable Energy Production by Risk Aversion", fontsize=14)
        ax_e.set_xlabel("β (Risk Aversion)", fontsize=12)
        ax_e.set_ylabel("Energy (GWh over horizon)", fontsize=12)
        ax_e.legend(title="Dispatchable", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax_e.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(Path(save_dir) / f"energy_mix_scen{scen}.png", dpi=200, bbox_inches='tight')
        plt.close(fig_e)

    # Plot deltas vs EOM (Scenario 3) for scenarios 5 and 7
    if 3 in scenario_data:
        try:
            # Get EOM base case (beta=1.0)
            # Need to make sure "1.0" exists in columns
            eom_col = "1.0" if "1.0" in scenario_data[3]['cap'].columns else scenario_data[3]['cap'].columns[0]
            
            eom_cap_base = scenario_data[3]['cap'][eom_col]
            eom_disp_base = scenario_data[3]['disp'][eom_col]
            eom_energy_base = scenario_data[3]['energy'][eom_col]
            
            for scen in [5, 7]:
                if scen not in scenario_data:
                    continue
                
                try:
                    # Subtract base case (broadcasting across all beta columns)
                    delta_cap = scenario_data[scen]['cap'].subtract(eom_cap_base, axis=0)
                    delta_disp = scenario_data[scen]['disp'].subtract(eom_disp_base, axis=0)
                    delta_energy = scenario_data[scen]['energy'].subtract(eom_energy_base, axis=0)
                    
                    # Plot dispatchable energy mix delta with consistent colors and labels
                    colors_energy = [TECH_COLORS[t] for t in ['Baseload', 'MidMerit', 'Peak']]
                    fig_e_d, ax_e_d = plt.subplots(figsize=(12, 7))
                    delta_energy.T.plot(kind="bar", stacked=True, ax=ax_e_d, color=colors_energy,
                                       alpha=0.85, edgecolor='white', linewidth=1)
                    
                    # Add value labels for each technology segment (white text, no box)
                    pos_cumulative = np.zeros(len(delta_energy.columns))
                    neg_cumulative = np.zeros(len(delta_energy.columns))
                    
                    for tech in delta_energy.index:
                        tech_values = delta_energy.loc[tech].values
                        for i, val in enumerate(tech_values):
                            # Calculate position and update cumulative
                            if val >= 0:
                                y_pos = pos_cumulative[i] + val / 2
                                pos_cumulative[i] += val
                            else:
                                y_pos = neg_cumulative[i] + val / 2
                                neg_cumulative[i] += val
                                
                            if abs(val) > 50:  # Only show label if segment is large enough (> 50 GWh)
                                ax_e_d.text(i, y_pos, f'{val:.0f}', 
                                           ha='center', va='center', fontsize=9, weight='bold', 
                                           color='white', zorder=10)

                    # Add total delta labels
                    totals_e = delta_energy.sum(axis=0)
                    for i, (beta_val, total) in enumerate(zip(delta_energy.columns, totals_e)):
                        if abs(total) > max(abs(totals_e)) * 0.05:
                            ax_e_d.text(i, total + (max(totals_e) - min(totals_e)) * 0.02,
                                       f'{total:.1f}', ha='center', 
                                       va='bottom' if total > 0 else 'top',
                                       fontsize=9, weight='bold',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
                    
                    ax_e_d.axhline(0, color='black', linewidth=1.2, linestyle='-', alpha=0.6)
                    ax_e_d.set_title(f"Scenario {scen} ({_scenario_label(scen)})\nDispatchable Energy Change vs EOM Base Case", fontsize=14)
                    ax_e_d.set_xlabel("β (Risk Aversion)", fontsize=12)
                    ax_e_d.set_ylabel("Δ Energy (GWh vs EOM Base Case)", fontsize=12)
                    ax_e_d.legend(title="Technology", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
                    ax_e_d.grid(True, alpha=0.3, axis='y')
                    plt.tight_layout()
                    plt.savefig(Path(save_dir) / f"energy_mix_delta_vs_eom_scen{scen}.png", dpi=200, bbox_inches='tight')
                    plt.close(fig_e_d)
                    
                    # Plot dispatchable capacity delta with consistent colors and labels
                    colors_disp = [TECH_COLORS[t] for t in ['Baseload', 'MidMerit', 'Peak']]
                    fig_c_d, ax_c_d = plt.subplots(figsize=(12, 7))
                    delta_disp.T.plot(kind="bar", stacked=True, ax=ax_c_d, color=colors_disp,
                                     alpha=0.85, edgecolor='white', linewidth=1)
                    
                    # Add value labels for each technology segment (white text, no box)
                    pos_cumulative = np.zeros(len(delta_disp.columns))
                    neg_cumulative = np.zeros(len(delta_disp.columns))
                    
                    for tech in delta_disp.index:
                        tech_values = delta_disp.loc[tech].values
                        for i, val in enumerate(tech_values):
                            # Calculate position and update cumulative
                            if val >= 0:
                                y_pos = pos_cumulative[i] + val / 2
                                pos_cumulative[i] += val
                            else:
                                y_pos = neg_cumulative[i] + val / 2
                                neg_cumulative[i] += val
                                
                            if abs(val) > 0.2:  # Only show label if segment is large enough (> 0.2 GW)
                                ax_c_d.text(i, y_pos, f'{val:.1f}', 
                                           ha='center', va='center', fontsize=9, weight='bold', 
                                           color='white', zorder=10)

                    # Add total delta labels
                    totals_c = delta_disp.sum(axis=0)
                    for i, (beta_val, total) in enumerate(zip(delta_disp.columns, totals_c)):
                        if abs(total) > max(abs(totals_c)) * 0.05:
                            ax_c_d.text(i, total + (max(totals_c) - min(totals_c)) * 0.02,
                                       f'{total:.1f}', ha='center', 
                                       va='bottom' if total > 0 else 'top',
                                       fontsize=9, weight='bold',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
                    
                    ax_c_d.axhline(0, color='black', linewidth=1.2, linestyle='-', alpha=0.6)
                    ax_c_d.set_title(f"Scenario {scen} ({_scenario_label(scen)})\nDispatchable Capacity Change vs EOM Base Case", fontsize=14)
                    ax_c_d.set_xlabel("β (Risk Aversion)", fontsize=12)
                    ax_c_d.set_ylabel("Δ Capacity (GW vs EOM Base Case)", fontsize=12)
                    ax_c_d.legend(title="Technology", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
                    ax_c_d.grid(True, alpha=0.3, axis='y')
                    plt.tight_layout()
                    plt.savefig(Path(save_dir) / f"dispatchable_capacity_delta_vs_eom_scen{scen}.png", dpi=200, bbox_inches='tight')
                    plt.close(fig_c_d)
                except Exception as e:
                    print(f"Warning: Could not create delta plots for scenario {scen}: {e}")
        except Exception as e:
            print(f"Warning: Could not prepare EOM base case data: {e}")
    
    # NEW: Direct comparison of RO (7) vs CM (5) for capacity and energy
    if 5 in scenario_data and 7 in scenario_data:
        try:
            # Compare RO(beta) vs CM(beta) - Equal Risk Aversion Comparison
            # Direct DataFrame subtraction aligns on columns (betas) and index (technologies)
            delta_disp_ro_cm = scenario_data[7]['disp'] - scenario_data[5]['disp']
            delta_energy_ro_cm = scenario_data[7]['energy'] - scenario_data[5]['energy']
            
            # Plot dispatchable capacity delta: RO vs CM with consistent colors and labels
            colors_disp = [TECH_COLORS[t] for t in ['Baseload', 'MidMerit', 'Peak']]
            fig_cap, ax_cap = plt.subplots(figsize=(12, 7))
            delta_disp_ro_cm.T.plot(kind="bar", stacked=True, ax=ax_cap, color=colors_disp, 
                                    alpha=0.85, edgecolor='white', linewidth=1)
            
            # Add value labels for each technology segment (white text, no box)
            pos_cumulative = np.zeros(len(delta_disp_ro_cm.columns))
            neg_cumulative = np.zeros(len(delta_disp_ro_cm.columns))
            
            for tech in delta_disp_ro_cm.index:
                tech_values = delta_disp_ro_cm.loc[tech].values
                for i, val in enumerate(tech_values):
                    # Calculate position and update cumulative
                    if val >= 0:
                        y_pos = pos_cumulative[i] + val / 2
                        pos_cumulative[i] += val
                    else:
                        y_pos = neg_cumulative[i] + val / 2
                        neg_cumulative[i] += val
                        
                    if abs(val) > 0.2:  # Only show label if segment is large enough (> 0.2 GW)
                        ax_cap.text(i, y_pos, f'{val:.1f}', 
                                   ha='center', va='center', fontsize=9, weight='bold', 
                                   color='white', zorder=10)

            # Add total delta labels
            totals = delta_disp_ro_cm.sum(axis=0)
            for i, (beta_val, total) in enumerate(zip(delta_disp_ro_cm.columns, totals)):
                if abs(total) > max(abs(totals)) * 0.05:
                    ax_cap.text(i, total + (max(totals) - min(totals)) * 0.02,
                               f'{total:.1f}', ha='center', 
                               va='bottom' if total > 0 else 'top',
                               fontsize=9, weight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
            
            ax_cap.axhline(0, color='black', linewidth=1.2, linestyle='-', alpha=0.6)
            ax_cap.set_xlabel("β (Risk Aversion)", fontsize=12)
            ax_cap.set_ylabel("Δ Capacity (GW): RO(β) − CM(β)", fontsize=12)
            ax_cap.set_title("Decentralized RO vs Centralized CM (Equal Beta)\nDispatchable Capacity Difference by Risk Aversion", fontsize=14)
            ax_cap.legend(title="Technology", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            ax_cap.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(Path(save_dir) / "dispatchable_capacity_delta_RO_vs_CM.png", dpi=200, bbox_inches='tight')
            plt.close(fig_cap)
            
            # Plot dispatchable energy mix delta: RO vs CM with consistent colors and labels
            colors_energy = [TECH_COLORS[t] for t in ['Baseload', 'MidMerit', 'Peak']]
            fig_en, ax_en = plt.subplots(figsize=(12, 7))
            delta_energy_ro_cm.T.plot(kind="bar", stacked=True, ax=ax_en, color=colors_energy,
                                      alpha=0.85, edgecolor='white', linewidth=1)
            
            # Add value labels for each technology segment (white text, no box)
            pos_cumulative = np.zeros(len(delta_energy_ro_cm.columns))
            neg_cumulative = np.zeros(len(delta_energy_ro_cm.columns))
            
            for tech in delta_energy_ro_cm.index:
                tech_values = delta_energy_ro_cm.loc[tech].values
                for i, val in enumerate(tech_values):
                    # Calculate position and update cumulative
                    if val >= 0:
                        y_pos = pos_cumulative[i] + val / 2
                        pos_cumulative[i] += val
                    else:
                        y_pos = neg_cumulative[i] + val / 2
                        neg_cumulative[i] += val
                        
                    if abs(val) > 50:  # Only show label if segment is large enough (> 50 GWh)
                        ax_en.text(i, y_pos, f'{val:.0f}', 
                                   ha='center', va='center', fontsize=9, weight='bold', 
                                   color='white', zorder=10)

            # Add total delta labels
            totals_energy = delta_energy_ro_cm.sum(axis=0)
            for i, (beta_val, total) in enumerate(zip(delta_energy_ro_cm.columns, totals_energy)):
                if abs(total) > max(abs(totals_energy)) * 0.05:
                    ax_en.text(i, total + (max(totals_energy) - min(totals_energy)) * 0.02,
                               f'{total:.1f}', ha='center', 
                               va='bottom' if total > 0 else 'top',
                               fontsize=9, weight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
            
            ax_en.axhline(0, color='black', linewidth=1.2, linestyle='-', alpha=0.6)
            ax_en.set_xlabel("β (Risk Aversion)", fontsize=12)
            ax_en.set_ylabel("Δ Energy (GWh): RO(β) − CM(β)", fontsize=12)
            ax_en.set_title("Decentralized RO vs Centralized CM (Equal Beta)\nDispatchable Energy Difference by Risk Aversion", fontsize=14)
            ax_en.legend(title="Technology", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            ax_en.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(Path(save_dir) / "energy_mix_delta_RO_vs_CM.png", dpi=200, bbox_inches='tight')
            plt.close(fig_en)
        except Exception as e:
            print(f"Warning: Could not create RO vs CM capacity/energy comparison: {e}")

def plot_beta_sweep_price_statistics(scenarios=[3,5,7], betas=None, save_dir="../Analysis/output/plots"):
    """Plot price statistics (Median, P5, P95, Mean) vs Beta for each scenario."""
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    if betas is None:
        betas = _beta_list()
    
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 7), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]
        
    for ax, scen in zip(axes, scenarios):
        p95s, medians, p05s, means = [], [], [], []
        
        for beta in betas:
            df = _load_variant_df_for_beta(scen, beta)
            prices = df["Price"]
            p95s.append(prices.quantile(0.95))
            medians.append(prices.median())
            p05s.append(prices.quantile(0.05))
            means.append(prices.mean())
            
        x = np.arange(len(betas))
        
        # Plot lines with markers
        ax.plot(x, p95s, 'o-', label='95% Quantile', color='#DC143C', linewidth=2.5, markersize=8, alpha=0.8)
        ax.plot(x, medians, 's-', label='Median', color='black', linewidth=3, markersize=8)
        ax.plot(x, means, 'd--', label='Mean', color='#4169E1', linewidth=2, markersize=7, alpha=0.7)
        ax.plot(x, p05s, '^-', label='5% Quantile', color='#2E8B57', linewidth=2.5, markersize=8, alpha=0.8)
        
        # Fill range with gradient
        ax.fill_between(x, p05s, p95s, color='lightgray', alpha=0.15, label='5-95% Range')
        
        # Add value labels for median at each point
        for i, (beta, med) in enumerate(zip(betas, medians)):
            if i % 2 == 0:  # Label every other point to avoid crowding
                ax.text(i, med, f'{med:.3f}', ha='center', va='bottom', 
                       fontsize=8, weight='bold', color='black',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))
        
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b:.1f}" for b in betas], rotation=45)
        ax.set_xlabel("β (Risk Aversion)", fontsize=12)
        ax.set_ylabel("Price (M€/GWh)", fontsize=12)
        ax.set_title(f"Scenario {scen} ({_scenario_label(scen)})\nPrice Statistics by Risk Aversion", fontsize=14)
        ax.legend(fontsize=10, loc='upper right', bbox_to_anchor=(0.5, 0.75))
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
        
    plt.suptitle('Price Distribution Statistics Across Risk Aversion Levels', fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "price_statistics_beta_sweep.png", dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_beta_sweep_price_duration_curves(scenarios=[3,5,7], betas=None, save_dir="../Analysis/output/plots"):
    """Price duration curves: one subplot per scenario, multiple lines per β."""
    os.makedirs(save_dir, exist_ok=True)
    if betas is None:
        betas = _beta_list()
    import matplotlib.pyplot as plt
    
    # Portrait layout: 3 rows, 1 column
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(12, 6 * len(scenarios)), sharex=True)
    if len(scenarios) == 1:
        axes = [axes]
    
    # Color gradient: blue (low risk aversion, β=1.0) → red (high risk aversion, β=0.1)
    # Note: _beta_list is [1.0, 0.9, ..., 0.1]
    colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(betas)))
    # Reverse so 1.0 is Blue (Standard) and 0.1 is Red (Extreme)
    colors = colors[::-1] 
    
    for ax, scen in zip(axes, scenarios):
        for i, beta in enumerate(betas):
            df = _load_variant_df_for_beta(scen, beta)
            prices = df['Price'].values
            prices_sorted = np.sort(prices)[::-1]  # Descending order
            hours = np.arange(len(prices_sorted))
            
            # Calculate stats for label
            median = np.median(prices)
            p95 = np.percentile(prices, 95)
            p05 = np.percentile(prices, 5)
            
            # Format label with stats
            label = f"β={beta:.1f} [P95:{p95:.3f}, Med:{median:.3f}, P05:{p05:.3f}]"
            
            ax.plot(hours, prices_sorted, 
                   label=label, 
                   color=colors[i], 
                   linewidth=2, 
                   alpha=0.8)
        
        ax.set_title(f"Scenario {scen} ({_scenario_label(scen)})", fontsize=14, pad=10)
        ax.set_ylabel("Price (M€/GWh)", fontsize=12)
        ax.grid(True, alpha=0.3)
        # Place legend in upper right (empty space due to sorted data)
        ax.legend(fontsize=14, loc='upper right', ncol=1)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    
    # X-label only on bottom subplot
    axes[-1].set_xlabel("Hours (sorted by price)", fontsize=14)
    
    fig.suptitle("Price Duration Curves by Risk Aversion (β) with Key Statistics", 
                fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(Path(save_dir) / "price_duration_curves_beta_sweep.png", dpi=200, bbox_inches='tight')
    plt.close()

def plot_beta_sweep_consumer_breakdown(scenarios=[3,5,7], betas=None, save_dir="../Analysis/output/plots"):
    """Plot breakdown of ENS by consumer type across betas - all scenarios stacked vertically in one portrait plot."""
    import matplotlib.pyplot as plt
    import metrics as _metrics
    
    os.makedirs(save_dir, exist_ok=True)
    if betas is None:
        betas = _beta_list()
    
    consumer_types = ["LV_LOW", "LV_MED", "LV_HIGH", "MV_LOAD"]
    colors = get_consumer_colors_direct()
    
    # Create portrait figure with 3 rows (one per scenario)
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(14, 6 * len(scenarios)), sharex=True)
    if len(scenarios) == 1:
        axes = [axes]
    
    # Iterate over scenarios
    for ax, scen in zip(axes, scenarios):
        # Data containers
        ens_data = {c: [] for c in consumer_types}
        demand_data = {c: [] for c in consumer_types}  # Store demand for each beta
        x_labels = []
        
        valid_consumers_ens = []
        
        for beta in betas:
            try:
                df = _load_variant_df_for_beta(scen, beta)
                
                # ENS Data
                if not valid_consumers_ens:
                    valid_consumers_ens = [c for c in consumer_types if f"ENS_{c}" in df.columns]
                
                for c in valid_consumers_ens:
                    if f"ENS_{c}" in df.columns:
                        ens_data[c].append(df[f"ENS_{c}"].sum())  # Already in GWh
                    else:
                        ens_data[c].append(0)
                    
                    # Store demand for this consumer at this beta
                    if f"D_{c}" in df.columns:
                        demand_data[c].append(df[f"D_{c}"].sum())  # Already in GWh
                    else:
                        demand_data[c].append(0)
                    
                x_labels.append(f"{beta:.1f}")
                
            except Exception as e:
                print(f"Warning: Error processing S{scen} B{beta} for breakdown: {e}")
                for c in consumer_types:
                    ens_data[c].append(0)
                    demand_data[c].append(0)
        
        # ENS Breakdown Plot for this scenario
        if valid_consumers_ens:
            x = np.arange(len(betas))
            bottom = np.zeros(len(betas))
            
            # Calculate total demand for each beta
            total_demand_per_beta = []
            for i in range(len(betas)):
                total = sum(demand_data[c][i] for c in valid_consumers_ens)
                total_demand_per_beta.append(total if total > 0 else 1)
            
            for c in valid_consumers_ens:
                vals = ens_data[c]
                color = colors.get(c, None)
                bars = ax.bar(x, vals, bottom=bottom, label=c, color=color, alpha=0.85, edgecolor='white', linewidth=1)
                
                # Add percentage labels inside bars (% of agent's own demand at this beta)
                for i, val in enumerate(vals):
                    if val > 0 and demand_data[c][i] > 0:  # Show if there's any ENS
                        # Calculate percentage of agent's own demand at this specific beta
                        pct_of_agent = (val / demand_data[c][i]) * 100
                        # Only show if percentage is significant or segment is visible
                        if pct_of_agent > 0.01 or val > 0.01:  # Show if > 0.01% or > 0.01 GWh
                            ax.text(i, bottom[i] + val/2, f'{pct_of_agent:.2f}%', 
                                   ha='center', va='center', fontsize=8, weight='bold', color='white')
                
                bottom += np.array(vals)
            
            # Add percentage of total demand labels on top (for each beta)
            for i, total in enumerate(bottom):
                if total > 0 and total_demand_per_beta[i] > 0:  # Show if there's any ENS
                    pct_of_total = (total / total_demand_per_beta[i]) * 100
                    ax.text(i, total + max(bottom) * 0.02, f'{pct_of_total:.2f}%', 
                           ha='center', va='bottom', fontsize=10, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8, edgecolor='gray'))
            
            ax.set_ylabel("ENS (GWh)", fontsize=12)
            ax.set_title(f"Scenario {scen} ({_scenario_label(scen)})\n(Labels: % of agent demand inside bars, % of total demand on top)", fontsize=13)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
            ax.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis labels only on bottom subplot
    axes[-1].set_xticks(np.arange(len(betas)))
    axes[-1].set_xticklabels(x_labels, rotation=45)
    axes[-1].set_xlabel("β (Risk Aversion)", fontsize=12)
    
    fig.suptitle("ENS Breakdown by Consumer Type and Risk Aversion\n(Percentages calculated as ENS/Demand for each scenario-beta combination)", 
                fontsize=15, y=0.995)
    plt.tight_layout(rect=[0, 0, 0.95, 0.99])
    plt.savefig(Path(save_dir) / f"consumer_ens_breakdown_all_scenarios.png", dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_beta_sweep_residual_load_duration_curves(scenarios=[3,5,7], betas=None, save_dir="../Analysis/output/plots"):
    """Residual load duration curves: one subplot per scenario, multiple lines per β."""
    os.makedirs(save_dir, exist_ok=True)
    if betas is None:
        betas = _beta_list()
    import matplotlib.pyplot as plt
    
    # Portrait layout: 3 rows, 1 column
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(12, 6 * len(scenarios)), sharex=True)
    if len(scenarios) == 1:
        axes = [axes]
    
    # Color gradient: blue (β=1.0) → red (β=0.1)
    colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(betas)))
    colors = colors[::-1]  # Reverse so 1.0 is Blue
    
    for ax, scen in zip(axes, scenarios):
        for i, beta in enumerate(betas):
            df = _load_variant_df_for_beta(scen, beta)
            
            # Calculate residual load: total demand - renewables
            # Identify renewable columns
            renewable_cols = [col for col in df.columns if 'Wind' in col or 'PV' in col]
            
            # Get total demand
            if 'Total_Demand' in df.columns:
                total_demand = df['Total_Demand'].values
            else:
                # Fallback: sum consumer demands
                consumer_cols = [col for col in df.columns if col.startswith('D_')]
                if consumer_cols:
                    total_demand = df[consumer_cols].sum(axis=1).values
                else:
                    print(f"Warning: Could not find demand data for scenario {scen}, beta {beta}")
                    continue
            
            # Calculate renewable generation
            if renewable_cols:
                renewable_gen = df[renewable_cols].sum(axis=1).values
            else:
                renewable_gen = np.zeros_like(total_demand)
            
            # Residual load
            residual_load = total_demand - renewable_gen
            
            # Sort descending for duration curve
            residual_sorted = np.sort(residual_load)[::-1]
            hours = np.arange(len(residual_sorted))
            
            # Calculate stats for label (removed Min)
            peak = np.max(residual_load)
            median = np.median(residual_load)
            p95 = np.percentile(residual_load, 95)
            p05 = np.percentile(residual_load, 5)
            
            # Format label with stats (without Min)
            label = f"β={beta:.1f} [Peak:{peak:.2f}, P95:{p95:.2f}, Med:{median:.2f}, P05:{p05:.2f}]"
            
            ax.plot(hours, residual_sorted, 
                   label=label, 
                   color=colors[i], 
                   linewidth=2, 
                   alpha=0.8)
        
        ax.set_title(f"Scenario {scen} ({_scenario_label(scen)})", fontsize=14, pad=10)
        ax.set_ylabel("Residual Load (GW)", fontsize=12)
        ax.grid(True, alpha=0.3)
        # Place legend to the right
        ax.legend(fontsize=11, loc='center left', bbox_to_anchor=(0.75, 0.75))
    
    # X-label only on bottom subplot
    axes[-1].set_xlabel("Hours (sorted by residual load)", fontsize=12)
    
    fig.suptitle("Residual Load Duration Curves by Risk Aversion (β)\n(Total Demand - Renewable Generation)", 
                fontsize=16, y=0.995)
    # Adjust layout to accommodate right legends
    fig.tight_layout(rect=[0, 0, 0.85, 0.99])
    plt.savefig(Path(save_dir) / "residual_load_duration_curves_beta_sweep.png", dpi=200, bbox_inches='tight')
    plt.close()

def plot_beta_sweep_generator_cost_recovery(scenarios=[3,5,7], betas=None, save_dir="../Analysis/output/plots"):
    """Plot generator cost recovery: Revenue (EOM + CM) vs Total Costs (Investment + Operational) by β."""
    import matplotlib.pyplot as plt
    import metrics as _metrics
    from config_utils import load_config, get_ntimesteps
    
    os.makedirs(save_dir, exist_ok=True)
    if betas is None:
        betas = _beta_list()
    
    # Load generator cost parameters from config
    config = load_config()
    generators_config = config['Generators']
    n_timesteps = get_ntimesteps()
    
    # Define consistent colors for generators
    TECH_COLORS = {
        'Baseload': '#8B4513',   # Saddle Brown
        'MidMerit': '#FF8C00',   # Dark Orange
        'Peak': '#DC143C',       # Crimson
        'WindOnshore': '#2E8B57' # Sea Green
    }
    
    # Calculate delta profit vs EOM base case (Scenario 3, beta=1.0)
    
    # Load EOM Base Case profit for reference (Scenario 3, Beta 1.0)
    eom_base_profit = {}
    try:
        df_eom_base = _load_variant_df_for_beta(3, 1.0)
        dispatchable_gens = ['Baseload', 'MidMerit', 'Peak', 'WindOnshore']
        
        for gen in dispatchable_gens:
            # EOM Revenue
            if f'G_{gen}' in df_eom_base.columns:
                generation = df_eom_base[f'G_{gen}'].values
                prices = df_eom_base['Price'].values
                revenue = (prices * generation).sum()
            else:
                revenue = 0
                generation = np.zeros(len(df_eom_base))
                
            # Costs (Operational + Investment)
            op_cost = 0
            inv_cost = 0
            if gen in generators_config:
                # Operational
                a = generators_config[gen]['a']
                b = generators_config[gen]['b']
                op_cost = sum(b * gen_val + a * gen_val**2 / 2 for gen_val in generation)
                
                # Investment
                if f'C_{gen}' in df_eom_base.columns:
                    cap = df_eom_base[f'C_{gen}'].iloc[0]
                    inv_total = generators_config[gen]['INV']
                    n_years = generators_config[gen]['n']
                    annual_inv = (inv_total * cap) / n_years
                    inv_cost = annual_inv * (n_timesteps / 8760)
            
            eom_base_profit[gen] = revenue - (op_cost + inv_cost)
            
    except Exception as e:
        print(f"Warning: Could not calculate EOM base profit: {e}")
        return

    # Comparisons to plot: 5 vs EOM(1.0), 7 vs EOM(1.0), 5 vs 7(beta) - wait, user said 5 vs 7 (which implies diff between them at same beta?)
    # Actually user query: "5 vs EOM", "7 vs EOM", "5 vs 7"
    # Interpreting as:
    # 1. Scenario 5 profit delta vs EOM Base (S3, b1.0)
    # 2. Scenario 7 profit delta vs EOM Base (S3, b1.0)
    # 3. Scenario 7 profit delta vs Scenario 5 (at same beta) - or maybe 5 vs 7 diff? User said "5 vs 7"
    # Let's stick to delta vs EOM Base for Scenarios 5 and 7, and maybe a direct comparison.
    
    # Re-reading carefully: "one plot per scenario... 5 vs EOM, 7 vs EOM, 5 vs 7"
    # This implies 3 distinct plots.
    
    # PLOT 1: Scenario 5 vs EOM Base
    if 5 in scenarios:
        fig, ax = plt.subplots(figsize=(14, 8))
        width = 0.15
        x = np.arange(len(betas))
        
        for i, gen in enumerate(dispatchable_gens):
            deltas = []
            for beta in betas:
                # Calculate profit for S5
                try:
                    df = _load_variant_df_for_beta(5, beta)
                    # ... calculation logic ...
                    # (Using the loop logic from before but simplified)
                    
                    # Revenue
                    gen_val = df[f'G_{gen}'].values if f'G_{gen}' in df.columns else np.zeros(len(df))
                    prices = df['Price'].values
                    eom_rev = (prices * gen_val).sum()
                    
                    cm_rev = 0
                    if f'C_cCM_{gen}' in df.columns and 'λ_cCM' in df.columns:
                        cm_rev = df['λ_cCM'].iloc[0] * df[f'C_cCM_{gen}'].iloc[0] * n_timesteps
                        
                    total_rev = eom_rev + cm_rev
                    
                    # Costs
                    op_cost = 0
                    inv_cost = 0
                    if gen in generators_config:
                        a = generators_config[gen]['a']
                        b = generators_config[gen]['b']
                        op_cost = sum(b * g + a * g**2 / 2 for g in gen_val)
                        
                        cap = df[f'C_{gen}'].iloc[0] if f'C_{gen}' in df.columns else 0
                        inv_cost = (generators_config[gen]['INV'] * cap / generators_config[gen]['n']) * (n_timesteps / 8760)
                        
                    profit = total_rev - (op_cost + inv_cost)
                    deltas.append(profit - eom_base_profit.get(gen, 0))
                except:
                    deltas.append(0)
            
            # Plot clustered bars
            # Offset based on generator index
            offset = (i - len(dispatchable_gens)/2 + 0.5) * width
            ax.bar(x + offset, deltas, width, label=gen, color=TECH_COLORS.get(gen, '#666666'), alpha=0.85, edgecolor='white')
            
        ax.axhline(0, color='black', linewidth=1, linestyle='-')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b:.1f}" for b in betas])
        ax.set_xlabel("β (Risk Aversion)", fontsize=12)
        ax.set_ylabel("Δ Net Profit (M€) vs EOM Base (β=1.0)", fontsize=12)
        ax.set_title(f"Scenario 5 (Centralized CM)\nProfit Delta vs EOM Base Case (S3, β=1.0)", fontsize=14)
        ax.legend(title="Technology", fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(Path(save_dir) / "generator_profit_delta_S5_vs_EOM_Base.png", dpi=200)
        plt.close(fig)

    # PLOT 2: Scenario 7 vs EOM Base
    if 7 in scenarios:
        fig, ax = plt.subplots(figsize=(14, 8))
        width = 0.15
        x = np.arange(len(betas))
        
        for i, gen in enumerate(dispatchable_gens):
            deltas = []
            for beta in betas:
                try:
                    df = _load_variant_df_for_beta(7, beta)
                    # ... calculation logic ...
                    gen_val = df[f'G_{gen}'].values if f'G_{gen}' in df.columns else np.zeros(len(df))
                    prices = df['Price'].values
                    eom_rev = (prices * gen_val).sum()
                    
                    cm_rev = 0
                    if f'C_dCM_{gen}' in df.columns and 'λ_dCM' in df.columns:
                        cm_rev = df['λ_dCM'].iloc[0] * df[f'C_dCM_{gen}'].iloc[0] * n_timesteps
                        
                    total_rev = eom_rev + cm_rev
                    
                    # Costs
                    op_cost = 0
                    inv_cost = 0
                    if gen in generators_config:
                        a = generators_config[gen]['a']
                        b = generators_config[gen]['b']
                        op_cost = sum(b * g + a * g**2 / 2 for g in gen_val)
                        
                        cap = df[f'C_{gen}'].iloc[0] if f'C_{gen}' in df.columns else 0
                        inv_cost = (generators_config[gen]['INV'] * cap / generators_config[gen]['n']) * (n_timesteps / 8760)
                        
                    profit = total_rev - (op_cost + inv_cost)
                    deltas.append(profit - eom_base_profit.get(gen, 0))
                except:
                    deltas.append(0)
            
            ax.bar(x + (i - len(dispatchable_gens)/2 + 0.5) * width, deltas, width, 
                   label=gen, color=TECH_COLORS.get(gen, '#666666'), alpha=0.85, edgecolor='white')
            
        ax.axhline(0, color='black', linewidth=1, linestyle='-')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b:.1f}" for b in betas])
        ax.set_xlabel("β (Risk Aversion)", fontsize=12)
        ax.set_ylabel("Δ Net Profit (M€) vs EOM Base (β=1.0)", fontsize=12)
        ax.set_title(f"Scenario 7 (Decentralized RO)\nProfit Delta vs EOM Base Case (S3, β=1.0)", fontsize=14)
        ax.legend(title="Technology", fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(Path(save_dir) / "generator_profit_delta_S7_vs_EOM_Base.png", dpi=200)
        plt.close(fig)
        
    # PLOT 3: Scenario 5 vs Scenario 7 (Difference at same beta)
    if 5 in scenarios and 7 in scenarios:
        fig, ax = plt.subplots(figsize=(14, 8))
        width = 0.15
        x = np.arange(len(betas))
        
        for i, gen in enumerate(dispatchable_gens):
            diffs = []
            for beta in betas:
                try:
                    # Calculate S5 Profit
                    df5 = _load_variant_df_for_beta(5, beta)
                    gen_val = df5[f'G_{gen}'].values if f'G_{gen}' in df5.columns else np.zeros(len(df5))
                    eom_rev = (df5['Price'].values * gen_val).sum()
                    cm_rev = 0
                    if f'C_cCM_{gen}' in df5.columns: cm_rev = df5['λ_cCM'].iloc[0] * df5[f'C_cCM_{gen}'].iloc[0] * n_timesteps
                    op_cost = 0
                    inv_cost = 0
                    if gen in generators_config:
                        a, b = generators_config[gen]['a'], generators_config[gen]['b']
                        op_cost = sum(b * g + a * g**2 / 2 for g in gen_val)
                        cap = df5[f'C_{gen}'].iloc[0] if f'C_{gen}' in df5.columns else 0
                        inv_cost = (generators_config[gen]['INV'] * cap / generators_config[gen]['n']) * (n_timesteps / 8760)
                    profit5 = (eom_rev + cm_rev) - (op_cost + inv_cost)
                    
                    # Calculate S7 Profit
                    df7 = _load_variant_df_for_beta(7, beta)
                    gen_val = df7[f'G_{gen}'].values if f'G_{gen}' in df7.columns else np.zeros(len(df7))
                    eom_rev = (df7['Price'].values * gen_val).sum()
                    cm_rev = 0
                    if f'C_dCM_{gen}' in df7.columns: cm_rev = df7['λ_dCM'].iloc[0] * df7[f'C_dCM_{gen}'].iloc[0] * n_timesteps
                    op_cost = 0
                    inv_cost = 0
                    if gen in generators_config:
                        a, b = generators_config[gen]['a'], generators_config[gen]['b']
                        op_cost = sum(b * g + a * g**2 / 2 for g in gen_val)
                        cap = df7[f'C_{gen}'].iloc[0] if f'C_{gen}' in df7.columns else 0
                        inv_cost = (generators_config[gen]['INV'] * cap / generators_config[gen]['n']) * (n_timesteps / 8760)
                    profit7 = (eom_rev + cm_rev) - (op_cost + inv_cost)
                    
                    # Diff: S7 - S5 (Positive means S7 is more profitable)
                    diffs.append(profit7 - profit5)
                except:
                    diffs.append(0)
            
            ax.bar(x + (i - len(dispatchable_gens)/2 + 0.5) * width, diffs, width, 
                   label=gen, color=TECH_COLORS.get(gen, '#666666'), alpha=0.85, edgecolor='white')
                   
        ax.axhline(0, color='black', linewidth=1, linestyle='-')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b:.1f}" for b in betas])
        ax.set_xlabel("β (Risk Aversion)", fontsize=12)
        ax.set_ylabel("Δ Net Profit (M€): S7 (RO) − S5 (CM)", fontsize=12)
        ax.set_title(f"Profit Difference: Decentralized RO (S7) vs Centralized CM (S5)\n(Positive = RO more profitable)", fontsize=14)
        ax.legend(title="Technology", fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(Path(save_dir) / "generator_profit_delta_S5_vs_S7.png", dpi=200)
        plt.close(fig)

def plot_beta_sweep_consumer_welfare_vs_eom(scenarios=[3,5,7], betas=None, save_dir="../Analysis/output/plots"):
    """Plot consumer welfare (economic surplus) differences from EOM base case (β=1.0) by β."""
    import matplotlib.pyplot as plt
    from welfare_analysis import calculate_welfare_for_scenario, load_generator_cost_parameters
    
    os.makedirs(save_dir, exist_ok=True)
    if betas is None:
        betas = _beta_list()
    
    # Load generator cost parameters for welfare calculation
    cost_curves = load_generator_cost_parameters()
    
    # Calculate EOM base case (scenario 3, beta=1.0) welfare
    try:
        df_eom_base = _load_variant_df_for_beta(3, 1.0)
        welfare_eom_base = calculate_welfare_for_scenario(df_eom_base, cost_curves)
        eom_consumer_surplus = welfare_eom_base['total_consumer_surplus']
        eom_producer_surplus = welfare_eom_base['total_producer_surplus']
        eom_total_welfare = welfare_eom_base['total_welfare']
        
        print(f"DEBUG: EOM Base Case Welfare - CS: {eom_consumer_surplus:.2f}, PS: {eom_producer_surplus:.2f}, Total: {eom_total_welfare:.2f}")
    except Exception as e:
        print(f"Warning: Could not calculate EOM base case welfare: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # For each scenario, calculate welfare differences from EOM
    for scen in scenarios:
        consumer_surplus_delta = []
        producer_surplus_delta = []
        total_welfare_delta = []
        
        for beta in betas:
            try:
                df = _load_variant_df_for_beta(scen, beta)
                welfare = calculate_welfare_for_scenario(df, cost_curves)
                
                cs_delta = welfare['total_consumer_surplus'] - eom_consumer_surplus
                ps_delta = welfare['total_producer_surplus'] - eom_producer_surplus
                tw_delta = welfare['total_welfare'] - eom_total_welfare
                
                consumer_surplus_delta.append(cs_delta)
                producer_surplus_delta.append(ps_delta)
                total_welfare_delta.append(tw_delta)
                
                if beta == 1.0:
                    print(f"DEBUG: Scenario {scen}, β={beta} - CS: {welfare['total_consumer_surplus']:.2f}, PS: {welfare['total_producer_surplus']:.2f}")
                    print(f"       Deltas - CS: {cs_delta:.2f}, PS: {ps_delta:.2f}, Total: {tw_delta:.2f}")
                
            except Exception as e:
                print(f"Warning: Error calculating welfare for scenario {scen}, beta {beta}: {e}")
                consumer_surplus_delta.append(0)
                producer_surplus_delta.append(0)
                total_welfare_delta.append(0)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        x = np.arange(len(betas))
        width = 0.3
        
        # Plot 1: Welfare components delta
        bars1 = ax1.bar(x - width/2, consumer_surplus_delta, width, label='Δ Consumer Surplus', 
                       color='lightblue', alpha=0.85, edgecolor='white', linewidth=1)
        bars2 = ax1.bar(x + width/2, producer_surplus_delta, width, label='Δ Producer Surplus', 
                       color='lightcoral', alpha=0.85, edgecolor='white', linewidth=1)
        
        # Add value labels
        for i, (cs, ps) in enumerate(zip(consumer_surplus_delta, producer_surplus_delta)):
            if abs(cs) > 0.1:
                ax1.text(i - width/2, cs + (max(consumer_surplus_delta) - min(consumer_surplus_delta)) * 0.02,
                        f'{cs:.1f}', ha='center', va='bottom' if cs > 0 else 'top', fontsize=8)
            if abs(ps) > 0.1:
                ax1.text(i + width/2, ps + (max(producer_surplus_delta) - min(producer_surplus_delta)) * 0.02,
                        f'{ps:.1f}', ha='center', va='bottom' if ps > 0 else 'top', fontsize=8)
        
        ax1.axhline(0, color='black', linewidth=1.2, linestyle='-', alpha=0.6)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{b:.1f}" for b in betas], rotation=45)
        ax1.set_xlabel("β (Risk Aversion)", fontsize=11)
        ax1.set_ylabel("Δ Welfare (M€)", fontsize=11)
        ax1.set_title(f"Welfare Components Δ vs EOM (β=1.0)", fontsize=13)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Total welfare delta
        bars3 = ax2.bar(x, total_welfare_delta, width*2, label='Δ Total Welfare', 
                       color='purple', alpha=0.85, edgecolor='white', linewidth=1)
        
        # Add value labels
        for i, val in enumerate(total_welfare_delta):
            if abs(val) > 0.1:
                ax2.text(i, val + (max(total_welfare_delta) - min(total_welfare_delta)) * 0.02,
                        f'{val:.1f}', ha='center', va='bottom' if val > 0 else 'top', 
                        fontsize=9, weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        ax2.axhline(0, color='black', linewidth=1.2, linestyle='-', alpha=0.6)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{b:.1f}" for b in betas], rotation=45)
        ax2.set_xlabel("β (Risk Aversion)", fontsize=11)
        ax2.set_ylabel("Δ Total Welfare (M€)", fontsize=11)
        ax2.set_title("Total Welfare Δ vs EOM (β=1.0)", fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f"Scenario {scen} ({_scenario_label(scen)})\nEconomic Welfare Changes vs EOM Base Case (β=1.0)", 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(Path(save_dir) / f"welfare_delta_vs_eom_scen{scen}.png", dpi=200, bbox_inches='tight')
        plt.close(fig)
    
    # Add direct comparison: RO (7) vs CM (5) for welfare
    if 5 in scenarios and 7 in scenarios:
        try:
            # Load CM base case (scenario 5, beta=1.0) welfare
            df_cm_base = _load_variant_df_for_beta(5, 1.0)
            welfare_cm_base = calculate_welfare_for_scenario(df_cm_base, cost_curves)
            cm_consumer_surplus = welfare_cm_base['total_consumer_surplus']
            cm_producer_surplus = welfare_cm_base['total_producer_surplus']
            
            ro_cs_delta = []
            ro_ps_delta = []
            ro_total_welfare_delta = []
            
            for beta in betas:
                try:
                    df_ro = _load_variant_df_for_beta(7, beta)
                    welfare_ro = calculate_welfare_for_scenario(df_ro, cost_curves)
                    
                    ro_cs_delta.append(welfare_ro['total_consumer_surplus'] - cm_consumer_surplus)
                    ro_ps_delta.append(welfare_ro['total_producer_surplus'] - cm_producer_surplus)
                    ro_total_welfare_delta.append(welfare_ro['total_welfare'] - welfare_cm_base['total_welfare'])
                    
                except Exception as e:
                    print(f"Warning: Error calculating RO vs CM welfare for beta {beta}: {e}")
                    ro_cs_delta.append(0)
                    ro_ps_delta.append(0)
                    ro_total_welfare_delta.append(0)
            
            # Create comparison plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            x = np.arange(len(betas))
            width = 0.3
            
            # Plot 1: Welfare components delta
            bars1 = ax1.bar(x - width/2, ro_cs_delta, width, label='Δ Consumer Surplus', 
                           color='lightblue', alpha=0.85, edgecolor='white', linewidth=1)
            bars2 = ax1.bar(x + width/2, ro_ps_delta, width, label='Δ Producer Surplus', 
                           color='lightcoral', alpha=0.85, edgecolor='white', linewidth=1)
            
            ax1.axhline(0, color='black', linewidth=1.2, linestyle='-', alpha=0.6)
            ax1.set_xticks(x)
            ax1.set_xticklabels([f"{b:.1f}" for b in betas], rotation=45)
            ax1.set_xlabel("β (Risk Aversion)", fontsize=11)
            ax1.set_ylabel("Δ Welfare: RO(β) − CM(β=1.0) [M€]", fontsize=11)
            ax1.set_title("Welfare Components: Decentralized RO vs Centralized CM Base Case", fontsize=13)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Plot 2: Total welfare delta
            bars3 = ax2.bar(x, ro_total_welfare_delta, width*2, label='Δ Total Welfare', 
                           color='purple', alpha=0.85, edgecolor='white', linewidth=1)
            
            # Add value labels
            for i, val in enumerate(ro_total_welfare_delta):
                if abs(val) > 0.1:
                    ax2.text(i, val + (max(ro_total_welfare_delta) - min(ro_total_welfare_delta)) * 0.02,
                            f'{val:.1f}', ha='center', va='bottom' if val > 0 else 'top', 
                            fontsize=10, weight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', 
                                    facecolor='yellow' if val > 0 else 'lightgreen', 
                                    alpha=0.8, edgecolor='black', linewidth=1))
            
            ax2.axhline(0, color='black', linewidth=1.2, linestyle='-', alpha=0.6)
            ax2.set_xticks(x)
            ax2.set_xticklabels([f"{b:.1f}" for b in betas], rotation=45)
            ax2.set_xlabel("β (Risk Aversion)", fontsize=11)
            ax2.set_ylabel("Δ Total Welfare: RO(β) − CM(β=1.0) [M€]", fontsize=11)
            ax2.set_title("Total Welfare: Decentralized RO vs Centralized CM Base Case", fontsize=13)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.suptitle("Decentralized RO vs Centralized CM: Economic Welfare Comparison", 
                        fontsize=16, y=0.98)
            plt.tight_layout()
            plt.savefig(Path(save_dir) / f"welfare_RO_vs_CM.png", dpi=200, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Could not create RO vs CM welfare comparison: {e}")

def run_beta_sweep_plots(scenarios=[3,5,7], betas=None, save_dir="../Analysis/output/plots"):
    """Entry point to generate all beta-sweep plots and save alongside other plots."""
    # Create sensitivity subdirectory
    sensitivity_dir = Path(save_dir) / "sensitivity_analysis"
    sensitivity_dir.mkdir(parents=True, exist_ok=True)
    sensitivity_path = str(sensitivity_dir)
    
    print(f"  - Saving sensitivity plots to: {sensitivity_path}")

    plot_beta_sweep_consumer_boxplots(scenarios=scenarios, betas=betas, save_dir=sensitivity_path)
    plot_beta_sweep_capacity_and_energy(scenarios=scenarios, betas=betas, save_dir=sensitivity_path)
    plot_beta_sweep_price_duration_curves(scenarios=scenarios, betas=betas, save_dir=sensitivity_path)
    plot_beta_sweep_price_statistics(scenarios=scenarios, betas=betas, save_dir=sensitivity_path)
    plot_beta_sweep_consumer_breakdown(scenarios=scenarios, betas=betas, save_dir=sensitivity_path)
    plot_beta_sweep_residual_load_duration_curves(scenarios=scenarios, betas=betas, save_dir=sensitivity_path)
    plot_beta_sweep_generator_cost_recovery(scenarios=scenarios, betas=betas, save_dir=sensitivity_path)
    plot_beta_sweep_consumer_welfare_vs_eom(scenarios=scenarios, betas=betas, save_dir=sensitivity_path)
    
    # New plots
    plot_beta_sweep_consumer_cost_delta_combined(scenarios=scenarios, betas=betas, save_dir=sensitivity_path)
    plot_beta_sweep_consumer_cost_ro_vs_cm(scenarios=scenarios, betas=betas, save_dir=sensitivity_path)

def plot_dispatchable_generation_timeseries(scenarios_data, save_path=None):
    """Plot stacked dispatchable generation timeseries (Baseload, MidMerit, Peak) for each scenario stacked vertically."""
    if not scenarios_data:
        print("No scenario data provided to plot_dispatchable_generation_timeseries; skipping.")
        return None
    
    # Dynamically detect available dispatchable generators
    sample_df = next(iter(scenarios_data.values()))  # Use first available scenario to detect columns
    generator_columns = [col for col in sample_df.columns if col.startswith('G_') and not any(c in col for c in ["LV_MED", "MV_LOAD", "LV_HIGH", "LV_LOW", "Wind"])]
    dispatchable_generators = [col[2:] for col in generator_columns]  # Remove "G_" prefix
    
    if not dispatchable_generators:
        print("No dispatchable generators found for timeseries plot")
        return None
    
    # Colors for different generators
    colors = ['#8B4513', '#FF8C00', '#DC143C', '#4682B4', '#32CD32']  # Brown, Orange, Red, Blue, Green
    
    # Create figure with subplots stacked vertically
    scenarios = sorted(list(scenarios_data.keys()))
    n_scenarios = len(scenarios)
    
    fig, axes = plt.subplots(n_scenarios, 1, figsize=(16, 4*n_scenarios), sharex=True)
    if n_scenarios == 1:
        axes = [axes]
    
    for i, (scen, df) in enumerate(scenarios_data.items()):
        ax = axes[i]
        
        # Create time array
        hours = np.arange(len(df))
        
        # Calculate stacked generation
        bottom = np.zeros(len(df))
        total_dispatchable = np.zeros(len(df))
        
        for j, gen_type in enumerate(dispatchable_generators):
            if f"G_{gen_type}" in df.columns:
                gen_data = df[f"G_{gen_type}"].values
                
                # Plot stacked area for this generator type
                ax.fill_between(hours, bottom, bottom + gen_data, 
                              color=colors[j % len(colors)], alpha=0.7, label=f'{gen_type}')
                bottom += gen_data
                total_dispatchable += gen_data
        
        # Add statistics box
        if np.any(total_dispatchable > 0):
            max_val = total_dispatchable.max()
            min_val = total_dispatchable.min()
            avg_val = total_dispatchable.mean()
            
            stats_text = f'Max: {max_val:.1f} GW\nMin: {min_val:.1f} GW\nAvg: {avg_val:.1f} GW'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Formatting
        ax.set_ylabel('Power (GW)', fontsize=12)
        ax.set_title(f'Scenario {scen} - {SCENARIO_NAMES.get(scen, f"Scenario {scen}")}\nStacked Dispatchable Generation', 
                    fontsize=14, pad=10)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set x-axis limits
        ax.set_xlim(0, len(df)-1)
    
    # Set x-label for bottom subplot only
    axes[-1].set_xlabel('Time (Hours)', fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.2)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)
    
    plt.close(fig)
    return fig

def plot_generation_vs_price_scatter(scenarios_data, save_path=None):
    """Create scatter plots of generation vs. prices for each technology to assess unit behavior."""
    
    # Define generator types and their colors
    generator_types = ['Baseload', 'MidMerit', 'Peak', 'WindOnshore']
    colors = ['#8B4513', '#FF8C00', '#DC143C', '#2E8B57']  # Brown, Orange, Red, Green
    
    # Create figure with subplots for each generator type
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (gen_type, color) in enumerate(zip(generator_types, colors)):
        ax = axes[i]
        
        # Collect data from all scenarios
        all_prices = []
        all_generation = []
        scenario_labels = []
        
        for scen, df in scenarios_data.items():
            if f"G_{gen_type}" in df.columns and "Price" in df.columns:
                prices = df["Price"].values  # M€/GWh
                generation = df[f"G_{gen_type}"].values  # GW
                
                all_prices.extend(prices)
                all_generation.extend(generation)
                scenario_labels.extend([f"Scenario {scen}"] * len(prices))
        
        if all_prices and all_generation:
            # Create scatter plot
            scatter = ax.scatter(all_prices, all_generation, c=color, alpha=0.6, s=20)
            
            # Add trend line
            if len(all_prices) > 1:
                z = np.polyfit(all_prices, all_generation, 1)
                p = np.poly1d(z)
                ax.plot(all_prices, p(all_prices), "--", color=color, alpha=0.8, linewidth=2)
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(all_prices, all_generation)[0, 1]
            
            # Add statistics
            max_gen = max(all_generation)
            min_gen = min(all_generation)
            avg_gen = np.mean(all_generation)
            max_price = max(all_prices)
            min_price = min(all_prices)
            
            stats_text = f'Max Gen: {max_gen:.2f} GW\nMin Gen: {min_gen:.2f} GW\nAvg Gen: {avg_gen:.2f} GW\nCorrelation: {correlation:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Price (M€/GWh)', fontsize=12)
        ax.set_ylabel('Generation (GW)', fontsize=12)
        ax.set_title(f'{gen_type} Generation vs. Price\n(All Scenarios)', fontsize=14, pad=10)
        ax.grid(True, alpha=0.3)
        
        # Format price axis to show more decimal places
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    
    plt.tight_layout(pad=3.0)
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.2)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)
    
    return fig

def plot_generation_vs_price_by_scenario(scenarios_data, save_path=None):
    """Create separate scatter plots for each scenario showing generation vs. prices for each technology."""
    
    # Define generator types and their colors
    generator_types = ['Baseload', 'MidMerit', 'Peak', 'WindOnshore']
    colors = ['#8B4513', '#FF8C00', '#DC143C', '#2E8B57']  # Brown, Orange, Red, Green
    
    scenarios = sorted(list(scenarios_data.keys()))
    n_scenarios = len(scenarios)
    
    # Create figure with subplots for each scenario
    fig, axes = plt.subplots(n_scenarios, 1, figsize=(16, 5*n_scenarios))
    if n_scenarios == 1:
        axes = [axes]
    
    for i, (scen, df) in enumerate(scenarios_data.items()):
        ax = axes[i]
        
        # Plot each generator type
        for j, (gen_type, color) in enumerate(zip(generator_types, colors)):
            if f"G_{gen_type}" in df.columns and "Price" in df.columns:
                prices = df["Price"].values  # M€/GWh
                generation = df[f"G_{gen_type}"].values  # GW
                
                # Create scatter plot
                ax.scatter(prices, generation, c=color, alpha=0.7, s=30, 
                          label=f'{gen_type}', edgecolors='white', linewidth=0.5)
                
                # Add trend line for this generator type
                if len(prices) > 1 and np.any(generation > 0):
                    # Only fit trend line if there's actual generation
                    z = np.polyfit(prices, generation, 1)
                    p = np.poly1d(z)
                    ax.plot(prices, p(prices), "--", color=color, alpha=0.8, linewidth=1.5)
        
        ax.set_xlabel('Price (M€/GWh)', fontsize=12)
        ax.set_ylabel('Generation (GW)', fontsize=12)
        ax.set_title(f'Scenario {scen} - {SCENARIO_NAMES.get(scen, f"Scenario {scen}")}\nGeneration vs. Price by Technology', 
                    fontsize=14, pad=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format price axis to show more decimal places
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.2)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)
    
    return fig

def plot_price_response_analysis(scenarios_data, save_path=None):
    """Create a comprehensive price response analysis showing how different technologies respond to price levels."""
    
    # Define generator types and their colors
    generator_types = ['Baseload', 'MidMerit', 'Peak', 'WindOnshore']
    colors = ['#8B4513', '#FF8C00', '#DC143C', '#2E8B57']  # Brown, Orange, Red, Green
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Price distribution across scenarios
    ax1 = fig.add_subplot(gs[0, 0])
    all_prices = []
    scenario_labels = []
    for scen, df in scenarios_data.items():
        if "Price" in df.columns:
            prices = df["Price"].values
            all_prices.extend(prices)
            scenario_labels.extend([f"Scenario {scen}"] * len(prices))
    
    ax1.hist(all_prices, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax1.set_xlabel('Price (M€/GWh)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Price Distribution Across All Scenarios', fontsize=14, pad=10)
    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    
    # 2. Generation distribution for each technology
    ax2 = fig.add_subplot(gs[0, 1])
    for gen_type, color in zip(generator_types, colors):
        all_gen = []
        for scen, df in scenarios_data.items():
            if f"G_{gen_type}" in df.columns:
                gen = df[f"G_{gen_type}"].values
                all_gen.extend(gen)
        
        if all_gen:
            ax2.hist(all_gen, bins=30, alpha=0.6, color=color, label=gen_type, edgecolor='white')
    
    ax2.set_xlabel('Generation (GW)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Generation Distribution by Technology', fontsize=14, pad=10)
    ax2.legend(fontsize=10)
    
    # 3. Price vs Generation scatter for all technologies (combined)
    ax3 = fig.add_subplot(gs[1, :])
    
    for gen_type, color in zip(generator_types, colors):
        all_prices = []
        all_generation = []
        
        for scen, df in scenarios_data.items():
            if f"G_{gen_type}" in df.columns and "Price" in df.columns:
                prices = df["Price"].values
                generation = df[f"G_{gen_type}"].values
                all_prices.extend(prices)
                all_generation.extend(generation)
        
        if all_prices and all_generation:
            ax3.scatter(all_prices, all_generation, c=color, alpha=0.6, s=20, label=gen_type)
            
            # Add trend line
            if len(all_prices) > 1:
                z = np.polyfit(all_prices, all_generation, 1)
                p = np.poly1d(z)
                ax3.plot(all_prices, p(all_prices), "--", color=color, alpha=0.8, linewidth=2)
    
    ax3.set_xlabel('Price (M€/GWh)', fontsize=14)
    ax3.set_ylabel('Generation (GW)', fontsize=14)
    ax3.set_title('Generation vs. Price by Technology (All Scenarios)', fontsize=16, pad=15)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    
    # 4. Price response analysis table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Calculate statistics for each technology
    stats_data = []
    for gen_type in generator_types:
        all_prices = []
        all_generation = []
        
        for scen, df in scenarios_data.items():
            if f"G_{gen_type}" in df.columns and "Price" in df.columns:
                prices = df["Price"].values
                generation = df[f"G_{gen_type}"].values
                all_prices.extend(prices)
                all_generation.extend(generation)
        
        if all_prices and all_generation:
            correlation = np.corrcoef(all_prices, all_generation)[0, 1]
            avg_gen = np.mean(all_generation)
            max_gen = max(all_generation)
            min_gen = min(all_generation)
            avg_price_when_generating = np.mean([p for p, g in zip(all_prices, all_generation) if g > 0.01])
            
            stats_data.append([
                gen_type,
                f"{avg_gen:.2f}",
                f"{max_gen:.2f}",
                f"{min_gen:.2f}",
                f"{correlation:.3f}",
                f"{avg_price_when_generating:.3f}"
            ])
    
    # Create table
    table_data = [['Technology', 'Avg Gen (GW)', 'Max Gen (GW)', 'Min Gen (GW)', 
                   'Price Correlation', 'Avg Price when Generating (M€/GWh)']]
    table_data.extend(stats_data)
    
    table = ax4.table(cellText=table_data[1:], colLabels=table_data[0], 
                     cellLoc='center', loc='center',
                     colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            if i == 0:  # Header row
                table[(i, j)].set_facecolor('#4CAF50')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax4.set_title('Price Response Analysis Summary', fontsize=16, pad=20)
    
    plt.suptitle('Comprehensive Price Response Analysis', fontsize=20, y=0.98)
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.3)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.3)
    
    return fig

def plot_comprehensive_generation_vs_price(scenarios_data, save_path=None):
    """Create a comprehensive generation vs. price analysis with combined and individual technology views."""
    
    # Define generator types and their colors - ONLY DISPATCHABLE
    generator_types = ['Baseload', 'MidMerit', 'Peak']
    colors = ['#8B4513', '#FF8C00', '#DC143C']  # Brown, Orange, Red
    
    # Create large figure with 2 rows: combined view on top, individual technologies below
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3, height_ratios=[1, 1])
    
    # Top row: Combined view (spans all 3 columns)
    ax_combined = fig.add_subplot(gs[0, :])
    
    # Plot all technologies on the same axes
    for gen_type, color in zip(generator_types, colors):
        all_prices = []
        all_generation = []
        
        for scen, df in scenarios_data.items():
            if f"G_{gen_type}" in df.columns and "Price" in df.columns:
                prices = df["Price"].values  # M€/GWh
                generation = df[f"G_{gen_type}"].values  # GW
                all_prices.extend(prices)
                all_generation.extend(generation)
        
        if all_prices and all_generation:
            # Create scatter plot with larger markers
            ax_combined.scatter(all_prices, all_generation, c=color, alpha=0.7, s=40, 
                              label=f'{gen_type}', edgecolors='white', linewidth=0.5)
            
            # Add trend line for this technology
            if len(all_prices) > 1:
                z = np.polyfit(all_prices, all_generation, 1)
                p = np.poly1d(z)
                ax_combined.plot(all_prices, p(all_prices), "--", color=color, alpha=0.8, linewidth=2.5)
    
    ax_combined.set_xlabel('Price (M€/GWh)', fontsize=16)
    ax_combined.set_ylabel('Generation (GW)', fontsize=16)
    ax_combined.set_title('Dispatchable Generation vs. Price - All Technologies Combined\n(All Scenarios)', fontsize=18, pad=20)
    ax_combined.legend(fontsize=14, loc='upper left')
    ax_combined.grid(True, alpha=0.3)
    ax_combined.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    
    # Add overall statistics
    all_prices_combined = []
    all_generation_combined = []
    for scen, df in scenarios_data.items():
        if "Price" in df.columns:
            prices = df["Price"].values
            for gen_type in generator_types:
                if f"G_{gen_type}" in df.columns:
                    generation = df[f"G_{gen_type}"].values
                    # Only add if both arrays have the same length
                    if len(prices) == len(generation):
                        all_prices_combined.extend(prices)
                        all_generation_combined.extend(generation)
    
    if all_prices_combined and all_generation_combined and len(all_prices_combined) == len(all_generation_combined):
        overall_correlation = np.corrcoef(all_prices_combined, all_generation_combined)[0, 1]
        stats_text = f'Overall Correlation: {overall_correlation:.3f}\nMax Price: {max(all_prices_combined):.3f} M€/GWh\nMin Price: {min(all_prices_combined):.3f} M€/GWh'
        ax_combined.text(0.02, 0.98, stats_text, transform=ax_combined.transAxes, 
                       fontsize=12, verticalalignment='top', weight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Bottom row: Individual technology plots
    for i, (gen_type, color) in enumerate(zip(generator_types, colors)):
        ax = fig.add_subplot(gs[1, i])
        
        # Collect data for this technology
        all_prices = []
        all_generation = []
        
        for scen, df in scenarios_data.items():
            if f"G_{gen_type}" in df.columns and "Price" in df.columns:
                prices = df["Price"].values  # M€/GWh
                generation = df[f"G_{gen_type}"].values  # GW
                all_prices.extend(prices)
                all_generation.extend(generation)
        
        if all_prices and all_generation:
            # Create scatter plot
            ax.scatter(all_prices, all_generation, c=color, alpha=0.8, s=50, 
                      edgecolors='white', linewidth=0.5)
            
            # Add trend line
            if len(all_prices) > 1:
                z = np.polyfit(all_prices, all_generation, 1)
                p = np.poly1d(z)
                ax.plot(all_prices, p(all_prices), "--", color=color, alpha=0.9, linewidth=3)
            
            # Calculate statistics
            correlation = np.corrcoef(all_prices, all_generation)[0, 1]
            avg_gen = np.mean(all_generation)
            max_gen = max(all_generation)
            min_gen = min(all_generation)
            avg_price_when_generating = np.mean([p for p, g in zip(all_prices, all_generation) if g > 0.01])
            
            # Add detailed statistics
            stats_text = f'Avg Gen: {avg_gen:.2f} GW\nMax Gen: {max_gen:.2f} GW\nMin Gen: {min_gen:.2f} GW\nCorrelation: {correlation:.3f}\nAvg Price when Gen: {avg_price_when_generating:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', weight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        ax.set_xlabel('Price (M€/GWh)', fontsize=12)
        ax.set_ylabel('Generation (GW)', fontsize=12)
        ax.set_title(f'{gen_type}\nGeneration vs. Price', fontsize=14, pad=10)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    
    plt.suptitle('Comprehensive Dispatchable Generation vs. Price Analysis\nCombined View and Individual Technology Breakdown', 
                fontsize=22, y=0.98)
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.3)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.3)
    
    return fig

def plot_generation_vs_price_by_scenario_dispatchable(scenarios_data, save_path=None):
    """Create separate scatter plots for each scenario showing dispatchable generation vs. prices for each technology."""
    
    # Define generator types and their colors - ONLY DISPATCHABLE
    generator_types = ['Baseload', 'MidMerit', 'Peak']
    colors = ['#8B4513', '#FF8C00', '#DC143C']  # Brown, Orange, Red
    
    scenarios = sorted(list(scenarios_data.keys()))
    n_scenarios = len(scenarios)
    
    # Create figure with subplots for each scenario
    fig, axes = plt.subplots(n_scenarios, 1, figsize=(16, 5*n_scenarios))
    if n_scenarios == 1:
        axes = [axes]
    
    for i, (scen, df) in enumerate(scenarios_data.items()):
        ax = axes[i]
        
        # Plot each generator type
        for j, (gen_type, color) in enumerate(zip(generator_types, colors)):
            if f"G_{gen_type}" in df.columns and "Price" in df.columns:
                prices = df["Price"].values  # M€/GWh
                generation = df[f"G_{gen_type}"].values  # GW
                
                # Create scatter plot
                ax.scatter(prices, generation, c=color, alpha=0.7, s=30, 
                          label=f'{gen_type}', edgecolors='white', linewidth=0.5)
                
                # Add trend line for this generator type
                if len(prices) > 1 and np.any(generation > 0):
                    # Only fit trend line if there's actual generation
                    z = np.polyfit(prices, generation, 1)
                    p = np.poly1d(z)
                    ax.plot(prices, p(prices), "--", color=color, alpha=0.8, linewidth=1.5)
        
        ax.set_xlabel('Price (M€/GWh)', fontsize=12)
        ax.set_ylabel('Generation (GW)', fontsize=12)
        ax.set_title(f'Scenario {scen} - {SCENARIO_NAMES.get(scen, f"Scenario {scen}")}\nDispatchable Generation vs. Price by Technology', 
                    fontsize=14, pad=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format price axis to show more decimal places
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.2)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)
    
    return fig

def plot_dispatchable_generation_summary(scenarios_data, save_path=None):
    """Plot dispatchable generation summary with dynamic generator detection."""
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 3, figure=fig)
    
    # Dynamically detect available generators
    if not scenarios_data:
        print("No scenario data provided to plot_dispatchable_generation_summary; skipping.")
        plt.close(fig)
        return None
    sample_df = next(iter(scenarios_data.values()))  # Use first available scenario to detect columns
    generator_columns = [col for col in sample_df.columns if col.startswith('G_') and not any(c in col for c in ["LV_MED", "MV_LOAD", "LV_HIGH", "LV_LOW"])]
    generators = [col[2:] for col in generator_columns]  # Remove "G_" prefix
    
    # Filter out wind generators (they have different behavior)
    dispatchable_generators = [gen for gen in generators if not gen.startswith('Wind')]
    
    if not dispatchable_generators:
        plt.text(0.5, 0.5, 'No dispatchable generators found', ha='center', va='center', transform=plt.gca().transAxes)
        if save_path:
            plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.2)
            png_path = str(save_path).replace('.svg', '.png')
            plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        return fig
    
    # Top plot: Combined view
    ax1 = fig.add_subplot(gs[0, :])
    colors = ['brown', 'orange', 'red', 'green', 'purple']
    
    for i, gen in enumerate(dispatchable_generators):
        if f'G_{gen}' in sample_df.columns:
            gen_data = sample_df[f'G_{gen}'].values
            price_data = sample_df['Price'].values
            ax1.scatter(price_data, gen_data, alpha=0.6, s=20, 
                       label=f'{gen} (S1)', color=colors[i % len(colors)], marker='s')
            
            # Add trend line
            z = np.polyfit(price_data, gen_data, 1)
            p = np.poly1d(z)
            ax1.plot(price_data, p(price_data), '--', color=colors[i % len(colors)], 
                    alpha=0.8, linewidth=2, label=f'{gen} Trend')
    
    ax1.set_xlabel('Price (M€/GWh)')
    ax1.set_ylabel('Generation (GW)')
    ax1.set_title('Dispatchable Generation vs. Price - All Technologies and Scenarios (Combined View)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Middle row: Individual plots
    for i, gen in enumerate(dispatchable_generators):
        if i >= 3:  # Only show first 3 generators
            break
        ax = fig.add_subplot(gs[1, i])
        
        if f'G_{gen}' in sample_df.columns:
            gen_data = sample_df[f'G_{gen}'].values
            price_data = sample_df['Price'].values
            
            ax.scatter(price_data, gen_data, alpha=0.6, s=20, 
                      color=colors[i % len(colors)], marker='s')
            
            # Add trend line
            z = np.polyfit(price_data, gen_data, 1)
            p = np.poly1d(z)
            ax.plot(price_data, p(price_data), '--', color=colors[i % len(colors)], 
                   alpha=0.8, linewidth=2)
            
            # Add statistics text box
            correlation = np.corrcoef(price_data, gen_data)[0, 1]
            avg_gen = np.mean(gen_data)
            max_gen = np.max(gen_data)
            min_gen = np.min(gen_data)
            
            stats_text = f'Correlation: {correlation:.3f}\nAvg Gen: {avg_gen:.2f} GW\nMax Gen: {max_gen:.2f} GW\nMin Gen: {min_gen:.2f} GW'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top')
        
        ax.set_xlabel('Price (M€/GWh)')
        ax.set_ylabel('Generation (GW)')
        ax.set_title(f'{gen}: Generation vs. Price')
        ax.grid(True, alpha=0.3)
    
    # Bottom left: Price distribution
    ax_dist = fig.add_subplot(gs[2, 0])
    price_data = sample_df['Price'].values
    ax_dist.hist(price_data, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax_dist.set_xlabel('Price (M€/GWh)')
    ax_dist.set_ylabel('Frequency')
    ax_dist.set_title('Price Distribution Across All Scenarios')
    ax_dist.grid(True, alpha=0.3)
    
    # Bottom right: Statistics table
    ax_table = fig.add_subplot(gs[2, 1:])
    ax_table.axis('tight')
    ax_table.axis('off')
    
    # Create statistics table
    table_data = []
    headers = ['Technology', 'Avg Gen (GW)', 'Max Gen (GW)', 'Min Gen (GW)', 'Price Correlation', 'Price when Gen > 0 (M€/GWh)']
    
    for gen in dispatchable_generators:
        if f'G_{gen}' in sample_df.columns:
            gen_data = sample_df[f'G_{gen}'].values
            price_data = sample_df['Price'].values
            
            correlation = np.corrcoef(price_data, gen_data)[0, 1]
            avg_gen = np.mean(gen_data)
            max_gen = np.max(gen_data)
            min_gen = np.min(gen_data)
            
            # Find price when generation > 0
            positive_gen_mask = gen_data > 0
            price_when_gen_positive = np.mean(price_data[positive_gen_mask]) if np.any(positive_gen_mask) else 0
            
            table_data.append([
                gen,
                f'{avg_gen:.2f}',
                f'{max_gen:.2f}',
                f'{min_gen:.2f}',
                f'{correlation:.3f}',
                f'{price_when_gen_positive:.3f}'
            ])
    
    if table_data:
        table = ax_table.table(cellText=table_data, colLabels=headers, 
                              cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
    
    plt.suptitle('Generation vs. Price Analysis - Combined View, Individual Technologies, and Statistics', 
                 fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.2)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)
    
    plt.close(fig)
    return fig

def plot_total_load_duration_curves(scenarios_data, save_path=None):
    """Plot total load duration curves for all scenarios on one graph."""
    import matplotlib.pyplot as plt
    import os
    plt.figure(figsize=(14, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    for i, (scen, df) in enumerate(scenarios_data.items()):
        total_load = df[[col for col in df.columns if 'Total_Demand' in col][0]]
        load_sorted = total_load.sort_values(ascending=False).values
        plt.plot(range(len(load_sorted)), load_sorted, linewidth=2.5, 
                 color=colors[i % len(colors)], 
                 label=scen)
    plt.xlabel('Hour (sorted)')
    plt.ylabel('Total Load (GW)')
    plt.title('Total Load Duration Curve (All Scenarios)')
    plt.legend()
    plt.tight_layout()
    if save_path is None:
        save_path = 'output/plots/'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'total_load_duration_curves.png'))
    plt.savefig(os.path.join(save_path, 'total_load_duration_curves.svg'))
    plt.close()

def plot_residual_load_duration_curves(scenarios_data, save_path=None):
    """Plot residual load duration curves for all scenarios on one graph."""
    import matplotlib.pyplot as plt
    import os
    plt.figure(figsize=(14, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    for i, (scen, df) in enumerate(scenarios_data.items()):
        # Residual load = Total_Demand - sum of all renewables (columns starting with 'G_Wind', 'G_Solar', etc.)
        total_load = df[[col for col in df.columns if 'Total_Demand' in col][0]]
        renewable_cols = [col for col in df.columns if col.startswith('G_Wind') or col.startswith('G_Solar') or col.startswith('G_PV')]
        if renewable_cols:
            renewables = df[renewable_cols].sum(axis=1)
        else:
            renewables = 0
        residual_load = total_load - renewables
        load_sorted = residual_load.sort_values(ascending=False).values
        plt.plot(range(len(load_sorted)), load_sorted, linewidth=2.5, 
                 color=colors[i % len(colors)], 
                 label=SCENARIO_NAMES.get(scen, f"Scenario {scen}"))
    plt.xlabel('Hour (sorted)', fontsize=14)
    plt.ylabel('Residual Load (GW)', fontsize=14)
    plt.title('Residual Load Duration Curve (All Scenarios)', fontsize=18, pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if save_path is None:
        save_path = 'output/plots/'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'residual_load_duration_curve.png'), bbox_inches='tight', pad_inches=0.2)
    plt.savefig(os.path.join(save_path, 'residual_load_duration_curve.svg'), format='svg', bbox_inches='tight', pad_inches=0.2)
    plt.close()

def plot_individual_residual_load_duration_curves(scenarios_data, save_path=None):
    """For each scenario, plot individual residual load duration curves per dispatchable generator."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # Ensure save_path is a Path object and resolve to absolute path
    if save_path is None:
        save_path = Path('output/plots/')
    else:
        save_path = Path(save_path)
        
    try:
        # Resolve to absolute path to avoid Windows relative path issues
        if not save_path.is_absolute():
            save_path = save_path.resolve()
    except Exception:
        # Fallback if resolve fails (e.g. strict checks)
        pass
        
    save_path.mkdir(parents=True, exist_ok=True)
    
    for scen, df in scenarios_data.items():
        # Identify dispatchable generators (not wind/solar/pv)
        generator_cols = [col for col in df.columns if col.startswith('G_') and not any(x in col for x in ['Wind', 'Solar', 'PV'])]
        
        if not generator_cols:
            continue
            
        total_load = df[[col for col in df.columns if 'Total_Demand' in col][0]]
        renewable_cols = [col for col in df.columns if col.startswith('G_Wind') or col.startswith('G_Solar') or col.startswith('G_PV')]
        if renewable_cols:
            renewables = df[renewable_cols].sum(axis=1)
        else:
            renewables = 0
        residual_load = total_load - renewables
        
        plt.figure(figsize=(14, 8))
        for col in generator_cols:
            gen_output = df[col]
            # Individual residual = residual load covered by this generator (could be just its output)
            gen_sorted = gen_output.sort_values(ascending=False).values
            plt.plot(range(len(gen_sorted)), gen_sorted, linewidth=2.5, label=col.replace('G_', ''))
        plt.xlabel('Hour (sorted)')
        plt.ylabel('Generator Output (GW)')
        plt.title(f'Individual Residual Load Duration Curves ({scen})')
        plt.legend()
        plt.tight_layout()
        
        # Save using string path to avoid potential Path object issues on Windows
        try:
            output_file = save_path / f'individual_residual_load_duration_curve_{scen}.png'
            plt.savefig(str(output_file))
            output_svg = save_path / f'individual_residual_load_duration_curve_{scen}.svg'
            plt.savefig(str(output_svg))
        except Exception as e:
            print(f"  ⚠️  Could not save individual residual plot for scenario {scen}: {e}")
        finally:
            plt.close()




