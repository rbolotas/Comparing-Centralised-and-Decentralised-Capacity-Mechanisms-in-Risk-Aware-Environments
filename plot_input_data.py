"""
Plot weather and load data analysis for representative days.
Generates publication-quality figures for thesis documentation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path
import warnings
from config_utils import get_ntimesteps
from config_utils import get_config_value
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.1)

# FIXED: Color scheme with correct column names
COLORS = {
    'WIND_ONSHORE': '#1f77b4',    # Clean blue
    'WIND_OFFSHORE': '#2E4B82',   # Darker blue  
    'PV': '#ff7f0e',              # Clean orange
    'LV_LOW': '#2ca02c',          # Clean green
    'LV_MED': '#d62728',          # Clean red
    'LV_HIGH': '#9467bd',         # Clean purple
    'MV_LOAD': '#8c564b',         # FIXED: Correct column name
    'HV_LOAD': '#7f7f7f',         # Gray for industrial load
}

def load_data_files():
    """Load weather and demand data from separate files."""
    try:
        # Get filenames from config
        weather_file = get_config_value("Files", "weather_timeseries", "ts_RES_AF_12d.csv")
        demand_file = get_config_value("Files", "demand_timeseries", "ts_demand_12d.csv")
        
        # Load data
        weather_df = pd.read_csv(f"../Input/{weather_file}")
        demand_df = pd.read_csv(f"../Input/{demand_file}")
        
        # Merge on timestamp
        df = pd.concat([weather_df, demand_df], axis=1)
        
        # FIXED: Ensure we have exactly N_TIMESTEPS hours after merging
        n_timesteps = get_ntimesteps()
        if len(df) != n_timesteps:
            print(f"⚠️  Warning: Expected {n_timesteps} hours after merge, got {len(df)} hours")
            # Keep only first N_TIMESTEPS rows if we have more
            if len(df) > n_timesteps:
                df = df.iloc[:n_timesteps].copy()
                print(f"✓ Trimmed to {n_timesteps} hours")
        
        return df
        
    except FileNotFoundError as e:
        print(f"❌ Error loading data files: {e}")
        raise

def load_weather_data(file_path):
    """Load weather data - compatibility function for analyze.py."""
    return load_data_files()



def load_scaled_demand_data(scenario_num=1, variant="ref"):
    """Load scaled demand data from Results files - data is already properly scaled by Julia model."""
    try:
        file_path = f"../Results/Scenario_{scenario_num}_{variant}.csv"
        df = pd.read_csv(file_path, delimiter=";")
        
        # Extract demand columns from results - these are already properly scaled by Julia
        # The data is in GW and represents total demand for each consumer segment
        scaled_df = pd.DataFrame()
        
        # Use the Input_D_* columns from Results (already scaled by share * totConsumers)
        column_mapping = {
            'LV_LOW': 'Input_D_LV_LOW',
            'LV_MED': 'Input_D_LV_MED', 
            'LV_HIGH': 'Input_D_LV_HIGH',
            'MV_LOAD': 'Input_D_MV_LOAD',
            'HV_LOAD': 'HV_LOAD'  # This one doesn't have Input_D_ prefix
        }
        
        for consumer_type, column_name in column_mapping.items():
            if column_name in df.columns:
                scaled_df[consumer_type] = df[column_name]
            else:
                print(f"⚠️  Warning: Column {column_name} not found in Results file")
                scaled_df[consumer_type] = 0
        
        print(f"✅ Loaded scaled demand data from {file_path}")
        print(f"   Sample values: LV_LOW={scaled_df['LV_LOW'].iloc[0]:.3f}, MV_LOAD={scaled_df['MV_LOAD'].iloc[0]:.3f}")
        return scaled_df
        
    except Exception as e:
        print(f"❌ Error loading scaled demand data: {e}")
        return None

def plot_weather_profiles(df, save_path=None):
    """Plot weather profiles - compatibility function for analyze.py."""
    return plot_weather_profiles_internal(df, save_path)

def plot_reference_demand_profiles(df, save_path=None):
    """Plot demand profiles - compatibility function for analyze.py."""
    return plot_reference_demand_profiles_internal(df, save_path)



def plot_scaled_demand_profiles_internal(df, save_path=None):
    """Plot demand profiles using scaled data from Results files (in GW)."""
    n_timesteps = get_ntimesteps()
    if len(df) != n_timesteps:
        print(f"⚠️  Warning: Expected {n_timesteps} hours, got {len(df)} hours")
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    hours = np.arange(0, len(df))
    
    # Plot scaled demand data (already in GW)
    ax1.plot(hours, df['LV_LOW'], color=COLORS['LV_LOW'], linewidth=1.5, alpha=0.8, label='LV Low')
    ax1.plot(hours, df['LV_MED'], color=COLORS['LV_MED'], linewidth=1.5, alpha=0.8, label='LV Medium')
    ax1.plot(hours, df['LV_HIGH'], color=COLORS['LV_HIGH'], linewidth=1.5, alpha=0.8, label='LV High')
    ax1.set_ylabel('Load (GW)', fontsize=12)
    ax1.set_title('Low Voltage Consumer Segments', fontsize=14, pad=10)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add some statistics to show the scaling
    lv_avg = df[['LV_LOW', 'LV_MED', 'LV_HIGH']].mean().mean()
    ax1.text(0.02, 0.98, f'LV Avg: {lv_avg:.3f} GW', transform=ax1.transAxes, 
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.plot(hours, df['LV_LOW'], color=COLORS['LV_LOW'], linewidth=1.5, alpha=0.8, label='LV Low')
    ax2.plot(hours, df['LV_MED'], color=COLORS['LV_MED'], linewidth=1.5, alpha=0.8, label='LV Medium')
    ax2.plot(hours, df['LV_HIGH'], color=COLORS['LV_HIGH'], linewidth=1.5, alpha=0.8, label='LV High')
    ax2.plot(hours, df['MV_LOAD'], color=COLORS['MV_LOAD'], linewidth=1.5, alpha=0.8, label='MV Industrial')
    ax2.set_ylabel('Load (GW)', fontsize=12)
    ax2.set_title('All Consumer Agent Types', fontsize=14, pad=10)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add statistics to show comparison
    mv_avg = df['MV_LOAD'].mean()
    ax2.text(0.02, 0.98, f'MV Avg: {mv_avg:.3f} GW', transform=ax2.transAxes, 
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Compute total system load like the model: sum scaled consumer demand + HV industrial load
    from config_utils import load_config
    cfg = load_config()
    tot = cfg['General']['totConsumers']
    consumer_total = 0.0
    for name, c in cfg['Consumers'].items():
        d_col = c['D']
        share = c['Share']
        if d_col in df.columns:
            consumer_total = consumer_total + (tot * share * df[d_col])
    hv = df['HV_LOAD'] if 'HV_LOAD' in df.columns else 0.0
    total_load = consumer_total + hv
    
    # Plot HV_LOAD and Total System Load
    ax3.plot(hours, hv, color='brown', linewidth=1.5, alpha=0.7, label='HV Industrial Load', linestyle='--')
    ax3.plot(hours, total_load, color='black', linewidth=2, alpha=0.8, label='Total System Load')
    ax3.set_ylabel('Load (GW)', fontsize=12)
    ax3.set_title('HV Industrial and Total System Load', fontsize=14, pad=10)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Add total load statistics
    total_avg = total_load.mean()
    hv_avg = hv.mean() if isinstance(hv, pd.Series) else hv
    ax3.text(0.02, 0.98, f'Total Avg: {total_avg:.3f} GW\nHV Avg: {hv_avg:.3f} GW', 
             transform=ax3.transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Day boundaries every 24 hours
    n_timesteps = get_ntimesteps()
    n_days = n_timesteps // 24
    day_boundaries = [24 * day for day in range(1, n_days)]
    for boundary in day_boundaries:
        for ax in [ax1, ax2, ax3]:
            ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.6, linewidth=1)
    
    # X-axis setup
    n_timesteps = get_ntimesteps()
    n_days = n_timesteps // 24
    ax3.set_xlabel(f'Time (Hours across {n_days} Representative Days)', fontsize=12)
    ax3.set_xlim(0, n_timesteps-1)
    
    # Day labels
    day_centers = [12 + 24*day for day in range(n_days)]
    day_labels = [f'D{day+1}' for day in range(n_days)]
    ax3.set_xticks(day_centers)
    ax3.set_xticklabels(day_labels, fontsize=10)
    
    n_days = get_ntimesteps() // 24
    plt.suptitle(f'Reference Input Demand for {n_days} Representative Days\n(Before Elastic Response)', fontsize=16, y=0.99)
    plt.tight_layout()
    
    _save_plot(save_path)
    return fig

def _save_plot(save_path):
    """Helper function to save plots in both SVG and PNG formats."""
    if save_path:
        save_path_str = str(save_path)
        plt.savefig(save_path_str, format='svg', dpi=300, bbox_inches='tight')
        png_path = save_path_str.replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')

def plot_weather_profiles_internal(df, save_path=None):
    """Plot clean weather profiles for representative days."""
    
    # Ensure we have exactly N_TIMESTEPS hours 
    n_timesteps = get_ntimesteps()
    if len(df) != n_timesteps:
        print(f"⚠️  Warning: Expected {n_timesteps} hours, got {len(df)} hours")
    
    # Check if offshore wind is actually used in the simulation by examining Results data
    try:
        # Load a sample Results file to check if offshore wind generation exists
        results_df = pd.read_csv("../Results/Scenario_1_ref.csv", delimiter=";")
        offshore_used = 'G_WindOffshore' in results_df.columns and results_df['G_WindOffshore'].sum() > 0
    except:
        # Fallback: check if offshore wind exists in input weather data and is nonzero
        offshore_used = 'WIND_OFFSHORE' in df.columns and df['WIND_OFFSHORE'].sum() > 0
    
    # Create figure with proper subplot layout
    if offshore_used:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        axes = [ax1, ax2, ax3]
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        axes = [ax1, ax2]
    
    # Create explicit hour sequence from 0 to 287
    hours = np.arange(0, len(df))
    
    # Plot 1: Wind Onshore
    axes[0].plot(hours, df['WIND_ONSHORE'], color=COLORS['WIND_ONSHORE'], 
                linewidth=1.5, alpha=0.8)
    axes[0].set_ylabel('Wind Onshore\nCapacity Factor', fontsize=12)
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Wind Onshore Generation Profile', fontsize=14, pad=10)
    
    # Plot 2: Wind Offshore (only if actually used in simulation)
    if offshore_used:
        axes[1].plot(hours, df['WIND_OFFSHORE'], color=COLORS['WIND_OFFSHORE'], 
                    linewidth=1.5, alpha=0.8)
        axes[1].set_ylabel('Wind Offshore\nCapacity Factor', fontsize=12)
        axes[1].set_ylim(0, 1.0)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('Wind Offshore Generation Profile', fontsize=14, pad=10)
        solar_idx = 2
    else:
        solar_idx = 1
    
    # Plot 3: Solar
    axes[solar_idx].plot(hours, df['PV'], color=COLORS['PV'], 
                        linewidth=1.5, alpha=0.8)
    axes[solar_idx].set_ylabel('Solar PV\nCapacity Factor', fontsize=12)
    axes[solar_idx].set_ylim(0, 1.0)
    axes[solar_idx].grid(True, alpha=0.3)
    axes[solar_idx].set_title('Solar PV Generation Profile', fontsize=14, pad=10)
    
    # Day boundaries - vertical lines every 24 hours
    n_timesteps = get_ntimesteps()
    n_days = n_timesteps // 24
    day_boundaries = [24 * day for day in range(1, n_days)]
    
    for boundary in day_boundaries:
        for ax in axes:
            ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.6, linewidth=1)
    
    # X-axis setup - dynamic based on timesteps
    n_timesteps = get_ntimesteps()
    n_days = n_timesteps // 24  # Calculate number of days
    axes[-1].set_xlabel(f'Time (Hours across {n_days} Representative Days)', fontsize=12)
    axes[-1].set_xlim(0, n_timesteps-1)  # 0-based indexing, so n_timesteps-1 is the last hour
    
    # Day labels - place at center of each day
    day_centers = [12 + 24*day for day in range(n_days)]  
    day_labels = [f'D{day+1}' for day in range(n_days)]
    
    axes[-1].set_xticks(day_centers)
    axes[-1].set_xticklabels(day_labels, fontsize=10)
    
    n_days = get_ntimesteps() // 24
    plt.suptitle(f'Weather Profiles for {n_days} Representative Days', fontsize=16, y=0.98)
    plt.tight_layout()
    
    _save_plot(save_path)
    return fig

def plot_reference_demand_profiles_internal(df, save_path=None):
    """Plot clean demand profiles - 3 subplots as requested, all in GW."""
    n_timesteps = get_ntimesteps()
    if len(df) != n_timesteps:
        print(f"⚠️  Warning: Expected {n_timesteps} hours, got {len(df)} hours")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    hours = np.arange(0, len(df))
    # Load data already in GW
    ax1.plot(hours, df['LV_LOW'], color=COLORS['LV_LOW'], linewidth=1.5, alpha=0.8, label='LV Low')
    ax1.plot(hours, df['LV_MED'], color=COLORS['LV_MED'], linewidth=1.5, alpha=0.8, label='LV Medium')
    ax1.plot(hours, df['LV_HIGH'], color=COLORS['LV_HIGH'], linewidth=1.5, alpha=0.8, label='LV High')
    ax1.set_ylabel('Load (GW)', fontsize=12)
    ax1.set_title('Low Voltage Consumer Segments', fontsize=14, pad=10)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add statistics to show scaling
    lv_avg = df[['LV_LOW', 'LV_MED', 'LV_HIGH']].mean().mean()
    ax1.text(0.02, 0.98, f'LV Avg: {lv_avg:.3f} GW', transform=ax1.transAxes, 
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.plot(hours, df['LV_LOW'], color=COLORS['LV_LOW'], linewidth=1.5, alpha=0.8, label='LV Low')
    ax2.plot(hours, df['LV_MED'], color=COLORS['LV_MED'], linewidth=1.5, alpha=0.8, label='LV Medium')
    ax2.plot(hours, df['LV_HIGH'], color=COLORS['LV_HIGH'], linewidth=1.5, alpha=0.8, label='LV High')
    ax2.plot(hours, df['MV_LOAD'], color=COLORS['MV_LOAD'], linewidth=1.5, alpha=0.8, label='MV Industrial')
    ax2.set_ylabel('Load (GW)', fontsize=12)
    ax2.set_title('All Consumer Agent Types', fontsize=14, pad=10)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add statistics to show comparison
    mv_avg = df['MV_LOAD'].mean()
    ax2.text(0.02, 0.98, f'MV Avg: {mv_avg:.3f} GW', transform=ax2.transAxes, 
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Third subplot: Show both HV_LOAD and Total System Load
    # Plot HV_LOAD
    ax3.plot(hours, df['HV_LOAD'], color='brown', linewidth=1.5, alpha=0.7, label='HV Industrial Load', linestyle='--')
    
    # Calculate total system load: sum of all scaled agent demand
    # Get config to properly scale agent demands
    from config_utils import load_config
    cfg = load_config()
    tot = cfg['General']['totConsumers']
    
    # Calculate total from scaled consumer demand
    consumer_total = 0.0
    for name, c in cfg['Consumers'].items():
        d_col = c['D']
        share = c['Share']
        if d_col in df.columns:
            consumer_total = consumer_total + (tot * share * df[d_col])
    
    # Add HV industrial load
    total_system_load = consumer_total + df['HV_LOAD']
    
    # Plot total system load
    ax3.plot(hours, total_system_load, color='black', linewidth=2, alpha=0.8, label='Total System Load')
    
    ax3.set_ylabel('Load (GW)', fontsize=12)
    ax3.set_title('HV Industrial and Total System Load', fontsize=14, pad=10)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Add statistics
    total_avg = total_system_load.mean()
    hv_avg = df['HV_LOAD'].mean()
    ax3.text(0.02, 0.98, f'Total Avg: {total_avg:.3f} GW\nHV Avg: {hv_avg:.3f} GW', 
             transform=ax3.transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Day boundaries every 24 hours
    n_timesteps = get_ntimesteps()
    n_days = n_timesteps // 24
    day_boundaries = [24 * day for day in range(1, n_days)]
    for boundary in day_boundaries:
        for ax in [ax1, ax2, ax3]:
            ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.6, linewidth=1)
    
    # X-axis setup
    n_timesteps = get_ntimesteps()
    n_days = n_timesteps // 24
    ax3.set_xlabel(f'Time (Hours across {n_days} Representative Days)', fontsize=12)
    ax3.set_xlim(0, n_timesteps-1)
    
    # Day labels
    day_centers = [12 + 24*day for day in range(n_days)]
    day_labels = [f'D{day+1}' for day in range(n_days)]
    ax3.set_xticks(day_centers)
    ax3.set_xticklabels(day_labels, fontsize=10)
    
    n_days = get_ntimesteps() // 24
    plt.suptitle(f'Reference Input Demand for {n_days} Representative Days\n(Before Elastic Response)', fontsize=16, y=0.99)
    plt.tight_layout()
    
    _save_plot(save_path)
    return fig



# Main functions
def plot_representative_days_load_profiles(save_path=None):
    """Main function for load profiles - uses properly scaled data from Results."""
    # Use properly scaled data from Results instead of raw input data
    df = load_scaled_demand_data(scenario_num=1, variant="ref")
    if df is None:
        print("❌ Could not load scaled demand data, falling back to raw data")
        # Fallback to raw data if scaled data loading fails
        df = load_data_files()
    return plot_scaled_demand_profiles_internal(df, save_path)

def plot_representative_days_weather_profiles(save_path=None):
    """Main function for weather profiles."""
    df = load_data_files()
    return plot_weather_profiles_internal(df, save_path)

def plot_generator_cost_curves(save_path=None):
    """
    Plot Long Run Marginal Cost (LRMC) including annualized investment cost.
    In the long run, all inputs are variable, including capital investment.
    LRMC = SRMC + Annualized Investment Cost per GWh
    LRMC(q) = A × q + B + INV_h
    where INV_h is the annualized hourly investment cost (M€/GW/h)
    """
    from config_utils import load_config
    
    config = load_config()
    generators = config['Generators']
    
    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Define output range for plotting (0 to 10 GW)
    q_range = np.linspace(0, 10, 100)
    
    # Technology colors (matching other plots)
    TECH_COLORS = {
        'Baseload': '#8B4513',   # Saddle Brown
        'MidMerit': '#FF8C00',   # Dark Orange
        'Peak': '#DC143C',       # Crimson
        'WindOnshore': '#2E8B57' # Sea Green
    }
    
    # Collect data for cost table
    cost_data = []
    
    # Plot: Marginal Cost + Investment Cost
    for gen_name, params in generators.items():
        a = params['a']  # Quadratic coefficient (M€/GWh)²
        b = params['b']  # Linear coefficient (M€/GWh)
        inv = params['INV']  # Total investment (M€/GW)
        n_years = params['n']  # Lifetime in years
        
        # Calculate short-run marginal cost: SRMC(q) = A × q + B
        # (derivative of variable cost A/2 × q² + B × q)
        srmc_curve = a * q_range + b
        
        # Calculate annualized hourly investment cost
        discount_rate = config['General']['r']
        inv_h = (inv / n_years) / 8760  # M€/GW/h
        
        # Long-run marginal cost includes investment cost
        # LRMC(q) = SRMC(q) + INV_h = A × q + B + INV_h
        lrmc_curve = srmc_curve + inv_h
        
        # Plot LRMC curve
        color = TECH_COLORS.get(gen_name, '#666666')
        ax.plot(q_range, lrmc_curve, linewidth=2.5, label=gen_name, color=color, alpha=0.8)
        
        # Add horizontal line for base LRMC (B + INV_h) at q=0
        base_lrmc = b + inv_h
        ax.axhline(y=base_lrmc, color=color, linestyle='--', alpha=0.2, linewidth=1)
        
        # Collect data for cost table
        lrmc_at_1gw = a * 1.0 + b + inv_h
        lrmc_at_5gw = a * 5.0 + b + inv_h
        
        cost_data.append({
            'Generator': gen_name,
            'B (Linear)': b,
            'A (Quadratic)': a,
            'INV_h': inv_h,
            'LRMC @ 1GW': lrmc_at_1gw,
            'LRMC @ 5GW': lrmc_at_5gw,
            'color': color
        })
    
    # Sort by LRMC at 1 GW for merit order
    cost_data.sort(key=lambda x: x['LRMC @ 1GW'])
    
    ax.set_xlabel('Generation Output (GW)', fontsize=14)
    ax.set_ylabel('Long-Run Marginal Cost (M€/GWh)', fontsize=14)
    ax.set_title('Long-Run Marginal Cost (LRMC)\nLRMC(q) = A×q + B + INV_h\n(Including Annualized Investment Cost)', 
                fontsize=16, pad=20)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, max([d['LRMC @ 5GW'] for d in cost_data]) * 1.1)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    
    # Add text box with merit order at 1 GW
    merit_order_text = "Merit Order @ 1GW:\n"
    for i, item in enumerate(cost_data, 1):
        merit_order_text += f"{i}. {item['Generator']}: {item['LRMC @ 1GW']:.3f} M€/GWh\n"
    
    ax.text(0.98, 0.35, merit_order_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='center', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    _save_plot(save_path)
    
    return fig 