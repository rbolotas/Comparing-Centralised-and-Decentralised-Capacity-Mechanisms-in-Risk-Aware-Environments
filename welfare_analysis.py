"""
Welfare Analysis Module - Economic Surplus Calculations
"""
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from compare import scenario_names

def load_demand_curve_parameters(config_file="../Input/config.yaml"):
    """Load demand curve parameters from config file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    consumers = config['Consumers']
    total_consumers = config['General']['totConsumers']
    
    demand_curves = {}
    
    for consumer_type, params in consumers.items():
        if params is None or not isinstance(params, dict):
            continue
        
        # FIXED: Handle both Greek lambda (Ξ») and Latin lambda (λ) 
        def get_param(key_base):
            # Try both Greek and Latin lambda
            for prefix in ['λ_', 'Ξ»_']:
                key = prefix + key_base
                if key in params:
                    return params[key]
            return None
        
        # Check for required parameters with flexible lambda handling
        lambda_ref = get_param('EOM_ref')
        e_eom = params.get('E_EOM')
        lambda_cap = get_param('EOM_cap')
        share = params.get('Share')
        
        if lambda_ref is None or e_eom is None or lambda_cap is None or share is None:
            continue
        
        # Calculate actual number of this consumer type
        n_agents = int(total_consumers * share)
        
        demand_curves[consumer_type] = {
            'n_agents': n_agents,
            'lambda_ref': lambda_ref,
            'e_eom': e_eom,
            'lambda_cap': lambda_cap,
            'share': share
        }
    
    return demand_curves

def load_generator_cost_parameters(config_file="../Input/config.yaml"):
    """Load generator marginal cost parameters from config file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    generators = config['Generators']
    
    cost_curves = {}
    for gen_type, params in generators.items():
        # Marginal cost curve: MC(q) = b + a*q
        a = params['a']  # Quadratic coefficient (M€/GWh)^2
        b = params['b']  # Linear marginal cost (M€/GWh)
        
        cost_curves[gen_type] = {
            'a': a,  # Quadratic coefficient
            'b': b,  # Linear marginal cost
        }
    
    return cost_curves

def calculate_consumer_surplus_timestep(consumer_type, t, quantity_consumed, market_price, params_df):
    """
    Calculate consumer surplus for a specific consumer type and timestep.
    Uses time series demand curve parameters.
    """
    # Get parameters for this timestep
    row = params_df.iloc[t]
    lambda_cap = row['lambda_EOM_cap']
    e_eom = row['E_EOM']
    # You can also use lambda_EOM_0 if you want the more general form

    if quantity_consumed <= 0:
        return 0

    cs = quantity_consumed * (lambda_cap - market_price) + e_eom * (quantity_consumed ** 2) / 2
    return max(0, cs)

def calculate_consumer_surplus(consumer_type, quantity_consumed, market_price, demand_params):
    """
    Calculate consumer surplus for a specific consumer type.
    
    CS = ∫[0 to q] P_D(x) dx - P* × q
    
    Where P_D(q) is the inverse demand curve.
    From the Julia code, the demand relationship is:
    D = (λ_EOM - λ_EOM_0) / E_EOM
    So inverse demand is: P_D(q) = λ_EOM_0 + E_EOM * q
    
    But we need to be careful about the reference point. 
    Looking at the code, λ_EOM_0 is calculated as:
    λ_EOM_0 = λ_EOM_ref - (D_EOM_ref * E_EOM)
    
    However, since we're using representative periods, let's use the price cap as the demand intercept.
    """
    
    # Use price cap as the maximum willingness to pay (demand intercept)
    # Inverse demand: P_D(q) = λ_cap + E_EOM * q (where E_EOM is negative)
    lambda_cap = demand_params['lambda_cap']
    e_eom = demand_params['e_eom']  # Negative slope
    
    # Consumer surplus = Area under demand curve above market price
    # CS = ∫[0 to q] (λ_cap + E_EOM * x) dx - P* × q
    # CS = λ_cap * q + E_EOM * q²/2 - P* × q
    # CS = q * (λ_cap - P*) + E_EOM * q²/2
    
    if quantity_consumed <= 0:
        return 0
    
    cs = quantity_consumed * (lambda_cap - market_price) + e_eom * (quantity_consumed ** 2) / 2
    
    return max(0, cs)  # Ensure non-negative

def calculate_producer_surplus(gen_type, quantity_produced, market_price, cost_params):
    """
    Calculate producer surplus for a specific generator.
    
    PS = P* × q - ∫[0 to q] MC(x) dx
    
    Where MC(q) = b + a*q
    """
    
    if quantity_produced <= 0:
        return 0
    
    a = cost_params['a']  # Quadratic coefficient  
    b = cost_params['b']  # Linear marginal cost
    
    # Producer surplus = Revenue - Variable costs
    # PS = P* × q - ∫[0 to q] (b + a*x) dx
    # PS = P* × q - (b*q + a*q²/2)
    # PS = q * (P* - b) - a*q²/2
    
    ps = quantity_produced * (market_price - b) - a * (quantity_produced ** 2) / 2
    
    return max(0, ps)  # Ensure non-negative

def calculate_welfare_for_scenario(scenario_data, cost_curves):
    """Calculate detailed welfare for a single scenario."""
    
    # Dynamically extract consumer types from columns
    consumer_types = [col.replace('E_EOM_', '') for col in scenario_data.columns if col.startswith('E_EOM_')]

    welfare_results = {
        'consumer_surplus': {},
        'producer_surplus': {},
        'total_consumer_surplus': 0,
        'total_producer_surplus': 0,
        'total_welfare': 0,
        'timestep_welfare': []
    }
    
    # Calculate welfare for each timestep
    for t in range(len(scenario_data)):
        timestep_welfare = {'t': t, 'price': scenario_data.iloc[t]['Price']}
        market_price = scenario_data.iloc[t]['Price']
        
        # Consumer surplus by type
        cs_timestep = {}
        for consumer_type in consumer_types:
            # Net consumption (negative G_consumer means consumption)
            if f'G_{consumer_type}' in scenario_data.columns:
                net_consumption = -scenario_data.iloc[t][f'G_{consumer_type}']  # Make positive
                # Extract demand curve parameters for this consumer and timestep
                e_eom = scenario_data.iloc[t][f'E_EOM_{consumer_type}']
                lambda_eom_0 = scenario_data.iloc[t][f'lambda_EOM_0_{consumer_type}']
                lambda_eom_cap = scenario_data.iloc[t][f'lambda_EOM_cap_{consumer_type}']
                d_eom_cap = scenario_data.iloc[t][f'D_EOM_cap_{consumer_type}']
                
                if net_consumption > 0:
                    cs = net_consumption * (lambda_eom_cap - market_price) + e_eom * (net_consumption ** 2) / 2
                    cs_timestep[consumer_type] = max(0, cs)
                else:
                    cs_timestep[consumer_type] = 0
            else:
                cs_timestep[consumer_type] = 0
        
        # Producer surplus by type
        ps_timestep = {}
        for gen_type in cost_curves.keys():
            if f'G_{gen_type}' in scenario_data.columns:
                generation = scenario_data.iloc[t][f'G_{gen_type}']
                cost_params = cost_curves[gen_type]
                
                if generation > 0:
                    ps = calculate_producer_surplus(gen_type, generation, market_price, cost_params)
                    ps_timestep[gen_type] = ps
                else:
                    ps_timestep[gen_type] = 0
            else:
                ps_timestep[gen_type] = 0
        
        timestep_welfare.update({
            'consumer_surplus': cs_timestep,
            'producer_surplus': ps_timestep,
            'total_cs_timestep': sum(cs_timestep.values()),
            'total_ps_timestep': sum(ps_timestep.values())
        })
        
        welfare_results['timestep_welfare'].append(timestep_welfare)
    
    # Aggregate over all timesteps
    for consumer_type in consumer_types:
        welfare_results['consumer_surplus'][consumer_type] = sum(
            tw['consumer_surplus'].get(consumer_type, 0) for tw in welfare_results['timestep_welfare']
        )
    
    for gen_type in cost_curves.keys():
        welfare_results['producer_surplus'][gen_type] = sum(
            tw['producer_surplus'].get(gen_type, 0) for tw in welfare_results['timestep_welfare']
        )
    
    welfare_results['total_consumer_surplus'] = sum(welfare_results['consumer_surplus'].values())
    welfare_results['total_producer_surplus'] = sum(welfare_results['producer_surplus'].values())
    welfare_results['total_welfare'] = welfare_results['total_consumer_surplus'] + welfare_results['total_producer_surplus']
    
    # Add missing fields that the plotting function expects
    total_energy = scenario_data['Total_Demand'].sum() if 'Total_Demand' in scenario_data.columns else 0
    welfare_results['welfare_per_mwh'] = welfare_results['total_welfare'] / total_energy if total_energy > 0 else 0
    welfare_results['welfare_efficiency'] = welfare_results['total_welfare'] / (welfare_results['total_consumer_surplus'] + welfare_results['total_producer_surplus']) if (welfare_results['total_consumer_surplus'] + welfare_results['total_producer_surplus']) > 0 else 0
    
    return welfare_results


def calculate_welfare_for_all_scenarios(scenarios_data, config_file="../Input/config.yaml"):
    """Calculate welfare for all scenarios."""
    
    # Only need generator cost parameters
    cost_curves = load_generator_cost_parameters(config_file)
    
    print(f"Loaded {len(cost_curves)} generator types")
    
    all_welfare = {}
    
    for scenario_num, scenario_data in scenarios_data.items():
        welfare = calculate_welfare_for_scenario(scenario_data, cost_curves)
        all_welfare[scenario_num] = welfare
    
    return all_welfare

# Update the existing welfare plotting function in visualize.py
def plot_economic_welfare_distribution(welfare_results, save_path=None):
    """Plot economic welfare distribution across scenarios."""
    scenarios = sorted(welfare_results.keys())
    scenario_labels = [scenario_names().get(s, f"Scenario {s}") for s in scenarios]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Total welfare by scenario
    total_welfare = [welfare_results[s]['total_welfare'] for s in scenarios]
    bars1 = ax1.bar(scenario_labels, total_welfare, color='steelblue', alpha=0.7)
    
    # Add value labels
    for bar, val in zip(bars1, total_welfare):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(total_welfare)*0.01,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax1.set_title('Total Economic Welfare by Scenario', fontsize=14, pad=15)
    ax1.set_ylabel('Welfare (M€)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    
    # 2. Welfare components breakdown
    consumer_surplus = [welfare_results[s]['total_consumer_surplus'] for s in scenarios]
    producer_surplus = [welfare_results[s]['total_producer_surplus'] for s in scenarios]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    bars2a = ax2.bar(x - width/2, consumer_surplus, width, label='Consumer Surplus', 
                      color='lightgreen', alpha=0.8)
    bars2b = ax2.bar(x + width/2, producer_surplus, width, label='Producer Surplus', 
                      color='gold', alpha=0.8)
    
    ax2.set_title('Welfare Components by Scenario', fontsize=14, pad=15)
    ax2.set_ylabel('Surplus (M€)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenario_labels, rotation=45, fontsize=10)
    ax2.legend(fontsize=10)
    
    # Add value labels
    for i, (cs, ps) in enumerate(zip(consumer_surplus, producer_surplus)):
        ax2.text(i - width/2, cs + max(consumer_surplus)*0.01, f'{cs:.1f}', 
                ha='center', va='bottom', fontsize=9)
        ax2.text(i + width/2, ps + max(producer_surplus)*0.01, f'{ps:.1f}', 
                ha='center', va='bottom', fontsize=9)
    
    # 3. Welfare per MWh
    welfare_per_mwh = [welfare_results[s]['welfare_per_mwh'] for s in scenarios]
    bars3 = ax3.bar(scenario_labels, welfare_per_mwh, color='coral', alpha=0.7)
    
    # Add value labels
    for bar, val in zip(bars3, welfare_per_mwh):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(welfare_per_mwh)*0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax3.set_title('Welfare per MWh by Scenario', fontsize=14, pad=15)
    ax3.set_ylabel('Welfare (M€/MWh)', fontsize=12)
    ax3.tick_params(axis='x', rotation=45, labelsize=10)
    
    # 4. Welfare efficiency (total welfare / total cost)
    welfare_efficiency = [welfare_results[s]['welfare_efficiency'] for s in scenarios]
    bars4 = ax4.bar(scenario_labels, welfare_efficiency, color='purple', alpha=0.7)
    
    # Add value labels
    for bar, val in zip(bars4, welfare_efficiency):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(welfare_efficiency)*0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax4.set_title('Welfare Efficiency by Scenario', fontsize=14, pad=15)
    ax4.set_ylabel('Efficiency Ratio', fontsize=12)
    ax4.tick_params(axis='x', rotation=45, labelsize=10)
    
    plt.tight_layout(pad=3.0)
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.2)
        png_path = str(save_path).replace('.svg', '.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)
        print(f"Economic welfare distribution saved to: {save_path}")
    
    plt.close(fig)
    return fig 