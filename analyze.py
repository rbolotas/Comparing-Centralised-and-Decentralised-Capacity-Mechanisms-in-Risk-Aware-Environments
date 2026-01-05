#Main analysis script ->  GreekCRM\Analysis> python analyze.py --scenarios 3 5 7 --variant beta_1.0 --plots   
import argparse
import os
from pathlib import Path
import metrics
import visualize
import compare
import plot_input_data
import matplotlib.pyplot as plt
import welfare_analysis
import enhanced_ens_analysis
import numpy as np # Added for numpy

# Get the project root directory (2 levels up from the Analysis directory)
PROJECT_ROOT = Path(__file__).parent.parent

def main():
    parser = argparse.ArgumentParser(description='Analyze Greek CRM simulation results')
    parser.add_argument('--scenarios', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7],
                        help='Scenario numbers to analyze')
    parser.add_argument('--variant', type=str, default='ref',
                        help='Scenario variant (e.g., "ref" for reference case)')
    parser.add_argument('--output-dir', type=str, default='../Analysis/output',
                        help='Directory to save output files')
    parser.add_argument('--excel', action='store_true',
                        help='Export results to Excel file')
    parser.add_argument('--json', action='store_true',
                        help='Export results to JSON file')
    parser.add_argument('--plots', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--beta-sweep', action='store_true',
                        help='Generate beta-sweep plots for scenarios 3,5,7 (β 1.0→0.1)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing scenarios: {args.scenarios}")
    
    # Calculate metrics for all scenarios
    results = {}
    for scen in args.scenarios:
        try:
            print(f"Processing Scenario {scen}...")
            results[scen] = metrics.calculate_all_metrics(scen, args.variant)
        except Exception as e:
            print(f"Error processing Scenario {scen}: {e}")
    
    # Export results if requested
    if args.excel:
        excel_path = output_dir / f"analysis_results_{args.variant}.xlsx"
        compare.export_results_to_excel(results, excel_path)
        print(f"Results exported to Excel: {excel_path}")
    
    if args.json:
        json_path = output_dir / f"analysis_results_{args.variant}.json"
        compare.export_results_to_json(results, json_path)
        print(f"Results exported to JSON: {json_path}")
    
    # Separate energy-only from capacity market scenarios
    energy_only_scenarios = [s for s in results.keys() if s < 4]
    capacity_market_scenarios = [s for s in results.keys() if s >= 3]  # Include scenario 3 as benchmark

    print(f"Energy-only market scenarios: {energy_only_scenarios}")
    print(f"Capacity market scenarios (including benchmark): {capacity_market_scenarios}")

    # Generate plots if requested
    if args.plots:
        print("Generating plots...")
        
        # Create plot directory structure
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        input_plots_dir = plots_dir / "input"
        input_plots_dir.mkdir(exist_ok=True)
        
        scenario_plots_dir = plots_dir / "scenario_analysis"
        scenario_plots_dir.mkdir(exist_ok=True)
        
        # Delete old plots before creating new ones
        print("  - Cleaning old plots...")
        old_plots = list(plots_dir.glob("*.png")) + list(plots_dir.glob("*.svg"))
        for old_plot in old_plots:
            try:
                old_plot.unlink()
                #print(f"    Deleted: {old_plot.name}")
            except Exception as e:
                print(f"    Could not delete {old_plot.name}: {e}")
        
        scenarios_data = visualize.load_all_scenarios(args.scenarios, args.variant)
        
        # NEW: Calculate economic welfare analysis
        print("  - Economic welfare analysis...")
        welfare_results = welfare_analysis.calculate_welfare_for_all_scenarios(scenarios_data)
        
        # Generate welfare distribution plot
        print("  - Economic welfare distribution...")
        welfare_analysis.plot_economic_welfare_distribution(welfare_results,
                                                          save_path=scenario_plots_dir / "economic_welfare_distribution.svg")
        
        # Generate individual plots
        print("  - Price duration curves...")
        visualize.plot_price_duration_curves(scenarios_data, 
                                           save_path=scenario_plots_dir / "price_duration_curves.svg")
        
        # NEW: Combined load duration curves
        print("  - Combined load duration curves...")
        visualize.plot_combined_load_duration_curves(scenarios_data,
                                                   save_path=scenario_plots_dir / "combined_load_duration_curves.svg")
        
        # NEW: Individual load duration curve plots
        print("  - Total load duration curves...")
        visualize.plot_total_load_duration_curves(scenarios_data,
                                               save_path=str(scenario_plots_dir))
        
        print("  - Residual load duration curves...")
        visualize.plot_residual_load_duration_curves(scenarios_data,
                                                   save_path=str(scenario_plots_dir))
        
        print("  - Individual residual load duration curves...")
        visualize.plot_individual_residual_load_duration_curves(scenarios_data, save_path=str(scenario_plots_dir))
        
        # NEW: Average hourly profiles
        print("  - Average hourly profiles...")
        visualize.plot_average_hourly_profiles(scenarios_data,
                                             save_path=scenario_plots_dir / "average_hourly_profiles.svg")

        # NEW: Consumer timeseries summary plots
        print("  - Consumer timeseries summary...")
        visualize.plot_consumer_timeseries_summary(scenarios_data, config_file="../Input/config.yaml", save_dir=str(scenario_plots_dir))
        
        # NEW: Price timeseries summary plots
        print("  - Price timeseries summary...")
        visualize.plot_price_timeseries_summary(scenarios_data, save_dir=str(scenario_plots_dir))
        
        # NEW: Generation by generator summary plots
        print("  - Generation by generator summary...")
        visualize.plot_generator_timeseries_per_scenario(scenarios_data, config_file="../Input/config.yaml", save_dir=str(scenario_plots_dir))
        visualize.plot_generator_timeseries_per_generator(scenarios_data, config_file="../Input/config.yaml", save_dir=str(scenario_plots_dir))
        
        # Stacked dispatchable generation timeseries
        visualize.plot_dispatchable_generation_timeseries(scenarios_data, save_path=scenario_plots_dir / "dispatchable_generation_timeseries.svg")
        
        # NEW: Dispatchable generation vs. price analysis
        print("  - Dispatchable generation vs. price analysis...")
        visualize.plot_comprehensive_generation_vs_price(scenarios_data, 
                                                      save_path=scenario_plots_dir / "comprehensive_generation_vs_price.svg")
        
        print("  - Dispatchable generation vs. price by scenario...")
        visualize.plot_generation_vs_price_by_scenario_dispatchable(scenarios_data, 
                                                                  save_path=scenario_plots_dir / "dispatchable_generation_vs_price_by_scenario.svg")
        
        print("  - Dispatchable generation summary...")
        visualize.plot_dispatchable_generation_summary(scenarios_data, 
                                                    save_path=scenario_plots_dir / "dispatchable_generation_summary.svg")
        

        
        # Generate capacity comparison (including scenario 3 as benchmark)
        if capacity_market_scenarios:
            print("  - Capacity comparison...")
            visualize.plot_capacity_comparison(
                {s: results[s] for s in capacity_market_scenarios}, 
                save_path=scenario_plots_dir / "capacity_comparison.svg")
        
        # Generate dispatchable capacity analysis for all scenarios
        print("  - Dispatchable capacity analysis...")
        visualize.plot_dispatchable_capacity_comparison(
            results, save_path=scenario_plots_dir / "dispatchable_capacity_comparison.svg")
        
        # Enhanced ENS analysis and dashboard
        print("  - Enhanced ENS analysis...")
        ens_results = enhanced_ens_analysis.analyze_ens_for_all_scenarios(args.scenarios, args.variant)
        visualize.plot_enhanced_ens_visualization(
            ens_results, save_path=scenario_plots_dir / "reliability_ens_dashboard.svg")
        
        print("  - Consumer costs...")
        visualize.plot_consumer_costs(results,
                                     save_path=scenario_plots_dir / "consumer_costs.svg")
        
        print("  - Enhanced consumer costs...")
        visualize.plot_enhanced_consumer_costs(results,
                                             save_path=scenario_plots_dir / "enhanced_consumer_costs.svg")
        
        print("  - Price boxplot...")
        visualize.plot_price_boxplot(results, save_path=scenario_plots_dir / "price_boxplot.svg")
        
        print("  - Generation mix...")
        visualize.plot_generation_mix(results,
                                     save_path=scenario_plots_dir / "generation_mix.svg")
        
        # Generate CM-specific plots if applicable
        cm_scenarios = [s for s in args.scenarios if s in [4, 5, 6, 7]]
        for scen in cm_scenarios:
            print(f"  - Generator revenue for scenario {scen}...")
            visualize.plot_generator_revenue(results, scen, 
                                           save_path=scenario_plots_dir / f"generator_revenue_s{scen}.svg")
        
        # Generate summary dashboard
        print("  - Summary dashboard...")
        visualize.create_summary_dashboard(results,
                                         save_path=scenario_plots_dir / "summary_dashboard.svg")
        
        # Generate generator revenue summary
        print("  - Generator revenue summary...")
        visualize.create_generator_revenue_summary(results,
                                                 save_path=scenario_plots_dir / "generator_revenue_summary.svg")
        
        # REMOVED: welfare_distribution functionality - no longer needed
        
        # Generate INPUT DATA plots (weather, load, generator costs)
        try:
            print("  - INPUT DATA PLOTS:")
            print("    - Weather profiles...")
            weather_df = plot_input_data.load_weather_data("../Input/timeseries_12d.csv")
            
            # Generate weather profiles
            plot_input_data.plot_weather_profiles(
                weather_df, save_path=input_plots_dir / "representative_days_weather_profiles.svg")
            
            # Generate load profiles  
            print("    - Load profiles...")
            plot_input_data.plot_reference_demand_profiles(
                weather_df, save_path=input_plots_dir / "representative_days_load_profiles.svg")
            
            # Generate generator cost curves
            print("    - Generator cost curves and merit order...")
            plot_input_data.plot_generator_cost_curves(
                save_path=input_plots_dir / "generator_cost_curves_merit_order.svg")
                
        except Exception as e:
            print(f"  Warning: Could not generate input data plots: {e}")
        
        # Individual residual load duration curves (already handled above with specific file paths)
        print("  - Individual residual load duration curves...")
        visualize.plot_individual_residual_load_duration_curves(scenarios_data, save_path=str(scenario_plots_dir))
        
        # Always attempt beta-sweep plots so the old command includes them
        try:
            print("  - Beta sweep plots (scenarios 3,5,7; beta 1.0->0.1)...")
            visualize.run_beta_sweep_plots(save_dir=str(plots_dir))
        except Exception as e:
            print(f"  Warning: Beta sweep plots failed: {e}")
        
        print(f"\nAll plots saved to:")
        print(f"   - Input plots: {input_plots_dir}")
        print(f"   - Scenario analysis: {scenario_plots_dir}")
        print(f"   - Sensitivity analysis: {plots_dir / 'sensitivity_analysis'}")
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
