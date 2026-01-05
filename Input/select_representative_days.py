"""
Script to select 12 representative days from Greek wind/solar data for CVAR modeling.
Creates clean weather data output with only wind and solar capacity factors.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_wind_solar_data(file_path):
    """Load and process wind/solar capacity factor data."""
    df = pd.read_csv(file_path)
    
    print("=== DATA QUALITY CHECK ===")
    
    # Check for Excel error values in raw data
    error_values = ['#VALUE!', '#N/A', '#DIV/0!', '#REF!', '#NUM!', '#NAME?', '#NULL!']
    total_errors = 0
    
    for col_idx in [1, 2]:  # Wind and solar columns
        col_data = df.iloc[:, col_idx].astype(str)
        error_mask = col_data.isin(error_values)
        errors_in_col = error_mask.sum()
        if errors_in_col > 0:
            error_rows = df.index[error_mask].tolist()[:5]  # Show first 5
            print(f"⚠️  WARNING: Found {errors_in_col} Excel error values in column {col_idx}")
            print(f"   Error rows (first 5): {error_rows}")
            if len(error_rows) > 5:
                print(f"   ... and {errors_in_col - 5} more")
            total_errors += errors_in_col
    
    if total_errors > 0:
        print(f"⚠️  TOTAL: {total_errors} error values found - these will be skipped")
        print("   Consider cleaning your Excel file to remove formula errors")
    
    # Parse datetime from first column
    try:
        df['datetime'] = pd.to_datetime(df.iloc[:, 0], dayfirst=True)
        print("✓ Successfully parsed dates with DD/MM/YYYY format")
    except Exception as e:
        print(f"⚠️  WARNING: Date parsing issue with DD/MM/YYYY: {e}")
        print("   Trying MM/DD/YYYY format...")
        try:
            df['datetime'] = pd.to_datetime(df.iloc[:, 0])
            print("✓ Successfully parsed dates with MM/DD/YYYY format")
        except Exception as e2:
            print(f"❌ ERROR: Could not parse dates: {e2}")
            raise
    
    # FIXED: Safe conversion function for capacity factors
    def safe_convert_to_float(value):
        try:
            if isinstance(value, str):
                if value in error_values:
                    return np.nan
                # Check if it's a percentage string
                if value.endswith('%'):
                    return float(value.rstrip('%')) / 100
                else:
                    return float(value)
            else:
                # Already a number, just convert to float
                return float(value)
        except (ValueError, TypeError):
            return np.nan
    
    # Extract capacity factors with error handling
    df['wind_cap'] = df.iloc[:, 1].apply(safe_convert_to_float)
    df['sol_cap'] = df.iloc[:, 2].apply(safe_convert_to_float)
    
    # Check for missing values after conversion and show specific locations
    wind_missing_mask = df['wind_cap'].isna()
    sol_missing_mask = df['sol_cap'].isna()
    
    if wind_missing_mask.any():
        missing_count = wind_missing_mask.sum()
        missing_rows = df.index[wind_missing_mask].tolist()
        missing_dates = df.loc[wind_missing_mask, 'datetime'].dt.strftime('%Y-%m-%d %H:%M').tolist()
        
        print(f"⚠️  WARNING: {missing_count} missing wind capacity factor values")
        print(f"   Missing at rows: {missing_rows[:10]}" + (f" (showing first 10 of {missing_count})" if missing_count > 10 else ""))
        print(f"   Missing at dates: {missing_dates[:5]}" + (f" (showing first 5)" if missing_count > 5 else ""))
    
    if sol_missing_mask.any():
        missing_count = sol_missing_mask.sum()
        missing_rows = df.index[sol_missing_mask].tolist()
        missing_dates = df.loc[sol_missing_mask, 'datetime'].dt.strftime('%Y-%m-%d %H:%M').tolist()
        
        print(f"⚠️  WARNING: {missing_count} missing solar capacity factor values")
        print(f"   Missing at rows: {missing_rows[:10]}" + (f" (showing first 10 of {missing_count})" if missing_count > 10 else ""))
        print(f"   Missing at dates: {missing_dates[:5]}" + (f" (showing first 5)" if missing_count > 5 else ""))
    
    # Drop rows with missing capacity factors
    initial_length = len(df)
    problematic_rows = df[wind_missing_mask | sol_missing_mask]
    if len(problematic_rows) > 0:
        print(f"⚠️  Dropping {len(problematic_rows)} rows with missing data:")
        for _, row in problematic_rows.head(5).iterrows():
            # Handle NaT datetime values safely
            if pd.isna(row['datetime']):
                print(f"   Row {row.name}: Invalid datetime")
            else:
                print(f"   Row {row.name}: {row['datetime'].strftime('%Y-%m-%d %H:%M')}")
        if len(problematic_rows) > 5:
            print(f"   ... and {len(problematic_rows) - 5} more")
    
    df = df.dropna(subset=['wind_cap', 'sol_cap'])
    
    # Add date components
    df['date'] = df['datetime'].dt.date
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour
    
    # Check for reasonable capacity factor ranges with specific problem locations
    wind_out_of_range_mask = (df['wind_cap'] < 0) | (df['wind_cap'] > 1)
    sol_out_of_range_mask = (df['sol_cap'] < 0) | (df['sol_cap'] > 1)
    
    if wind_out_of_range_mask.any():
        out_of_range_count = wind_out_of_range_mask.sum()
        out_of_range_rows = df.index[wind_out_of_range_mask].tolist()[:5]
        out_of_range_values = df.loc[wind_out_of_range_mask, 'wind_cap'].head(5).tolist()
        out_of_range_dates = df.loc[wind_out_of_range_mask, 'datetime'].dt.strftime('%Y-%m-%d %H:%M').head(5).tolist()
        
        print(f"⚠️  WARNING: {out_of_range_count} wind capacity factors outside [0,1] range")
        print(f"   Problem rows: {out_of_range_rows}")
        print(f"   Problem values: {out_of_range_values}")
        print(f"   Problem dates: {out_of_range_dates}")
    
    if sol_out_of_range_mask.any():
        out_of_range_count = sol_out_of_range_mask.sum()
        out_of_range_rows = df.index[sol_out_of_range_mask].tolist()[:5]
        out_of_range_values = df.loc[sol_out_of_range_mask, 'sol_cap'].head(5).tolist()
        out_of_range_dates = df.loc[sol_out_of_range_mask, 'datetime'].dt.strftime('%Y-%m-%d %H:%M').head(5).tolist()
        
        print(f"⚠️  WARNING: {out_of_range_count} solar capacity factors outside [0,1] range")
        print(f"   Problem rows: {out_of_range_rows}")
        print(f"   Problem values: {out_of_range_values}")
        print(f"   Problem dates: {out_of_range_dates}")
    
    print(f"✓ Data quality check complete")
    print(f"✓ Loaded {len(df)} valid hourly observations")
    print(f"✓ Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    return df

def calculate_daily_stats(df):
    """Calculate daily statistics for representative day selection."""
    daily_stats = []
    
    for date, day_data in df.groupby('date'):
        if len(day_data) == 24:  # Only complete days
            stats = {
                'date': date,
                'month': day_data['month'].iloc[0],
                'wind_mean': day_data['wind_cap'].mean(),
                'wind_std': day_data['wind_cap'].std(),
                'sol_mean': day_data['sol_cap'].mean(),
                'sol_std': day_data['sol_cap'].std(),
                'correlation': day_data['wind_cap'].corr(day_data['sol_cap']),
                'variability': day_data['wind_cap'].std() + day_data['sol_cap'].std()
            }
            daily_stats.append(stats)
    
    return pd.DataFrame(daily_stats)

def select_representative_days(daily_stats, n_days=12):
    """Select representative days including at least one dunkelflaute day."""
    print("\n=== Representative Day Selection Process ===")
    
    # Step 1: Identify dunkelflaute days (low wind + solar)
    daily_stats = daily_stats.copy()
    daily_stats['dunkelflaute_score'] = daily_stats['wind_mean'] + daily_stats['sol_mean']
    
    # Find the worst dunkelflaute days across the entire year
    worst_dunkelflaute = daily_stats.nsmallest(10, 'dunkelflaute_score')
    print(f"Worst dunkelflaute days identified:")
    for _, day in worst_dunkelflaute.head(5).iterrows():
        print(f"  {day['date']} (Month {day['month']}): Wind={day['wind_mean']:.3f}, Solar={day['sol_mean']:.3f}, Total={day['dunkelflaute_score']:.3f}")
    
    # Step 2: Ensure at least one dunkelflaute day is included
    selected_days = []
    forced_dunkelflaute_day = worst_dunkelflaute.iloc[0]
    selected_days.append(forced_dunkelflaute_day)
    dunkelflaute_month = forced_dunkelflaute_day['month']
    
    print(f"\n✓ FORCED DUNKELFLAUTE: Selected {forced_dunkelflaute_day['date']} from month {dunkelflaute_month}")
    print(f"  Wind: {forced_dunkelflaute_day['wind_mean']:.3f}, Solar: {forced_dunkelflaute_day['sol_mean']:.3f}")
    
    # Step 3: Select representative days for remaining months using existing logic
    remaining_days_needed = n_days - 1
    months_to_fill = [m for m in range(1, 13) if m != dunkelflaute_month]
    
    print(f"\nSelecting {remaining_days_needed} additional representative days from {len(months_to_fill)} remaining months...")
    
    # For months with available data, select representative days
    monthly_selections = []
    for month in months_to_fill:
        month_data = daily_stats[daily_stats['month'] == month]
        
        if len(month_data) == 0:
            print(f"⚠️  No data for month {month}")
            continue
        
        # Select day closest to monthly mean characteristics (original logic)
        month_wind_mean = month_data['wind_mean'].mean()
        month_sol_mean = month_data['sol_mean'].mean()
        
        month_data = month_data.copy()
        month_data['distance'] = np.sqrt(
            (month_data['wind_mean'] - month_wind_mean)**2 + 
            (month_data['sol_mean'] - month_sol_mean)**2
        )
        
        best_day = month_data.loc[month_data['distance'].idxmin()]
        monthly_selections.append(best_day)
        
        print(f"Month {month}: Selected {best_day['date']} "
              f"(Wind: {best_day['wind_mean']:.3f}, Solar: {best_day['sol_mean']:.3f})")
    
    # Step 4: If we have more monthly selections than needed, prioritize diversity
    if len(monthly_selections) > remaining_days_needed:
        print(f"\n⚠️  Have {len(monthly_selections)} monthly candidates but only need {remaining_days_needed}")
        print("Selecting most diverse subset...")
        
        # Sort by dunkelflaute score to include more challenging days
        monthly_selections_df = pd.DataFrame(monthly_selections)
        monthly_selections_df = monthly_selections_df.sort_values('dunkelflaute_score').head(remaining_days_needed)
        selected_days.extend(monthly_selections_df.to_dict('records'))
        
        print("Selected most challenging days from remaining months")
    else:
        # Add all available monthly selections
        selected_days.extend(monthly_selections)
        
        # If we still need more days, add additional dunkelflaute days
        if len(selected_days) < n_days:
            remaining_needed = n_days - len(selected_days)
            print(f"\nNeed {remaining_needed} additional days - selecting more dunkelflaute days")
            
            # Get dates already selected
            selected_dates = {day['date'] for day in selected_days}
            
            # Find additional dunkelflaute days not already selected
            additional_dunkelflaute = worst_dunkelflaute[
                ~worst_dunkelflaute['date'].isin(selected_dates)
            ].head(remaining_needed)
            
            for _, day in additional_dunkelflaute.iterrows():
                selected_days.append(day)
                print(f"Additional dunkelflaute: {day['date']} (Month {day['month']}): "
                      f"Wind={day['wind_mean']:.3f}, Solar={day['sol_mean']:.3f}")
    
    final_df = pd.DataFrame(selected_days)
    
    # Summary statistics
    print(f"\n=== FINAL SELECTION SUMMARY ===")
    print(f"Total days selected: {len(final_df)}")
    
    dunkelflaute_count = (final_df['dunkelflaute_score'] <= 0.3).sum()
    print(f"Dunkelflaute days (wind+solar ≤ 0.3): {dunkelflaute_count}")
    
    print(f"Wind mean range: {final_df['wind_mean'].min():.3f} - {final_df['wind_mean'].max():.3f}")
    print(f"Solar mean range: {final_df['sol_mean'].min():.3f} - {final_df['sol_mean'].max():.3f}")
    print(f"Combined renewable range: {final_df['dunkelflaute_score'].min():.3f} - {final_df['dunkelflaute_score'].max():.3f}")
    
    # Show months covered
    months_covered = sorted(final_df['month'].unique())
    print(f"Months represented: {months_covered}")
    
    return final_df

def create_offshore_wind_profile(onshore_values):
    """Create offshore wind with higher capacity factor and smoother profile."""
    # Convert to series for processing
    onshore_series = pd.Series(onshore_values)
    
    # Apply smoothing (3-hour rolling mean)
    smoothed = onshore_series.rolling(window=3, center=True, min_periods=1).mean()
    
    # Increase capacity factor by 30% but cap at 100%
    offshore = np.minimum(smoothed * 1.3, 1.0)
    
    # FIXED: Use 4 decimal places for smoother curves
    offshore_rounded = np.round(offshore, 4)
    
    return offshore_rounded.values

def create_weather_timeseries(df, selected_days, output_file):
    """Create weather timeseries from selected days (nTimesteps from config)."""
    
    # First, try to load existing timeseries file to preserve load data
    try:
        existing_df = pd.read_csv(output_file)
        print(f"✓ Found existing file {output_file} with {len(existing_df)} rows and columns: {existing_df.columns.tolist()}")
        preserve_existing = True
    except FileNotFoundError:
        print(f"✓ Creating new file {output_file}")
        preserve_existing = False
        existing_df = None
    
    # Create weather data for selected days
    timeseries_data = []
    
    for _, day_info in selected_days.iterrows():
        # Get hourly data for this day
        day_data = df[df['date'] == day_info['date']].sort_values('hour')
        
        if len(day_data) != 24:
            print(f"Warning: Day {day_info['date']} has {len(day_data)} hours, skipping")
            continue
        
        for _, hour_row in day_data.iterrows():
            timeseries_data.append({
                'WIND_ONSHORE': round(hour_row['wind_cap'], 4),  # FIXED: 4 decimal places
                'WIND_OFFSHORE': round(hour_row['wind_cap'], 4),  # Will be processed later
                'PV': round(hour_row['sol_cap'], 4)              # FIXED: 4 decimal places
            })
    
    # Create DataFrame for weather data
    weather_df = pd.DataFrame(timeseries_data)
    
    # Process offshore wind for entire timeseries (now with higher precision)
    weather_df['WIND_OFFSHORE'] = create_offshore_wind_profile(weather_df['WIND_ONSHORE'])
    
    # Always create new weather file (don't try to merge)
    final_df = weather_df
    
    # Save to file
    final_df.to_csv(output_file, index=False)
    
    print(f"\nCreated {len(final_df)}-hour weather timeseries")
    print(f"Saved to: {output_file}")
    print(f"Final columns: {final_df.columns.tolist()}")
    
    # Print summary statistics for weather columns only
    print("\n=== Weather Data Summary ===")
    for col in ['WIND_ONSHORE', 'WIND_OFFSHORE', 'PV']:
        if col in final_df.columns:
            mean_val = final_df[col].mean()
            std_val = final_df[col].std()
            print(f"{col} - Mean: {mean_val:.3f}, Std: {std_val:.3f}")
    
    return final_df

def main():
    """Main execution function."""
    # File paths
    wind_solar_file = "GR_wind_sol_AF_hourly_2024.csv"
    output_file = "ts_RES_AF_12d.csv"  # CHANGED: Separate weather file
    
    print("=== Representative Weather Day Selection for Capacity Market Research ===")
    print("Including dunkelflaute days (low wind + solar) for dispatchable generation analysis")
    
    # Load and process data
    df = load_wind_solar_data(wind_solar_file)
    daily_stats = calculate_daily_stats(df)
    selected_days = select_representative_days(daily_stats)
    weather_timeseries = create_weather_timeseries(df, selected_days, output_file)
    
    print(f"\n✓ Representative weather data created successfully!")
    print(f"   Output file: {output_file}")
    print(f"   Total timesteps: {len(weather_timeseries)}")
    print(f"   Includes dunkelflaute days for capacity market stress testing")
    print(f" Weather data saved to separate file: {output_file}")
    print(f" Demand data should remain in: ts_demand_12d.csv")
    
    # Additional analysis for capacity market research
    print(f"\n=== CAPACITY MARKET RESEARCH INSIGHTS ===")
    
    # Calculate hours with very low renewable generation
    low_renewable_threshold = 0.2
    combined_renewable = weather_timeseries['WIND_ONSHORE'] + weather_timeseries['PV']  # CHANGED: Use PV
    low_renewable_hours = (combined_renewable <= low_renewable_threshold).sum()
    
    print(f"Hours with combined renewables ≤ {low_renewable_threshold}: {low_renewable_hours}/{len(weather_timeseries)} ({low_renewable_hours/len(weather_timeseries)*100:.1f}%)")
    
    # Calculate minimum renewable generation period
    min_combined = combined_renewable.min()
    min_hour_idx = combined_renewable.idxmin()
    print(f"Minimum renewable generation: {min_combined:.3f} at hour {min_hour_idx + 1}")
    
    # Calculate statistics for capacity adequacy
    print(f"\nRenewable generation statistics (for capacity adequacy assessment):")
    print(f"  Combined (Wind + Solar) - Mean: {combined_renewable.mean():.3f}, Min: {combined_renewable.min():.3f}, Max: {combined_renewable.max():.3f}")
    print(f"  Wind Onshore - Mean: {weather_timeseries['WIND_ONSHORE'].mean():.3f}, Min: {weather_timeseries['WIND_ONSHORE'].min():.3f}")
    print(f"  Wind Offshore - Mean: {weather_timeseries['WIND_OFFSHORE'].mean():.3f}, Min: {weather_timeseries['WIND_OFFSHORE'].min():.3f}")
    print(f"  Solar - Mean: {weather_timeseries['PV'].mean():.3f}, Min: {weather_timeseries['PV'].min():.3f}")  # CHANGED: Use PV

if __name__ == "__main__":
    main() 