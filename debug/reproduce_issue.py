import sys
import os
from pathlib import Path
import traceback

# Add the current directory to sys.path so we can import modules
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'Analysis'))

try:
    import Analysis.visualize as visualize
except ImportError:
    # Try importing without Analysis prefix if running from Analysis dir
    sys.path.append(os.getcwd())
    import visualize

def reproduce():
    print("Attempting to reproduce 'Beta sweep plots failed: LV_MED'...")
    
    # We need to mock the output directory
    output_dir = Path("Analysis/output/plots_debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # The error happens in run_beta_sweep_plots
        # It iterates over scenarios [3, 5, 7]
        # We can try running it and see the full traceback
        visualize.run_beta_sweep_plots(scenarios=[3, 5, 7], save_dir=str(output_dir))
        print("No error occurred!")
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    reproduce()
