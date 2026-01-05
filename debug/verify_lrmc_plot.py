import sys
import os
from pathlib import Path

# Add Analysis directory to path
sys.path.append(os.path.abspath("Analysis"))

import plot_input_data

# Create output directory if it doesn't exist
Path("Analysis/output/plots").mkdir(parents=True, exist_ok=True)

try:
    print("Generating LRMC plot...")
    plot_input_data.plot_generator_cost_curves(save_path="Analysis/output/plots/test_lrmc.png")
    print("Plot generated successfully at Analysis/output/plots/test_lrmc.png")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
