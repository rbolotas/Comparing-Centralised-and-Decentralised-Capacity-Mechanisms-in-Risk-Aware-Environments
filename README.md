# GreekCRM - Greek Capacity Resource Mechanism Simulation

A Julia-based agent-based model for analyzing capacity resource mechanisms in the Greek electricity market. This project simulates various market designs and compares their effects on system reliability, costs, and agent behavior.

## Overview

GreekCRM is a sophisticated electricity market simulation that models the interaction between generators and consumers under different market mechanisms. The model uses the Alternating Direction Method of Multipliers (ADMM) algorithm to find market equilibria and incorporates risk-aware behavior through Conditional Value at Risk (CVaR) metrics.

### Key Features

- **Multi-market simulation**: Energy-only markets (EOM), centralized capacity markets (cCM), and decentralized capacity markets (dCM/reliability options)
- **Agent-based modeling**: Representative generators and consumers with heterogeneous characteristics
- **Risk-aware agents**: CVaR-based risk metrics for uncertainty handling
- **Elastic demand**: Price-responsive consumer behavior
- **High-performance computing**: Optimized for parallel execution and HPC environments
- **Comprehensive analysis**: Python-based post-processing and visualization tools

## Project Structure

```
GreekCRM/
├── MAIN.jl                    # Main simulation entry point
├── Source/                    # Core Julia simulation code
│   ├── ADMM.jl               # ADMM algorithm implementation
│   ├── build_*_agent.jl      # Agent model builders
│   ├── solve_*_agent.jl      # Agent optimization solvers
│   └── define_*_parameters.jl # Parameter definitions
├── Input/                     # Configuration and input data
│   ├── config.yaml           # Main configuration file
│   ├── demand_timeseries_12d.csv    # Representative days time series
│   ├── weather_timeseries_12d.csv # generative weather availiability ts 
│   └── GR_wind_sol_AF_hourly_2024.csv # Renewable availability factors
├── Results/                   # Simulation output files
├── Analysis/                  # Python analysis and visualization
│   ├── analyze.py            # Main analysis script
│   ├── visualize.py          # Plotting functions
│   ├── metrics.py            # Performance metrics calculation
│   └── output/               # Generated plots and Excel files
├── docs/                      # Research papers and documentation
└── Hpc/                       # High-performance computing setup
```

## Market Scenarios

The model simulates seven different market scenarios:

1. **Scenario 1**: Energy-only market with inelastic demand (EOM Inelastic)
2. **Scenario 2**: Energy-only market with elastic demand (EOM)
3. **Scenario 3**: EOM with risk-aware agents using CVaR (EOM + CVaR)
4. **Scenario 4**: EOM with elastic demand and centralized capacity market (EOM + cCM)
5. **Scenario 5**: EOM with CVaR and centralized capacity market (EOM + cCM + CVaR)
6. **Scenario 6**: EOM with elastic demand and reliability options (EOM + RO)
7. **Scenario 7**: EOM with CVaR and reliability options (EOM + RO + CVaR)

## Agent Types

### Generators
- **Baseload**: Low variable cost, high investment cost (CCGT)
- **MidMerit**: Medium flexibility and costs (OCGT)
- **Peak**: High variable cost, low investment cost (peaking units)
- **WindOnshore**: Zero marginal cost, renewable generation

### Consumers
- **LV_LOW**: Low-voltage residential consumers (low consumption)
- **LV_MED**: Low-voltage residential consumers (medium consumption)
- **LV_HIGH**: Low-voltage residential consumers (high consumption)
- **MV_LOAD**: Medium-voltage industrial consumers

## Requirements

### Software Dependencies
- **Julia** (≥1.6)
  - JuMP.jl
  - Gurobi.jl
  - DataFrames.jl
  - CSV.jl
  - YAML.jl
  - ProgressBars.jl
- **Python** (≥3.8)
  - pandas
  - matplotlib
  - seaborn
  - openpyxl
  - numpy
- **Gurobi Optimizer** (license required)

### Hardware Requirements
- Recommended: 8+ GB RAM
- Multi-core CPU (parallel execution support)
- For HPC: Compatible with DelftBlue and VSC clusters

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd GreekCRM
   ```

2. **Set up Julia environment**:
   ```julia
   using Pkg
   Pkg.instantiate()
   ```

3. **Install Python dependencies**:
   ```bash
   pip install pandas matplotlib seaborn openpyxl numpy argparse
   ```

4. **Configure Gurobi**:
   - Install Gurobi Optimizer
   - Obtain and place license file in `Hpc/gurobi.lic`
   - Update paths in `MAIN.jl` for your system

## Usage

### Running Simulations

**Basic execution (all scenarios)**:
```julia
julia MAIN.jl
```

**HPC execution** (single scenario):
```bash
julia MAIN.jl --start_scen 1 --stop_scen 1
```

**Custom scenario range**:
```julia
# Edit MAIN.jl and modify:
start_scen = 2
stop_scen = 5
```

### Configuration

Edit `Input/config.yaml` to modify:
- **Market parameters**: Price caps, reserve margins, risk metrics
- **Agent characteristics**: Technologies, costs, demand elasticities
- **ADMM settings**: Convergence tolerance, penalty parameters
- **Time horizons**: Number of representative days/hours

### Analysis and Visualization

**Generate comprehensive analysis**:
```bash
cd Analysis
python analyze.py --scenarios 1 2 3 4 5 6 7 --excel --plots
```

**Options**:
- `--excel`: Export results to Excel workbook
- `--plots`: Generate visualization plots
- `--json`: Export to JSON format
- `--scenarios`: Specify which scenarios to analyze

**Outputs**:
- Excel workbook with detailed metrics (`analysis_results_ref.xlsx`)
- Publication-ready plots (PNG/SVG format)
- Summary dashboard with key comparisons

## Key Metrics

The model tracks comprehensive performance indicators:

### Economic Metrics
- **Consumer costs**: Total expenditure across consumer types
- **Generator revenues**: Revenue breakdown by technology and scenario
- **System costs**: Total cost of electricity supply

### Reliability Metrics
- **Capacity adequacy**: Available vs. required capacity
- **Price volatility**: Price duration curves and statistics

### Market Metrics
- **Generation mix**: Technology deployment and utilization
- **Market prices**: Energy and capacity price formation
- **Investment patterns**: Capacity decisions under uncertainty

## Model Features

### ADMM Algorithm
- **Convergence**: Dual and primal residual tracking
- **Parallelization**: Multi-threaded agent solving
- **Robustness**: Adaptive penalty parameter updates

### Risk Modeling
- **CVaR implementation**: Conditional Value at Risk for tail risk
- **Confidence levels**: Configurable risk aversion parameters
- **Portfolio effects**: Risk aggregation across time periods

### Market Mechanisms
- **Energy-only markets**: Marginal cost pricing with price caps
- **Capacity markets**: Centralized procurement with target demand
- **Reliability options**: Decentralized capacity mechanisms

## Output Files

### Simulation Results
- `Results/Scenario_X_ref.csv`: Detailed results for each scenario
- `overview_results.csv`: Convergence and performance summary

### Analysis Outputs
- `Analysis/output/analysis_results_ref.xlsx`: Comprehensive metrics
- `Analysis/output/plots/`: Publication-ready visualizations
- Time series plots, cost comparisons, and reliability metrics

## Research Context

This model supports research on:
- **Capacity mechanism design**: Comparing centralized vs. decentralized approaches
- **Risk management**: Impact of risk-averse behavior on market outcomes


## Contributing

For research collaborations or contributions:
1. Follow the existing code structure and documentation standards
2. Test modifications with multiple scenarios
3. Update configuration files and documentation as needed
4. Ensure compatibility with HPC environments

## License

This project is developed for academic research purposes. Please cite appropriately when using this code in publications.

## Support

For technical issues or research questions:
- Check the `docs/` directory for relevant research papers
- Review configuration examples in `Input/config.yaml`
- Examine analysis scripts in `Analysis/` for post-processing guidance

---

**Authors**: Romannos Bolotas, Kenneth Bruninx  
**Last Updated**: July 2025  
**Institution**: TU Delft, TPM