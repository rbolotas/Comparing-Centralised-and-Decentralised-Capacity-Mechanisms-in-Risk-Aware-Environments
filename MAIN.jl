## Topic: Go-E
# Author: Romannos Bolotas - Kenneth Bruninx
# Last update: April 2024

## 0. Set-up code
# HPC or not?
HPC = "ThinKing" # NA (not applicable) or DelftBlue  

# Home directory
const home_dir = @__DIR__

if HPC == "DelftBlue"  # only for running this on DelftBlue
    ENV["GRB_LICENSE_FILE"] = "./Hpc/gurobi.lic"
    ENV["GUROBI_HOME"] = "./scratch/kbruninx/gurobi950/linux64"
    println(string("Number of threads: ", Threads.nthreads()))
end

if HPC == "ThinKing"  # only for running this on VSC
    ENV["GRB_LICENSE_FILE"] = "C:/Users/user/gurobi.lic"
    ENV["GUROBI_HOME"] = "C:/gurobi1103/win64"
    ENV["PATH"] = string(ENV["PATH"], ";", ENV["GUROBI_HOME"], "\\bin")
end

# Include packages 
using JuMP, Gurobi # Optimization packages
using DataFrames, CSV, YAML, DataStructures # dataprocessing
using ProgressBars, Printf # progress bar
using TimerOutputs # profiling 
using Base.Threads: @spawn 
using Base: split
using ArgParse # Parsing arguments from the command line
# using JLD2 # save workspace

# # Gurobi environment to suppress output
# println("Define Gurobi environment...")
# println("        ")
# const GUROBI_ENV = Gurobi.Env()
# # set parameters:
# GRBsetparam(GUROBI_ENV, "OutputFlag", "0")   
# GRBsetparam(GUROBI_ENV, "Threads", "4")   
# println("        ")

# Gurobi environment to suppress output
println("Define Gurobi environment...")
println("        ")

# Gurobi environment to suppress output
println("Define Gurobi environment...")
println("        ")

if !@isdefined(GUROBI_ENV)
    const global GUROBI_ENV = Gurobi.Env()
    # set parameters:
    GRBsetparam(GUROBI_ENV, "OutputFlag", "0")  
    GRBsetparam(GUROBI_ENV, "NumericFocus", "3") 
    GRBsetparam(GUROBI_ENV, "Threads", "4")
    GRBsetparam(GUROBI_ENV, "ScaleFlag", "2")
    #Controls model scaling. By default, the rows and columns of the model are scaled in order to improve the numerical properties of the constraint matrix. 
    #The scaling is removed before the final solution is returned. 
    #Scaling typically reduces solution times, but it may lead to larger constraint violations in the original, unscaled model. 
    #Turning off scaling (ScaleFlag=0) can sometimes produce smaller constraint violations. 
    #Choosing a different scaling option can sometimes improve performance for particularly numerically difficult models. 
    #Using geometric mean scaling (ScaleFlag=2) is especially well suited for models with a wide range of coefficients in the constraint matrix rows or columns.
    println("GUROBI_ENV defined and parameters set.")
else
    println("GUROBI_ENV already defined. Skipping definition.")
end
println("        ")

# Include functions
include(joinpath(home_dir,"Source","define_common_parameters.jl"))
include(joinpath(home_dir,"Source","define_EOM_parameters.jl"))
include(joinpath(home_dir,"Source","define_cCM_parameters.jl"))
include(joinpath(home_dir,"Source","define_dCM_parameters.jl"))
include(joinpath(home_dir,"Source","define_consumer_parameters.jl"))
include(joinpath(home_dir,"Source","define_generator_parameters.jl"))
include(joinpath(home_dir,"Source","build_consumer_agent.jl"))
include(joinpath(home_dir,"Source","build_generator_agent.jl"))
include(joinpath(home_dir,"Source","define_results.jl"))
include(joinpath(home_dir,"Source","ADMM.jl"))
include(joinpath(home_dir,"Source","ADMM_subroutine.jl"))
include(joinpath(home_dir,"Source","solve_consumer_agent.jl"))
include(joinpath(home_dir,"Source","solve_generator_agent.jl"))
include(joinpath(home_dir,"Source","update_rho.jl"))
include(joinpath(home_dir,"Source","save_results.jl"))
include(joinpath(home_dir,"Source","diagnose_numerical_issues.jl"))
include(joinpath(home_dir,"Source","print_config_summary.jl"))

# Data common to all scenarios data 
data = YAML.load_file(joinpath(home_dir,"Input","config.yaml"))


#ts = CSV.read(joinpath(home_dir,"Input","timeseries_24h.csv"),delim=";",DataFrame)
# Load weather data
weather_df = CSV.read(joinpath(home_dir,"Input", data["Files"]["weather_timeseries"]), DataFrame)

# Load demand data  
demand_df = CSV.read(joinpath(home_dir,"Input", data["Files"]["demand_timeseries"]), DataFrame)

# Create unified ts DataFrame by merging weather and demand data
ts = DataFrame()

# Add all weather columns
for col in names(weather_df)
    ts[!, col] = weather_df[!, col]
end

# Add all demand columns  
for col in names(demand_df)
    ts[!, col] = demand_df[!, col]
end

println("✓ Created unified timeseries with columns: ", names(ts))

# Merge them if needed, or reference them separately


# Overview scenarios
scenario_overview = CSV.read(joinpath(home_dir,"overview_scenarios.csv"),DataFrame,delim=";")
sensitivity_overview = CSV.read(joinpath(home_dir,"overview_sensitivity.csv"),DataFrame,delim=";") 

# Create file with results 
# Delete old results and create fresh file for this run
overview_results_path = joinpath(home_dir, "overview_results.csv")
if isfile(overview_results_path)
    rm(overview_results_path)
end
CSV.write(overview_results_path, DataFrame(), delim=";",
          header=["scen_number";"sensitivity";"n_iter";"walltime";
                  "converged";"PrimalResidual_EOM";"DualResidual_EOM";
                  "PrimalResidual_cCM";"DualResidual_cCM";
                  "PrimalResidual_dCM";"DualResidual_dCM";
                  "λ_cCM";"λ_dCM";
                  "C_system_total";"C_cCM_vol";"C_dCM_vol";
                  "beta_co";"beta_gc"])

# Create folder for results
if isdir(joinpath(home_dir,string("Results"))) != 1
    mkdir(joinpath(home_dir,string("Results")))
end

# ============================================================================
# SCENARIO & BETA CONFIGURATION
# ============================================================================
# Option 1: Run specific (scenario, beta) combinations
# Format: [(scenario, beta), ...] where beta=nothing means "ref" for scenario 1
# Example: [(3, 1.0), (5, 1.0), (7, 1.0)] runs only these three combinations
# Set to empty [] to use Option 2 (full sweep)
SPECIFIC_RUNS = []  # Set to [] for full sweep, or e.g., [(3, 1.0), (5, 1.0), (7, 1.0)]

# Option 2: Full scenario sweep configuration (used when SPECIFIC_RUNS is empty)
SCENARIOS_TO_RUN = [1, 3, 5, 7]  # Which scenarios to run
BETA_RANGE = collect(1.0:-0.1:0.1)  # Beta values for scenarios 2-7
# ============================================================================

# Scenario number 
if HPC == "DelftBlue"  
   function parse_commandline()
       s = ArgParseSettings()
       @add_arg_table! s begin
           "--start_scen"
               help = "Enter the number of the first scenario here"
               arg_type = Int
               default = 1
            "--stop_scen"
               help = "Enter the number of the last scenario here"
               arg_type = Int
               default = 1
       end
       return parse_args(s)
   end
   # Simulation number as argument:
   dict_sim_number =  parse_commandline()
   start_scen = dict_sim_number["start_scen"]
   stop_scen = dict_sim_number["stop_scen"]
else
    # Range of scenarios to be simulated (not used if SPECIFIC_RUNS is set)
    start_scen = 1
    stop_scen = 7
end

# Build run list based on configuration
if !isempty(SPECIFIC_RUNS)
    # Use specific runs
    println("Running specific scenario-beta combinations: ", SPECIFIC_RUNS)
    run_list = SPECIFIC_RUNS
else
    # Build full run list from SCENARIOS_TO_RUN and BETA_RANGE
    run_list = []
    for scen in SCENARIOS_TO_RUN
        if scen == 1
            push!(run_list, (1, nothing))  # Scenario 1 always uses "ref"
        else
            for beta in BETA_RANGE
                push!(run_list, (scen, beta))
            end
        end
    end
    println("Running full sweep: ", length(run_list), " total runs")
end

#scen_number = 1  # for debugging purposes, comment the for-loop and replace it by a explicit definition of the scenario you'd like to study
for (scen_number, beta_override) in run_list

println("    ")
println(string("######################                  Scenario ",scen_number,"                 #########################"))

## 1. Read associated input for this simulation
scenario_overview_row = scenario_overview[scen_number,:]
global data = YAML.load_file(joinpath(home_dir,"Input","config.yaml")) # reload data to avoid previous sensitivity analysis affected data

# Print the data dictionary to ensure it's correctly loaded
println("Loaded data dictionary: ", keys(data))

if scenario_overview_row["Sens_analysis"] == "YES"  
    numb_of_sens = length((sensitivity_overview[!,:Parameter]))
else
    numb_of_sens = 0 
end    
sens_number = 1 # for debugging purposes, comment the for-loop and replace it by a explicit definition of the sensitivity you'd like to study
# for sens_number in range(1,stop=numb_of_sens+1,step=1) 
if sens_number >= 2
    println("    ") 
    println(string("#                                  Sensitivity ",sens_number-1,"                                      #"))
    
    # Get the current sensitivity parameters and scaling
    current_sens = sensitivity_overview[sens_number-1,:]
    parameter_str = current_sens[:Parameter]
    scaling = current_sens[:Scaling]
    
    # Check if this is a multi-parameter sensitivity (indicated by a special prefix or format)
    if startswith(parameter_str, "MULTI:")
        # Handle multiple parameters - example: "MULTI:RiskMetrics β_gc,RiskMetrics β_co"
        param_list = split(replace(parameter_str, "MULTI:" => ""), ",")
        for param_str in param_list
            parameter = split(strip(param_str))
            if length(parameter) == 2
                data[parameter[1]][parameter[2]] = scaling * data[parameter[1]][parameter[2]]
            elseif length(parameter) == 3
                data[parameter[1]][parameter[2]][parameter[3]] = scaling * data[parameter[1]][parameter[2]][parameter[3]]
            end
        end
    else
        # Handle single parameter (existing logic)
        parameter = split(parameter_str)
        if length(parameter) == 2
            data[parameter[1]][parameter[2]] = scaling * data[parameter[1]][parameter[2]]
        elseif length(parameter) == 3
            data[parameter[1]][parameter[2]][parameter[3]] = scaling * data[parameter[1]][parameter[2]][parameter[3]]
        else
            println("warning! Sensitivity analysis is not well defined!")
        end
    end
end

println("    ")
println("Including all required input data: done")
println("   ")

# Use beta from run_list (beta_override)
global data = YAML.load_file(joinpath(home_dir,"Input","config.yaml"))
if beta_override === nothing
    sens_label = "ref"
    println("Using default beta values from config (β_co=", data["RiskMetrics"]["β_co"], ", β_gc=", data["RiskMetrics"]["β_gc"], ")")
else
    # Set risk preferences
    data["RiskMetrics"]["β_co"] = beta_override
    data["RiskMetrics"]["β_gc"] = beta_override
    beta_label = @sprintf("%.1f", beta_override)
    sens_label = string("beta_", beta_label)
    println(string("Running with β_co=β_gc=", beta_label))
end

## 2. Initiate models for representative agents 
agents = Dict()
agents[:Gen] = [id for id in keys(data["Generators"])] 
agents[:Cons] = [id for id in keys(data["Consumers"])]
agents[:all] = union(agents[:Gen],agents[:Cons]) # all agents in the game  
agents[:eom] = union(agents[:Gen],agents[:Cons]) # agents participating in the EOM       
agents[:cCM] = union(agents[:Gen],agents[:Cons]) # agents participating in the cCM     
agents[:dCM] = union(agents[:Gen],agents[:Cons]) # agents participating in the dCM                    
mdict = Dict(i => Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV))) for i in agents[:all])

## 3. Define parameters for markets and representative agents
# Parameters/variables EOM
EOM = Dict()
define_EOM_parameters!(EOM, data, ts, scenario_overview_row)
#Print the EOM dictionary to ensure it's correctly defined
#println("EOM dictionary before passing to define_consumer_parameters!: ", EOM)

# Market 2
cCM = Dict()
define_cCM_parameters!(cCM, data, ts, scenario_overview_row, scen_number)

# Market 3
dCM = Dict()
define_dCM_parameters!(dCM, data, ts, scenario_overview_row)

# consumer models
for m in agents[:Cons]
    define_common_parameters!(m, mdict[m], data, ts, agents, scenario_overview_row)                                  
    define_consumer_parameters!(mdict[m], data, ts, m, scen_number)  
end

# ... after consumer models are defined ...
print_solar_capacity_summary(data)

# Generator models
for m in agents[:Gen]
    define_common_parameters!(m, mdict[m], data, ts, agents, scenario_overview_row)                                  
    define_generator_parameters!(mdict[m], data, ts, m)  
end

# Print generator merit order
print_generator_merit_order(mdict, agents)

# Calculate number of agents in each market
EOM["nAgents"] = length(agents[:eom])
cCM["nAgents"] = length(agents[:cCM])
dCM["nAgents"] = length(agents[:dCM])

println("Inititate model, sets and parameters: done")
println("   ")

## 4. Build models
for m in agents[:Cons]
    build_consumer_agent!(mdict[m], scen_number)
end
for m in agents[:Gen]
    build_generator_agent!(mdict[m], scen_number, m)
end

println("Build model: done")
println("   ")

## 5. ADMM proces to calculate equilibrium
println("Find equilibrium solution...")
println("   ")
println("(Progress indicators on primal residuals, relative to tolerance: <", data["ADMM"]["epsilon"], " indicates convergence)")
println("   ")

results = Dict()
ADMM = Dict()
TO = TimerOutput()
define_results!(data,results,ADMM,agents)
ADMM!(results, ADMM, EOM, cCM, dCM, mdict, agents, scenario_overview_row, data, TO)
ADMM["walltime"] =  TimerOutputs.tottime(TO)*10^-9/60                              # wall time 

println(string("Done!"))
println(string("        "))
println(string("Required iterations: ",ADMM["n_iter"]))
println(string("        "))
println(string("RP EOM: ",  ADMM["Residuals"]["Primal"]["EOM"][end], " -- Tolerance: ",ADMM["Tolerance"]["EOM"]))
println(string("RD EOM: ",  ADMM["Residuals"]["Dual"]["EOM"][end], " -- Tolerance: ",ADMM["Tolerance"]["EOM"]))
if scen_number == 4
println(string("RP cCM: ",  ADMM["Residuals"]["Primal"]["cCM"][end], " -- Tolerance: ",ADMM["Tolerance"]["cCM"]))
println(string("RD cCM: ",  ADMM["Residuals"]["Dual"]["cCM"][end], " -- Tolerance: ",ADMM["Tolerance"]["cCM"]))
elseif scen_number == 5
println(string("RP dCM: ",  ADMM["Residuals"]["Primal"]["dCM"][end], " -- Tolerance: ",ADMM["Tolerance"]["dCM"]))
println(string("RD dCM: ",  ADMM["Residuals"]["Dual"]["dCM"][end], " -- Tolerance: ",ADMM["Tolerance"]["dCM"]))
end
println(string("        "))

## 6. Postprocessing and save results 
save_results(mdict, EOM, cCM, dCM, ADMM, results, data, agents, scenario_overview_row, sens_label) 

println("Postprocessing & save results: done")
println("   ")

end #end for loop over run_list (scenario-beta combinations)

println(string("##############################################################################################"))


