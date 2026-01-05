function define_results!(data::Dict, results::Dict, ADMM::Dict, agents::Dict) 
    # Initialize results["g"]
    results["g"] = Dict()
    for m in agents[:eom]
        results["g"][m] = CircularBuffer{Array{Float64,1}}(data["ADMM"]["CircularBufferSize"]) 
        push!(results["g"][m], zeros(data["General"]["nTimesteps"]))
    end

    # Initialize results["ENS"]
    results["ENS"] = Dict()
    for m in agents[:Cons]
        results["ENS"][m] = CircularBuffer{Array{Float64,1}}(data["ADMM"]["CircularBufferSize"]) 
        push!(results["ENS"][m], zeros(data["General"]["nTimesteps"]))
    end

    # Initialize results["D"]
    results["D"] = Dict()
    for m in agents[:Cons]
        results["D"][m] = CircularBuffer{Array{Float64,1}}(data["ADMM"]["CircularBufferSize"]) 
        push!(results["D"][m], zeros(data["General"]["nTimesteps"]))
    end

    # Initialize results["D_elastic"]
    results["D_elastic"] = Dict()
    for m in agents[:Cons]
        results["D_elastic"][m] = CircularBuffer{Array{Float64,1}}(data["ADMM"]["CircularBufferSize"]) 
        push!(results["D_elastic"][m], zeros(data["General"]["nTimesteps"]))
    end

    # Initialize results["ENS_involuntary"] for separated involuntary curtailment
    results["ENS_involuntary"] = Dict()
    for m in agents[:Cons]
        results["ENS_involuntary"][m] = CircularBuffer{Array{Float64,1}}(data["ADMM"]["CircularBufferSize"]) 
        push!(results["ENS_involuntary"][m], zeros(data["General"]["nTimesteps"]))
    end

    # Initialize results["PV_curt"] (curtailed PV generation)
    results["PV_curt"] = Dict()
    for m in agents[:Cons]
        results["PV_curt"][m] = CircularBuffer{Array{Float64,1}}(data["ADMM"]["CircularBufferSize"]) 
        push!(results["PV_curt"][m], zeros(data["General"]["nTimesteps"]))
    end

    # Initialize results["C"]
    results["C"] = Dict()
    for m in agents[:eom]
        results["C"][m] = CircularBuffer{Float64}(data["ADMM"]["CircularBufferSize"]) 
        push!(results["C"][m], 0)
    end
    
    # Initialize results["C_cCM"]
    results["C_cCM"] = Dict()
    for m in agents[:cCM]
        results["C_cCM"][m] = CircularBuffer{Float64}(data["ADMM"]["CircularBufferSize"]) 
        push!(results["C_cCM"][m], 0)
    end

    # Initialize results["C_dCM"]
    results["C_dCM"] = Dict()
    for m in agents[:dCM]
        results["C_dCM"][m] = CircularBuffer{Float64}(data["ADMM"]["CircularBufferSize"]) 
        push!(results["C_dCM"][m], 0)
    end

    # Initialize results["λ"]
    results["λ"] = Dict()
    #results["λ"]["EOM"] = CircularBuffer{Array{Float64,1}}(data["ADMM"]["CircularBufferSize"])
    #push!(results["λ"]["EOM"], zeros(data["General"]["nTimesteps"]))
    
    # Warm-start EOM price at min marginal cost of active generators
    mc = minimum([data["Generators"][m]["b"] for m in agents[:Gen]])
    results["λ"]["EOM"] = CircularBuffer{Array{Float64,1}}(data["ADMM"]["CircularBufferSize"])
    push!(results["λ"]["EOM"], fill(mc, data["General"]["nTimesteps"]))

    results["λ"]["cCM"] = CircularBuffer{Float64}(data["ADMM"]["CircularBufferSize"])
    push!(results["λ"]["cCM"], 0.0)
    
    results["λ"]["dCM"] = CircularBuffer{Float64}(data["ADMM"]["CircularBufferSize"])
    push!(results["λ"]["dCM"], 0.0004) # try a price

    # Initialize ADMM components
    ADMM["Imbalances"] = Dict()
    ADMM["Imbalances"]["EOM"] = CircularBuffer{Array{Float64,1}}(data["ADMM"]["CircularBufferSize"])
    push!(ADMM["Imbalances"]["EOM"], zeros(data["General"]["nTimesteps"]))
    ADMM["Imbalances"]["cCM"] = CircularBuffer{Float64}(data["ADMM"]["CircularBufferSize"])
    push!(ADMM["Imbalances"]["cCM"], 0.0)
    ADMM["Imbalances"]["dCM"] = CircularBuffer{Float64}(data["ADMM"]["CircularBufferSize"])
    push!(ADMM["Imbalances"]["dCM"], 0.0)

    # Initialize residuals
    ADMM["Residuals"] = Dict()
    ADMM["Residuals"]["Primal"] = Dict()
    ADMM["Residuals"]["Dual"] = Dict()
    
    # Create circular buffers for residuals
    ADMM["Residuals"]["Primal"]["EOM"] = CircularBuffer{Float64}(data["ADMM"]["CircularBufferSize"])
    ADMM["Residuals"]["Dual"]["EOM"] = CircularBuffer{Float64}(data["ADMM"]["CircularBufferSize"])
    ADMM["Residuals"]["Primal"]["cCM"] = CircularBuffer{Float64}(data["ADMM"]["CircularBufferSize"])
    ADMM["Residuals"]["Dual"]["cCM"] = CircularBuffer{Float64}(data["ADMM"]["CircularBufferSize"])
    ADMM["Residuals"]["Primal"]["dCM"] = CircularBuffer{Float64}(data["ADMM"]["CircularBufferSize"])
    ADMM["Residuals"]["Dual"]["dCM"] = CircularBuffer{Float64}(data["ADMM"]["CircularBufferSize"])

    # Initialize residuals with zeros
    push!(ADMM["Residuals"]["Primal"]["EOM"], 0.0)
    push!(ADMM["Residuals"]["Dual"]["EOM"], 0.0)
    push!(ADMM["Residuals"]["Primal"]["cCM"], 0.0)
    push!(ADMM["Residuals"]["Dual"]["cCM"], 0.0)
    push!(ADMM["Residuals"]["Primal"]["dCM"], 0.0)
    push!(ADMM["Residuals"]["Dual"]["dCM"], 0.0)

    # Initialize tolerances
    ADMM["Tolerance"] = Dict()
    ADMM["Tolerance"]["EOM"] = data["ADMM"]["epsilon"]
    ADMM["Tolerance"]["cCM"] = data["ADMM"]["epsilon"]
    ADMM["Tolerance"]["dCM"] = data["ADMM"]["epsilon"]

    # Initialize rho values
    ADMM["ρ"] = Dict()
    ADMM["ρ"]["EOM"] = CircularBuffer{Float64}(data["ADMM"]["CircularBufferSize"])
    push!(ADMM["ρ"]["EOM"], data["ADMM"]["rho_EOM"])
    ADMM["ρ"]["cCM"] = CircularBuffer{Float64}(data["ADMM"]["CircularBufferSize"])
    push!(ADMM["ρ"]["cCM"], data["ADMM"]["rho_cCM"])
    ADMM["ρ"]["dCM"] = CircularBuffer{Float64}(data["ADMM"]["CircularBufferSize"])
    push!(ADMM["ρ"]["dCM"], data["ADMM"]["rho_dCM"])

    # Initialize iteration counter and walltime
    ADMM["n_iter"] = 1 
    ADMM["walltime"] = 0
    
    return results, ADMM
end
