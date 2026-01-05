function define_generator_parameters!(mod::Model, data::Dict, ts::DataFrame, m::String)
    # Parameters 
    mod.ext[:parameters][:A] = data["Generators"][m]["a"]
    mod.ext[:parameters][:B] = data["Generators"][m]["b"]
    r = mod.ext[:parameters][:r] = data["General"]["r"] #discount rate
    n = mod.ext[:parameters][:n] = data["Generators"][m]["n"] #lifetime of the generator
    
    # Raw INV value from config and basic calculation
    raw_inv = data["Generators"][m]["INV"]
    crf = r / (1 - (1 + r)^(-n))
    
    # Calculate normal annual hourly investment cost
    # For representative periods, use normal hourly rate (not scaled)
    # The representative hours should capture typical investment cost burden
    hours_per_year = 8760
    hourly_inv = raw_inv * crf / hours_per_year
    
    # Store the hourly investment cost
    mod.ext[:parameters][:INV_h] = hourly_inv

    mod.ext[:parameters][:β_gc] = data["RiskMetrics"]["β_gc"]
    mod.ext[:parameters][:γ_gc] = data["RiskMetrics"]["γ_gc"]

    # Availability factors
    if haskey(data["Generators"][m], "AF")
        mod.ext[:timeseries][:AC] = ts[!, data["Generators"][m]["AF"]]  
    else
        mod.ext[:timeseries][:AC] = ones(data["General"]["nTimesteps"]) 
    end 

    # Derating factor for capacity
    mod.ext[:parameters][:F_cCM] = data["Generators"][m]["F_cm"]  
    
    # Wind capacity constraint (for wind generators)
    if haskey(data["Generators"][m], "C")
        mod.ext[:parameters][:C_wind] = data["Generators"][m]["C"]
    end
    
    # Reliability option strike price (formerly scarcity price limit)
    mod.ext[:parameters][:λ_EOM_scr] = data["dCM"]["λ_EOM_scr"]
   
    return mod


end

