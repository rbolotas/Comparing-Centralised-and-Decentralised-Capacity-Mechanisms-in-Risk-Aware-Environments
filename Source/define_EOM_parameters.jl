function define_EOM_parameters!(EOM::Dict,data::Dict,ts::DataFrame,scenario_overview_row::DataFrameRow)
    # timeseries
    EOM["HV_LOAD"] = ts[!,:HV_LOAD][1:data["General"]["nTimesteps"]]
    
    return EOM
end