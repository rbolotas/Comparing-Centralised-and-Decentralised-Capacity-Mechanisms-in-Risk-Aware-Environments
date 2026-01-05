python analyze.py --scenarios 1 2 3 4 5 6 7 --excel --plots

python analyze.py --scenarios 1 2 3 4 5 6 7 --plots

python analyze.py --scenarios 1 2 3 4 5 6 7 --json --plots

The --excel flag in the command is an optional parameter that exports all analysis results to an Excel workbook.
When you include --excel, the script will:
Create a comprehensive Excel file (analysis_results_ref.xlsx) containing:
A summary sheet with key metrics for all scenarios
Comparison tables for generators and consumers
Detailed sheets for each scenario's metrics
Cost breakdowns and revenue analysis
This Excel export is extremely useful for:
Creating tables for your thesis
Examining detailed numerical results
Sharing findings with collaborators who don't use Python
Performing additional custom analysis in Excel

You can run the command without --excel if you only want the visualizations:

python analyze.py --scenarios 1 2 3 4 5 --plots

Or if you want to export to JSON format instead:

python analyze.py --scenarios 1 2 3 4 5 --json --plots

The modular design allows you to choose exactly what outputs you need for your analysis.