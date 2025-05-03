import re
import statistics

# File to analyze
log_file = "performance.log"

# Define patterns for each performance metric
patterns = {
    "LED Indicator Response": r"\[PERF\] LED Indicator Response.*?: ([\d.]+) ms",
    "Web Interface Update": r"\[PERF\] Web Interface Update: ([\d.]+) ms",
    "End-to-End System Response": r"\[PERF\] End-to-End System Response: ([\d.]+) ms"
}

# Store extracted results
results = {}

# Read log file content
with open(log_file, "r") as file:
    log_data = file.read()

# Extract and compute statistics for each metric
for label, pattern in patterns.items():
    matches = re.findall(pattern, log_data)
    values = list(map(float, matches))

    if values:
        results[label] = {
            "Average": round(statistics.mean(values), 2),
            "Std Dev": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
            "Min": round(min(values), 2),
            "Max": round(max(values), 2),
            "Samples": len(values)
        }
    else:
        results[label] = {
            "Average": "N/A",
            "Std Dev": "N/A",
            "Min": "N/A",
            "Max": "N/A",
            "Samples": 0
        }

# Print the formatted performance summary
print("\n📊 PERFORMANCE METRICS SUMMARY (Enhanced)")
print("-" * 90)
print(f"{'Operation':<30} {'Avg (ms)':<10} {'Std Dev':<10} {'Min':<10} {'Max':<10} {'Samples':<10}")
print("-" * 90)

for op, stats in results.items():
    print(f"{op:<30} {str(stats['Average']):<10} {str(stats['Std Dev']):<10} {str(stats['Min']):<10} {str(stats['Max']):<10} {str(stats['Samples']):<10}")

print("-" * 90)
