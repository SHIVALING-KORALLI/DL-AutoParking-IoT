import re
from collections import defaultdict

# Path to your updated log file
log_file = "resource_usage.log"

# Match lines like:
# [YOLO] CPU Time: 0.1234 s, Memory: 1244.07 MB
pattern = re.compile(r"\[([A-Za-z\s]+)\] CPU Time: ([\d.]+) s, Memory: ([\d.]+) MB")

# Storage
cpu_data = defaultdict(list)
mem_data = defaultdict(list)

# Parse log
with open(log_file, "r") as file:
    for line in file:
        match = pattern.search(line)
        if match:
            component = match.group(1).strip()
            cpu_time = float(match.group(2))
            memory = float(match.group(3))
            cpu_data[component].append(cpu_time)
            mem_data[component].append(memory)

# Summary printer
def print_summary(data, label, unit):
    print(f"\n=== {label.upper()} USAGE SUMMARY ===")
    for comp, values in data.items():
        count = len(values)
        avg = sum(values) / count if count else 0
        peak = max(values) if values else 0
        print(f"[{comp}]")
        print(f"  • Entries: {count}")
        print(f"  • Avg {label}: {avg:.4f} {unit}")
        print(f"  • Peak {label}: {peak:.4f} {unit}")
        print("-" * 40)

print_summary(cpu_data, "CPU Time", "s")
print_summary(mem_data, "Memory", "MB")
