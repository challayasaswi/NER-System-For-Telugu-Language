import json
from pathlib import Path

# Hardcoded paths
val_file = Path(r"C:\Users\dell\Desktop\Telugu\te_val.json")  # Path to your te_val.json
output_file = Path(r"C:\Users\dell\Desktop\Telugu\val_merged.json")  # Output path

# Load te_val.json
with open(val_file, "r", encoding="utf-8") as f:
    val_data = json.load(f)

# Save to val_merged.json
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

print(f"✅ val_merged.json created with {len(val_data)} examples at {output_file}")
