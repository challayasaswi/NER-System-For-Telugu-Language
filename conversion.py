import json
from pathlib import Path

DATA_DIR = Path(r"C:\Users\dell\Desktop\Telugu")
JSON_FILE = DATA_DIR / "train_merged.json"
NDJSON_FILE = DATA_DIR / "train_merged.ndjson"

# Load the JSON array
with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Write as NDJSON (one JSON object per line)
with open(NDJSON_FILE, "w", encoding="utf-8") as f:
    for item in data:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ Converted {JSON_FILE} to NDJSON: {NDJSON_FILE}")
