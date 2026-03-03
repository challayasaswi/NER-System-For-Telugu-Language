import json
import config
from pathlib import Path

def load_json_file(path):
    """Load JSON file that may be array or JSON lines"""
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)  # reset pointer

        if first_char == "[":  # JSON array
            return json.load(f)
        else:  # JSON Lines (NDJSON)
            return [json.loads(line) for line in f if line.strip()]

def merge_datasets():
    # Load Naamaapadam train
    train_path = Path(config.DATA_DIR) / config.TRAIN_FILE
    train_data = load_json_file(train_path)

    # Load pseudo-labeled wiki
    pseudo_path = Path(config.PSEUDO_LABELED_DATA)
    pseudo_data = load_json_file(pseudo_path)

    print(f"Naamaapadam samples: {len(train_data)}")
    print(f"Pseudo-labeled wiki samples: {len(pseudo_data)}")

    # Merge
    merged_data = train_data + pseudo_data
    print(f"Total merged samples: {len(merged_data)}")

    # Save new training file (as array for simplicity)
    merged_path = Path(config.DATA_DIR) / "train_merged.json"
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Merged dataset saved at {merged_path}")

if __name__ == "__main__":
    merge_datasets()
