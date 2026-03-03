import subprocess
import json
from pathlib import Path
import config  # Ensure config.VAL_DATA points to your dev/test file

def evaluate_checkpoint(model_path, val_data):
    """Run spaCy evaluate command and return metrics"""
    output_file = Path("results") / f"{model_path.name}_results.json"
    output_file.parent.mkdir(exist_ok=True)

    cmd = [
        "python", "-m", "spacy", "evaluate",
        str(model_path),
        str(val_data),
        "--output", str(output_file)
    ]
    print(f"\n🔍 Evaluating checkpoint: {model_path.name} ...")
    subprocess.run(cmd, check=True)

    with open(output_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    return results


def main():
    # Path where your checkpoints/models are stored
    checkpoints_dir = Path(r"C:\Users\dell\Desktop\Telugu\telugu_ner_model_merged")

    # Path to validation data (use dev.spacy for checkpoint selection)
    val_data = Path(config.VAL_DATA) if hasattr(config, "VAL_DATA") else Path(r"C:\Users\dell\Desktop\Telugu\dev.spacy")

    best_model = None
    best_f1 = -1.0

    if not checkpoints_dir.exists():
        print(f"❌ Checkpoints folder not found: {checkpoints_dir}")
        return

    for checkpoint in sorted(checkpoints_dir.iterdir()):
        if checkpoint.is_dir() and (checkpoint / "meta.json").exists():
            try:
                results = evaluate_checkpoint(checkpoint, val_data)
                f1 = results.get("ents_f", 0)
                p = results.get("ents_p", 0)
                r = results.get("ents_r", 0)

                print(f"📊 {checkpoint.name} → P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}")

                if f1 > best_f1:
                    best_f1 = f1
                    best_model = checkpoint
            except Exception as e:
                print(f"⚠️ Skipping {checkpoint.name} due to error: {e}")
        else:
            print(f"ℹ️ Skipping {checkpoint.name} (not a valid spaCy model directory)")

    if best_model:
        print("\n==============================")
        print(f"✅ Best model: {best_model} with F1-score = {best_f1:.4f}")
        print("==============================")
    else:
        print("❌ No valid checkpoints found.")


if __name__ == "__main__":
    main()
