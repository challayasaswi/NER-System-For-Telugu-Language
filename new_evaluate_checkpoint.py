import spacy
import json
from pathlib import Path
from spacy.training import Example

# === Paths ===
MODEL_DIR = Path("telugu_ner_model_merged_new")
DATA_DIR = Path(r"C:\Users\dell\Desktop\Telugu")  # Folder containing val_merged.json

# === Load validation dataset ===
def load_val_data():
    with open(DATA_DIR / "val_merged.json", "r", encoding="utf-8") as f:
        return json.load(f)

# === Convert dataset to spaCy Example objects ===
def convert_to_examples(nlp, dataset):
    examples = []
    for ex in dataset:
        words = ex["words"]
        labels = ex["ner"]

        text = " ".join(words)
        ents = []
        idx_map, pointer = {}, 0
        start, current_label = None, None

        # Map word indices to character offsets
        for i, word in enumerate(words):
            idx_map[i] = pointer
            pointer += len(word) + 1  # +1 for space

        # Convert BIO labels to (start, end, label)
        for i, label in enumerate(labels):
            if label.startswith("B-"):
                if start is not None:
                    ents.append((start, idx_map[i-1] + len(words[i-1]), current_label))
                start = idx_map[i]
                current_label = label[2:]
            elif label.startswith("I-") and current_label:
                continue
            else:
                if start is not None:
                    ents.append((start, idx_map[i-1] + len(words[i-1]), current_label))
                    start, current_label = None, None

        if start is not None:
            ents.append((start, idx_map[len(words)-1] + len(words[-1]), current_label))

        doc = nlp.make_doc(text)
        examples.append(Example.from_dict(doc, {"entities": ents}))
    return examples

# === Evaluate a single checkpoint ===
def evaluate_checkpoint(model_path, val_examples):
    nlp = spacy.load(model_path)
    return nlp.evaluate(val_examples)

# === Main function ===
def main():
    val_data = load_val_data()
    best_f1 = 0
    best_model = None

    # Convert examples once (to reuse for all checkpoints)
    # We'll pass the nlp object later to align tokenizer
    temp_nlp = spacy.blank("te")
    val_examples = convert_to_examples(temp_nlp, val_data)

    # Loop through all checkpoint folders
    for checkpoint in MODEL_DIR.glob("checkpoint_*"):
        print(f"🔍 Evaluating {checkpoint} ...")
        results = evaluate_checkpoint(checkpoint, val_examples)
        f1 = results["ents_f"]

        print(f"📊 {checkpoint.name} → Precision: {results['ents_p']:.4f}, Recall: {results['ents_r']:.4f}, F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = checkpoint

    # Also evaluate the final saved model
    final_model_path = MODEL_DIR
    print(f"\n🔍 Evaluating final saved model {final_model_path} ...")
    results_final = evaluate_checkpoint(final_model_path, val_examples)
    f1_final = results_final["ents_f"]
    print(f"📊 Final model → Precision: {results_final['ents_p']:.4f}, Recall: {results_final['ents_r']:.4f}, F1: {f1_final:.4f}")

    # Compare final model with best checkpoint
    if f1_final > best_f1:
        best_f1 = f1_final
        best_model = final_model_path

    print("\n✅ Best model:", best_model)
    print("📊 Best F1 Score:", best_f1)

    # Save best checkpoint/model path for later
    with open(DATA_DIR / "best_model.txt", "w", encoding="utf-8") as f:
        f.write(str(best_model))
    print(f"💾 Best model path saved to {DATA_DIR / 'best_model.txt'}")

if __name__ == "__main__":
    main()
