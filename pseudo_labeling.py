# pseudo_labeling.py
import json
from pathlib import Path
import spacy
import config

def load_trained_model(model_path=None):
    """Load the trained spaCy model"""
    if model_path is None:
        model_path = config.MODEL_DIR

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"✅ Loading model from {model_path}")
    return spacy.load(model_path)

def generate_pseudo_labels_for_text(nlp, text):
    """Generate pseudo-labels for a single text"""
    # Skip long documents if configured
    if config.SKIP_LONG_DOCS and len(text) > config.MAX_DOC_LENGTH:
        return None

    doc = nlp(text)
    words = [token.text for token in doc]
    ner = ["O"] * len(words)

    # Use ent.label_ directly to match Naamaapadam format
    for ent in doc.ents:
        for token in ent:
            ner[token.i] = ent.label_  # Correct: B-PER, I-LOC, etc.

    return {"words": words, "ner": ner}

def main():
    print("=" * 50)
    print("Generating Pseudo-Labels (Memory-Efficient, JSON Array)")
    print("=" * 50)

    try:
        # Load trained model
        nlp = load_trained_model(config.BEST_CHECKPOINT_DIR)

        # Prepare output file
        if config.PSEUDO_LABELED_DATA is None:
            output_path = Path(config.DATA_DIR) / "pseudo_labels.json"
        else:
            output_path = Path(config.PSEUDO_LABELED_DATA)
        if output_path.exists():
            output_path.unlink()  # Remove old file if exists

        # Stream unlabeled texts one by one
        wiki_dir = Path(config.WIKI_DIR)
        txt_files = list(wiki_dir.glob("*.txt"))
        print(f"📂 Found {len(txt_files)} files to process")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("[\n")  # Start JSON array
            first_example = True

            for i, file in enumerate(txt_files, start=1):
                with open(file, "r", encoding="utf-8") as file_f:
                    text = file_f.read().strip()
                    if not text:
                        continue

                    example = generate_pseudo_labels_for_text(nlp, text)
                    if example:
                        example["id"] = f"pseudo-{i}"

                        # Write comma between examples
                        if not first_example:
                            f.write(",\n")
                        else:
                            first_example = False

                        json.dump(example, f, ensure_ascii=False)

                if i % 100 == 0:
                    print(f"Processed {i}/{len(txt_files)} files")

            f.write("\n]\n")  # End JSON array

        print(f"🎉 Pseudo-labeling completed! Output saved at {output_path}")

    except Exception as e:
        print(f"❌ Error during pseudo-labeling: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
