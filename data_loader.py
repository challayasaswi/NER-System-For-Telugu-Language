import json
from pathlib import Path
import spacy
from spacy.tokens import DocBin, Doc

# Paths
DATASET_DIR = Path(r'C:\Users\dell\Desktop\Telugu')
TRAIN_FILE = DATASET_DIR / "te_train.json"
VAL_FILE   = DATASET_DIR / "te_val.json"
TEST_FILE  = DATASET_DIR / "te_test.json"

# Output files
TRAIN_SPACY = DATASET_DIR / "train_subset.spacy"
DEV_SPACY   = DATASET_DIR / "dev.spacy"
TEST_SPACY  = DATASET_DIR / "test.spacy"

# Number of train examples to load (adjust based on RAM)
SUBSET_SIZE = 50000  # e.g., 50k examples

def load_json(file_path):
    """Load JSON or JSONL file safely"""
    with open(file_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":  # JSON array
            return json.load(f)
        else:  # JSONL format
            return [json.loads(line) for line in f if line.strip()]

def convert_to_docbin(data, nlp):
    """Convert JSON dataset into SpaCy DocBin"""
    doc_bin = DocBin()
    for example in data:
        tokens = example["words"]
        labels = example["ner"]

        # Remove empty tokens and corresponding labels
        filtered = [(t, l) for t, l in zip(tokens, labels) if t.strip() != ""]
        if not filtered:
            continue
        tokens, labels = zip(*filtered)

        doc = Doc(nlp.vocab, words=list(tokens))
        ents = []
        char_idx = 0
        for token, label in zip(tokens, labels):
            token_len = len(token)
            if label != "O":
                span = doc.char_span(char_idx, char_idx + token_len, label=label)
                if span is not None:
                    ents.append(span)
            char_idx += token_len + 1  # +1 for space

        doc.ents = ents
        doc_bin.add(doc)

    return doc_bin

def prepare_training_data(subset=True):
    """
    Prepare DocBins. If subset=True, creates train subset and saves DocBins.
    Returns DocBin objects by loading from disk (memory-efficient).
    """
    nlp = spacy.blank("te")

    # If .spacy files do not exist, create them
    if not TRAIN_SPACY.exists() or not DEV_SPACY.exists() or not TEST_SPACY.exists():
        print("📂 Saving .spacy files...")
        # Load train subset
        train_data = load_json(TRAIN_FILE)
        train_data_subset = train_data[:SUBSET_SIZE] if subset else train_data
        train_doc_bin = convert_to_docbin(train_data_subset, nlp)
        train_doc_bin.to_disk(TRAIN_SPACY)
        print(f"✅ Train subset saved: {TRAIN_SPACY} ({len(train_data_subset)} examples)")

        # Load validation and test sets
        val_data = load_json(VAL_FILE)
        dev_doc_bin = convert_to_docbin(val_data, nlp)
        dev_doc_bin.to_disk(DEV_SPACY)

        test_data = load_json(TEST_FILE)
        test_doc_bin = convert_to_docbin(test_data, nlp)
        test_doc_bin.to_disk(TEST_SPACY)
        print(f"✅ Validation and test sets saved: {DEV_SPACY} ({len(val_data)}), {TEST_SPACY} ({len(test_data)})")
    else:
        print("📂 Loading existing .spacy files from disk...")

    # Load DocBins from disk (memory-efficient)
    train_doc_bin = DocBin().from_disk(TRAIN_SPACY)
    dev_doc_bin = DocBin().from_disk(DEV_SPACY)
    test_doc_bin = DocBin().from_disk(TEST_SPACY)

    return train_doc_bin, dev_doc_bin, test_doc_bin

if __name__ == "__main__":
    prepare_training_data()
