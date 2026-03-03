import spacy
from pathlib import Path

# === Paths ===
DATA_DIR = Path(r"C:\Users\dell\Desktop\Telugu")
BEST_MODEL_PATH = Path(DATA_DIR / "best_model.txt").read_text().strip()  # Load best checkpoint path

# === Load the model ===
print(f"🔍 Loading best model from {BEST_MODEL_PATH} ...")
nlp = spacy.load(BEST_MODEL_PATH)

# === Function to test NER on a sentence ===
def test_ner(text):
    doc = nlp(text)
    print(f"\nInput Text: {text}\n")
    if doc.ents:
        print("Detected Entities:")
        for ent in doc.ents:
            print(f" - {ent.text} → {ent.label_}")
    else:
        print("No entities detected.")

# === Example usage ===
if __name__ == "__main__":
    while True:
        sentence = input("\nEnter a Telugu sentence (or 'exit' to quit):\n> ")
        if sentence.lower() == "exit":
            break
        test_ner(sentence)
