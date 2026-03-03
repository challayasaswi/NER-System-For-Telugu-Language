import json
import random
from pathlib import Path
import spacy
from spacy.training import Example
from spacy.util import minibatch
from tqdm import tqdm
from config import N_ITERATIONS, DROPOUT, CHECKPOINT_EVERY, DATA_DIR

# === Settings ===
CHUNK_SIZE = 5000      # Reduce to 2000-3000 if memory is low
BATCH_SIZE = 16
MODEL_DIR = "telugu_ner_model_merged_new"

# === Stream dataset in chunks ===
def stream_dataset(path, chunk_size=CHUNK_SIZE):
    path = Path(path)
    ext = path.suffix.lower()
    
    if ext == ".ndjson":
        with open(path, "r", encoding="utf-8") as f:
            chunk = []
            for line in f:
                chunk.append(json.loads(line))
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk
    else:  # JSON array
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for i in range(0, len(data), chunk_size):
                yield data[i:i+chunk_size]

# === Convert example to spaCy Example object ===
def make_example(nlp, example):
    words = example["words"]
    labels = example["ner"]
    entities = []
    start, current_label = None, None
    text = " ".join(words)
    idx_map, pointer = {}, 0
    for i, word in enumerate(words):
        idx_map[i] = pointer
        pointer += len(word) + 1
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if start is not None:
                entities.append((start, idx_map[i-1] + len(words[i-1]), current_label))
            start = idx_map[i]
            current_label = label[2:]
        elif label.startswith("I-") and current_label:
            continue
        else:
            if start is not None:
                entities.append((start, idx_map[i-1] + len(words[i-1]), current_label))
                start, current_label = None, None
    if start is not None:
        entities.append((start, idx_map[len(words)-1] + len(words[-1]), current_label))
    doc = nlp.make_doc(text)
    return Example.from_dict(doc, {"entities": entities})

# === Save model ===
def save_model(nlp, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_dir)
    print(f"✅ Model saved at {output_dir}")

# === Find last checkpoint if exists ===
def get_last_checkpoint(model_dir):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        return None
    checkpoints = [d for d in model_dir.iterdir() if d.is_dir() and "checkpoint_" in d.name]
    if not checkpoints:
        return None
    # Sort by checkpoint number
    checkpoints.sort(key=lambda x: int(x.name.split("_")[-1]))
    return checkpoints[-1]

# === Memory-efficient training with checkpoint resume ===
def train():
    dataset_path = Path(DATA_DIR) / "train_merged.json"  # Use .json or .ndjson
    print(f"🚀 Training on full dataset: {dataset_path}")

    last_checkpoint = get_last_checkpoint(MODEL_DIR)
    if last_checkpoint:
        print(f"🔄 Resuming training from {last_checkpoint}")
        nlp = spacy.load(last_checkpoint)
        # Extract last iteration number from checkpoint name
        start_iteration = int(last_checkpoint.name.split("_")[-1])
    else:
        print("⚡ Starting training from scratch")
        nlp = spacy.blank("te")
        if "ner" not in nlp.pipe_names:
            nlp.add_pipe("ner")
        start_iteration = 0

        # Add labels from first small chunk
        first_chunk = next(stream_dataset(dataset_path, chunk_size=1000))
        for ex in first_chunk:
            for label in ex["ner"]:
                if label != "O":
                    nlp.get_pipe("ner").add_label(label.split("-")[-1])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()

        for itn in range(start_iteration, N_ITERATIONS):
            print(f"\n--- Iteration {itn+1}/{N_ITERATIONS} ---")
            losses = {}
            chunk_count = 0

            for chunk in stream_dataset(dataset_path, chunk_size=CHUNK_SIZE):
                chunk_count += 1
                print(f"\nProcessing chunk {chunk_count} ({len(chunk)} examples)...")
                random.shuffle(chunk)

                # Memory-efficient batching
                for batch in tqdm(minibatch(chunk, size=BATCH_SIZE), desc="Updating batches", ncols=100):
                    examples = [make_example(nlp, ex) for ex in batch]
                    nlp.update(examples, drop=DROPOUT, losses=losses)

            print(f"Iteration {itn+1} - Losses: {losses}")

            # Save checkpoint
            if (itn + 1) % CHECKPOINT_EVERY == 0:
                save_model(nlp, Path(MODEL_DIR) / f"checkpoint_{itn+1}")

    # Save final model
    save_model(nlp, MODEL_DIR)
    print("🎉 Training complete!")

if __name__ == "__main__":
    train()
