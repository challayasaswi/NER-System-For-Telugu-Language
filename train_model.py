import spacy
from spacy.training import Example
from spacy.util import minibatch
from pathlib import Path
from config import MODEL_DIR, N_ITERATIONS, DROPOUT, BATCH_SIZE, CHECKPOINT_EVERY
from data_loader import TRAIN_SPACY, DEV_SPACY, TEST_SPACY
from utils import fix_random_seed, set_device, print_banner
from spacy.tokens import DocBin

def create_blank_model():
    """Create a blank Telugu NER model"""
    nlp = spacy.blank("te")
    if "ner" not in nlp.pipe_names:
        nlp.add_pipe("ner")
    return nlp

def save_model(nlp, output_dir):
    """Save spaCy model to disk"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_dir)
    print(f"✅ Model saved at {output_dir}")

def stream_docs_from_disk(docbin_path, nlp):
    """Generator: yields Doc objects one by one from a DocBin file on disk"""
    doc_bin = DocBin().from_disk(docbin_path)
    for doc in doc_bin.get_docs(nlp.vocab):
        yield doc

def train_model_memory_efficient():
    """Train model streaming batches directly from disk (memory-efficient)"""
    fix_random_seed(0)
    device = set_device()
    print_banner("Training Telugu NER Model")

    nlp = create_blank_model()
    ner = nlp.get_pipe("ner")

    # Add labels from training data by scanning a small sample
    sample_batches = minibatch(stream_docs_from_disk(TRAIN_SPACY, nlp), size=100)
    for batch in sample_batches:
        for doc in batch:
            for ent in doc.ents:
                ner.add_label(ent.label_)

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()

        for itn in range(N_ITERATIONS):
            losses = {}

            # Stream training data in batches
            doc_generator = stream_docs_from_disk(TRAIN_SPACY, nlp)
            batches = minibatch(doc_generator, size=BATCH_SIZE)

            for batch in batches:
                examples = [
                    Example.from_dict(
                        doc,
                        {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]}
                    )
                    for doc in batch
                ]
                nlp.update(examples, drop=DROPOUT, losses=losses)

            print(f"Iteration {itn+1}/{N_ITERATIONS} - Losses: {losses}")

            # Save checkpoints periodically
            if (itn + 1) % CHECKPOINT_EVERY == 0:
                save_model(nlp, Path(MODEL_DIR) / f"checkpoint_{itn+1}")

    # Save final model
    save_model(nlp, MODEL_DIR)
    print("✅ Training complete. Model saved.")

    # Evaluate on dev set
    evaluate_model(nlp, DEV_SPACY)

    return nlp

def evaluate_model(nlp, dev_docbin_path):
    """Evaluate model performance on validation set using streaming"""
    print("\nEvaluating on validation data...")
    dev_docs = list(DocBin().from_disk(dev_docbin_path).get_docs(nlp.vocab))
    examples = [
        Example.from_dict(doc, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]})
        for doc in dev_docs
    ]
    scorer = nlp.evaluate(examples)
    print("Evaluation Results:", scorer)
    return scorer

if __name__ == "__main__":
    train_model_memory_efficient()
