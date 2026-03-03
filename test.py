import spacy

# Path to your trained model (final model or best checkpoint)
MODEL_PATH = r"C:\Users\dell\Desktop\Telugu\telugu_ner_complete_fixed_best"

# Load the trained model
nlp = spacy.load(MODEL_PATH)

# Test sentence
test_sentence = "అర్జున్ ఆపిల్ ఫోన్ కొనుగోలు చేశాడు."

# Run NER
doc = nlp(test_sentence)

# Print entities
print("Detected Entities:")
for ent in doc.ents:
    print(f"Text: {ent.text}, Label: {ent.label_}, Start: {ent.start_char}, End: {ent.end_char}")
