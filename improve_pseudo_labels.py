import json
from collections import Counter

def analyze_and_improve_pseudo_labels():
    """Analyze pseudo-labels and improve quality"""
    
    with open("train_merged.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print("🔍 Analyzing pseudo-label quality...")
    
    # Analyze entity distribution
    entity_stats = Counter()
    sentence_lengths = []
    entity_density = []
    
    for ex in data:
        words = ex["words"]
        labels = ex["ner"]
        
        # Entity statistics
        entities = []
        current_entity = []
        current_label = None
        
        for word, label in zip(words, labels):
            if label.startswith("B-"):
                if current_entity:
                    entities.append((" ".join(current_entity), current_label))
                current_entity = [word]
                current_label = label[2:]
            elif label.startswith("I-") and current_label == label[2:]:
                current_entity.append(word)
            else:
                if current_entity:
                    entities.append((" ".join(current_entity), current_label))
                current_entity = []
                current_label = None
        
        if current_entity:
            entities.append((" ".join(current_entity), current_label))
        
        for entity_text, label in entities:
            entity_stats[label] += 1
        
        # Quality metrics
        sentence_lengths.append(len(words))
        entity_count = sum(1 for label in labels if label != "O")
        density = entity_count / len(words) if words else 0
        entity_density.append(density)
    
    print("\n📊 Entity Statistics:")
    for label, count in entity_stats.most_common():
        print(f"  {label}: {count}")
    
    print(f"\n📏 Average sentence length: {sum(sentence_lengths)/len(sentence_lengths):.1f}")
    print(f"📈 Average entity density: {sum(entity_density)/len(entity_density):.1%}")
    
    # Filter low-quality examples
    improved_data = []
    for ex in data:
        words = ex["words"]
        labels = ex["ner"]
        
        # Quality filters
        entity_count = sum(1 for label in labels if label != "O")
        density = entity_count / len(words)
        
        # Keep if:
        # - Reasonable length (5-50 words)
        # - Reasonable entity density (10-60%)
        # - No single-character entities
        if (5 <= len(words) <= 50 and 
            0.1 <= density <= 0.6 and
            not any(len(word) == 1 and label != "O" for word, label in zip(words, labels))):
            improved_data.append(ex)
    
    print(f"\n✅ Improved dataset: {len(improved_data)}/{len(data)} examples kept")
    
    # Save improved dataset
    with open("train_merged_improved.json", "w", encoding="utf-8") as f:
        json.dump(improved_data, f, ensure_ascii=False, indent=2)
    
    return improved_data

if __name__ == "__main__":
    analyze_and_improve_pseudo_labels()