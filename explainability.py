# explainability_word_level.py
import spacy
import numpy as np
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
from pathlib import Path

# === Config ===
MODEL_PATH = Path(r"C:\Users\dell\Desktop\Telugu\telugu_ner_model_merged\checkpoint_30")  
LABELS = ["PER", "LOC", "ORG", "MISC", "O"]  # Adjust based on your project

def split_telugu_words(text):
    """
    Split Telugu text into words for LIME.
    Assumes your training text uses spaces between words.
    """
    return text.split()

class TeluguNERExplainer:
    def __init__(self, model_path=MODEL_PATH):
        """Initialize the explainer with a trained model"""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"❌ Model not found at {model_path}")
        print(f"✅ Loading model from {model_path}")
        self.nlp = spacy.load(model_path)
        self.class_names = LABELS
        self.explainer = LimeTextExplainer(
            class_names=self.class_names,
            split_expression=split_telugu_words  # Word-level split
        )
    
    def predict_proba(self, texts):
        """Predict probability distribution for texts"""
        results = []
        for text in texts:
            doc = self.nlp(text)
            probas = np.zeros(len(self.class_names))
            
            for ent in doc.ents:
                if ent.label_ in self.class_names:
                    idx = self.class_names.index(ent.label_)
                    probas[idx] += 1  # Count-based probability
            
            # Normalize
            if probas.sum() > 0:
                probas = probas / probas.sum()
            else:
                probas = np.ones(len(self.class_names)) / len(self.class_names)
            
            results.append(probas)
        
        return np.array(results)
    
    def explain(self, text, num_features=10, num_samples=5000):
        """Explain model prediction for a given text"""
        predict_fn = lambda texts: self.predict_proba(texts)
        explanation = self.explainer.explain_instance(
            text, 
            predict_fn, 
            num_features=num_features, 
            num_samples=num_samples
        )
        return explanation
    
    def visualize_explanation(self, explanation, output_path=None):
        """Visualize the explanation"""
        fig = explanation.as_pyplot_figure()
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"📊 Explanation saved to {output_path}")
        
        plt.show()
        return fig

def main():
    print("=" * 50)
    print("Model Explainability Analysis (Word-Level)")
    print("=" * 50)
    
    try:
        explainer = TeluguNERExplainer()
        
        # Sample texts
        sample_texts = [
            "నరేంద్ర మోదీ భారత ప్రధాన మంత్రి",
            "టాటా స్టీల్ జామ్షెడ్పూర్ లో ఉంది",
            "హైదరాబాద్ తెలంగాణ రాష్ట్ర రాజధాని"
        ]
        
        for i, text in enumerate(sample_texts):
            print(f"\nAnalyzing text {i+1}: '{text}'")
            explanation = explainer.explain(text)
            
            print("Explanation (word → weight):")
            for feature, weight in explanation.as_list():
                print(f"  {feature}: {weight:.4f}")
            
            output_path = f"explanation_{i+1}.png"
            explainer.visualize_explanation(explanation, output_path)
        
        print("\n✅ Explainability analysis completed successfully!")
    
    except Exception as e:
        print(f"❌ Error during explainability analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
