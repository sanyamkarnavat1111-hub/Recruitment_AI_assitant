import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

class ScreenAI:
    MODEL_PATH = 'AI_Model/resume_screening_model.pkl'

    def __init__(self):
        if not os.path.exists(self.MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {self.MODEL_PATH}")

        with open(self.MODEL_PATH, 'rb') as f:
            loaded = pickle.load(f)

        self.vectorizer: TfidfVectorizer = loaded['vectorizer']
        self.classifier = loaded['classifier']

        # Store known classes to avoid partial_fit errors
        self.classes = self.classifier.classes_

    def predict_hiring_decision(self, resume_text, job_description):
        combined_text = (resume_text + " " + job_description).strip()
        text_vector = self.vectorizer.transform([combined_text])

        prediction = self.classifier.predict(text_vector)[0]
        probability = self.classifier.predict_proba(text_vector)[0]

        decision = "Hire" if prediction == 1 else "Reject"
        
        return {
            'Decision': decision,
            'Probability': round(probability[1], 4),
            'Confidence': round(max(probability), 4)
        }

    def improve_model(self, resume_text, job_description, true_label):
        """
        Incrementally improve the model with ONE new labeled example.
        Updates vectorizer (if new words) and classifier via partial_fit.
        Saves the updated model automatically.

        Args:
            resume_text (str): Raw resume text
            job_description (str): Raw job description
            true_label (int): 1 = Hire, 0 = Reject
        """
        if true_label not in [0, 1]:
            raise ValueError("true_label must be 0 (Reject) or 1 (Hire)")

        combined_text = (resume_text + " " + job_description).strip()

        # --- 1. Update Vectorizer (optional: expand vocab) ---
        # Transform with current vocab
        X_new = self.vectorizer.transform([combined_text])

        # Check if any new words exist (all zeros = unseen words)
        if X_new.nnz == 0:  # No known words → fit new ones
            print("New words detected. Updating vectorizer vocabulary...")
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                strip_accents='unicode',
                stop_words='english',
                ngram_range=(1, 2),
                vocabulary=self.vectorizer.vocabulary_  # preserve old
            )
            # Re-fit on old + new text
            # We need to keep old data in memory or re-transform — here we just expand
            self.vectorizer.fit([combined_text])  # This adds new terms
            X_new = self.vectorizer.transform([combined_text])

        # --- 2. Partial fit the classifier ---
        try:
            self.classifier.partial_fit(X_new, [true_label], classes=self.classes)
            print(f"Model improved with label: {true_label} (Hire)" if true_label == 1 else "Reject")
        except Exception as e:
            print(f"partial_fit failed: {e}")
            return False

        # --- 3. Save updated model ---
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier
        }
        try:
            with open(self.MODEL_PATH, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model updated and saved to {self.MODEL_PATH}")
            return True
        except Exception as e:
            print(f"Failed to save model: {e}")
            return False
        

if __name__ == "__main__":
    ai = ScreenAI()

    # === 1. Predict ===
    resume_text_sample = "jason jones ... (your long resume)"
    jd_sample_text = "part passionate team forefront machine learning ecommerce specialist..."

    result = ai.predict_hiring_decision(resume_text_sample, jd_sample_text)
    print("Prediction:", result)

    # === 2. Improve model with feedback (e.g., human corrected it) ===
    print("\nImproving model with human feedback...")
    ai.improve_model(
        resume_text=resume_text_sample,
        job_description=jd_sample_text,
        true_label=1  # Human says: "Actually, HIRE this one!"
    )

    # === 3. Predict again — should now lean toward Hire ===
    result2 = ai.predict_hiring_decision(resume_text_sample, jd_sample_text)
    print("After feedback:", result2)