from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import joblib
from utils import load_dataset

class SpamClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('classifier', MultinomialNB())
        ])
        self.model_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'spam_classifier.pkl')
    
    def train(self, dataset_path, test_size=0.2, random_state=42):
        """Train the model using the provided dataset"""
        # Load and preprocess dataset
        df = load_dataset(r'C:\Users\Ishaq\Downloads\combined_emails_with_natural_pii.csv')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], 
            df['label'], 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Train the model
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        print("\nConfusion Matrix:")
        print(matrix)
        
        # Save the model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
        print(f"Model saved to {self.model_path}")
        
        return accuracy
    
    def load_model(self):
        """Load trained model from disk"""
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)
            return True
        return False
    
    def predict(self, text):
        """Predict whether a text is spam or not"""
        from utils import preprocess_text
        # Preprocess the input text
        processed_text = preprocess_text(text)
        # Make prediction
        prediction = self.pipeline.predict([processed_text])[0]
        probability = self.pipeline.predict_proba([processed_text])[0]
        
        result = {
            'is_spam': bool(prediction),
            'confidence': float(probability[1]) if prediction == 1 else float(probability[0])
        }
        return result