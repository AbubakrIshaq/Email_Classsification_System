from typing import List, Dict, Any, Tuple
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np

class EmailClassifier:
    """Email classification model using ML techniques"""
    
    def __init__(self):
        """Initialize the email classifier"""
        # Define email categories
        self.categories = [
            "Billing Issues", 
            "Technical Support", 
            "Account Management",
            "Product Inquiry",
            "Return Request",
            "Shipping Information"
        ]
        
        # Create ML pipeline with TF-IDF and Random Forest
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Check if model file exists, load if it does
        if os.path.exists('email_classifier.pkl'):
            self.load_model()
        else:
            # When no model exists, we'll train on first prediction call
            self.is_trained = False
    
    def train(self, texts: List[str], labels: List[str]) -> None:
        """
        Train the classification model
        
        Args:
            texts: List of email texts (masked)
            labels: List of corresponding category labels
        """
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        accuracy = self.model.score(X_test, y_test)
        print(f"Model trained with accuracy: {accuracy:.4f}")
        
        # Save the model
        self.save_model()
        self.is_trained = True
    
    def predict(self, text: str) -> str:
        """
        Classify an email into one of the predefined categories
        
        Args:
            text: Masked email text
            
        Returns:
            Predicted category
        """
        # For demo purposes: if model is not trained, use dummy training data
        if not hasattr(self, 'is_trained') or not self.is_trained:
            self._train_dummy_data()
        
        # Make prediction
        prediction = self.model.predict([text])[0]
        return prediction
    
    def save_model(self) -> None:
        """Save model to disk"""
        with open('email_classifier.pkl', 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self) -> None:
        """Load model from disk"""
        with open('email_classifier.pkl', 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
    
    def _train_dummy_data(self) -> None:
        """Train model on dummy data when no real training data is available"""
        dummy_texts = [
            "I was charged twice for my subscription this month",
            "My account shows the wrong billing information",
            "Can you help me understand the charges on my invoice?",
            "My app keeps crashing when I try to open it",
            "I can't log in to my account, it says incorrect password",
            "The website is very slow to load on my computer",
            "How do I change my email address on my account?",
            "I need to update my shipping address",
            "Can you delete my account and all my data?",
            "What are the specifications of your latest product?",
            "Do you offer discounts for bulk orders?",
            "Is this product compatible with my device?",
            "I want to return the item I bought last week",
            "The product arrived damaged, I need a replacement",
            "What's your return policy?",
            "When will my order be delivered?",
            "Can I change my shipping address for my current order?",
            "My package hasn't arrived yet, can you track it?"
        ]
        
        dummy_labels = [
            "Billing Issues", "Billing Issues", "Billing Issues",
            "Technical Support", "Technical Support", "Technical Support",
            "Account Management", "Account Management", "Account Management",
            "Product Inquiry", "Product Inquiry", "Product Inquiry",
            "Return Request", "Return Request", "Return Request",
            "Shipping Information", "Shipping Information", "Shipping Information"
        ]
        
        self.train(dummy_texts, dummy_labels)