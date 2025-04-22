import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import spacy # type: ignore
from typing import Dict, List, Tuple, Any

def preprocess_text(text):
    """Clean and preprocess text data"""
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            stop_words = set(stopwords.words('english'))
        
        tokens = [word for word in tokens if word not in stop_words]
        
        # Stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        
        return ' '.join(tokens)
    return ""

def load_dataset(file_path):
    """Load and preprocess the dataset"""
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Check if expected columns exist
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns")
    
    # Preprocess text
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Map labels to binary format (assuming 'spam' and 'ham' or similar labels)
    if df['label'].dtype == 'object':
        label_mapping = {'spam': 1, 'ham': 0}
        df['label'] = df['label'].map(lambda x: label_mapping.get(x.lower(), 0) if isinstance(x, str) else x)
    
    return df



# Load SpaCy NER model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # Download if not available
    import spacy.cli # type: ignore
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class PIIMasker:
    """Class for masking PII/PCI data in text"""
    
    def __init__(self):
        # Regex patterns for different PII types
        self.patterns = {
            "full_name": r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            "email": r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',
            "phone_number": r'\b(?:\+\d{1,3}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
            "dob": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            "aadhar_num": r'\b\d{4}\s?\d{4}\s?\d{4}\b',
            "credit_debit_no": r'\b(?:\d{4}[- ]?){4}\b',
            "cvv_no": r'\bCVV:?\s*\d{3,4}\b',
            "expiry_no": r'\b(?:0[1-9]|1[0-2])[/-]\d{2,4}\b'
        }
        
    def mask_pii(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Mask PII in text and return masked text with entity metadata
        
        Args:
            text: Input text containing potential PII
            
        Returns:
            Tuple containing:
                - Masked text
                - List of entity dictionaries with position, classification, and original value
        """
        entities = []
        masked_text = text
        
        # First use SpaCy for name detection (more accurate than regex for names)
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                start_idx = ent.start_char
                end_idx = ent.end_char
                entity_value = text[start_idx:end_idx]
                entities.append({
                    "position": [start_idx, end_idx],
                    "classification": "full_name",
                    "entity": entity_value
                })
        
        # Then use regex patterns for other entities
        for entity_type, pattern in self.patterns.items():
            if entity_type == "full_name":  # Skip as we handled with SpaCy
                continue
            
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start_idx = match.start()
                end_idx = match.end()
                entity_value = text[start_idx:end_idx]
                
                # Check if this position overlaps with any existing entity
                overlap = False
                for entity in entities:
                    if (start_idx < entity["position"][1] and end_idx > entity["position"][0]):
                        overlap = True
                        break
                
                if not overlap:
                    entities.append({
                        "position": [start_idx, end_idx],
                        "classification": entity_type,
                        "entity": entity_value
                    })
        
        # Sort entities by start position (descending) to avoid index issues when replacing
        entities.sort(key=lambda x: x["position"][0], reverse=True)
        
        # Replace entities with masks
        for entity in entities:
            start, end = entity["position"]
            entity_type = entity["classification"]
            masked_text = masked_text[:start] + f"[{entity_type}]" + masked_text[end:]
        
        return masked_text, entities