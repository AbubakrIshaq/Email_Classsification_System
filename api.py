from fastapi import APIRouter, Body # type: ignore
from pydantic import BaseModel
from typing import List, Dict, Any
from utils import PIIMasker
from models import EmailClassifier

# Initialize router
router = APIRouter()

# Initialize PII masker and classifier
masker = PIIMasker()
classifier = EmailClassifier()

class EmailInput(BaseModel):
    """Pydantic model for email input"""
    email_body: str

class Entity(BaseModel):
    """Pydantic model for entity information"""
    position: List[int]
    classification: str
    entity: str

class EmailOutput(BaseModel):
    """Pydantic model for API response"""
    input_email_body: str
    list_of_masked_entities: List[Entity]
    masked_email: str
    category_of_the_email: str

@router.post("/classify-email", response_model=EmailOutput)
async def classify_email(email: EmailInput = Body(...)):
    """
    Classify an email and mask PII
    
    Args:
        email: Email body text
        
    Returns:
        Processed email with masked PII and classification
    """
    # Mask PII in the email
    masked_email, entities = masker.mask_pii(email.email_body)
    
    # Classify the masked email
    category = classifier.predict(masked_email)
    
    # Return the result
    return EmailOutput(
        input_email_body=email.email_body,
        list_of_masked_entities=entities,
        masked_email=masked_email,
        category_of_the_email=category
    )