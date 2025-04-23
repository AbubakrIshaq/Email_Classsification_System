# Email Classification System
This system classifies support emails into predefined categories while masking personally identifiable information (PII) before processing.

## Features

+ Email classification into support categories (Billing, Technical Support, etc.)
+ PII/PCI data masking using regex and NER (without LLMs)
+ FastAPI deployment for easy integration
+ Hugging Face Spaces deployment

## Setup Instructions
## Local Development

__1. Clone the repository:__
git clone https://github.com/yourusername/Email_Classification_System.git
cd email-classification

__2. Create a virtual environment and install dependencies:__
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

__3. Download the SpaCy model:__
python -m spacy download en_core_web_sm

__4. Run the application:__
uvicorn app:app --reload

__5. Access the API at http:__//localhost:8000

## Hugging Face Spaces Deployment

Fork this repository to your GitHub account
2.Create a new Hugging Face Space:

+ Go to https://huggingface.co/spaces
+ Click "Create new Space"
+ Select the "FastAPI" SDK
+ Connect your GitHub repository

3.In your Space settings, ensure you have:

+ Python 3.9+ selected
+ Add any required secrets if needed

4.Deploy your Space and wait for the build to complete

## API Usage
__Request Format__

POST /classify-email
Content-Type: application/json

{

  "email_body": "Your email text here..."

}

__Response Format__

json
{

  "input_email_body": "string containing the email",
  "list_of_masked_entities":
  
  [
  
    {
    
      "position": [start_index, end_index],
      "classification": "entity_type",
      "entity": "original_entity_value"
   
    }
  
  ],
  
  "masked_email": "string containing the masked email",
  "category_of_the_email": "string containing the class"

}

## Project Structure

email-classification/

├── app.py              # Main FastAPI application

├── models.py           # Classification model implementation

├── utils.py            # PII masking and utility functions

├── api.py              # API endpoints

├── requirements.txt    # Dependencies

└── README.md           # Setup instructions

## Implementation Details
__PII Masking__
The system detects and masks various types of PII:

+ Full names
+ Email addresses
+ Phone numbers
+ Dates of birth
+ Aadhar card numbers
+ Credit/debit card numbers
+ CVV numbers
+ Card expiry dates

## Email Classification
The classification system uses a machine learning pipeline with:

+ TF-IDF Vectorization for feature extraction
+ Random Forest classifier for prediction

The system includes a dummy training dataset for demonstration purposes.






