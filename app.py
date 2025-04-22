from fastapi import FastAPI, Request # type: ignore
from fastapi.responses import HTMLResponse # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
from fastapi.templating import Jinja2Templates # type: ignore
import uvicorn # type: ignore
from api import router as api_router

# Create FastAPI app
app = FastAPI(
    title="Email Classification API",
    description="API for classifying support emails and masking PII",
    version="1.0.0"
)

# Include API router
app.include_router(api_router)

# Simple UI for testing
@app.get("/", response_class=HTMLResponse)
async def root():
    """Render a simple HTML UI for testing the API"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Email Classification API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            textarea { width: 100%; height: 200px; margin: 10px 0; padding: 8px; }
            button { padding: 10px 15px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
            pre { background-color: #f5f5f5; padding: 15px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <h1>Email Classification API</h1>
        <p>Enter an email to classify and mask personal information:</p>
        <textarea id="emailInput" placeholder="Type your email here..."></textarea>
        <button onclick="classifyEmail()">Classify Email</button>
        <h3>Result:</h3>
        <pre id="result">Results will appear here...</pre>
        
        <script>
            async function classifyEmail() {
                const emailText = document.getElementById('emailInput').value;
                const resultElement = document.getElementById('result');
                
                try {
                    const response = await fetch('/classify-email', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ email_body: emailText }),
                    });
                    
                    const data = await response.json();
                    resultElement.textContent = JSON.stringify(data, null, 2);
                } catch (error) {
                    resultElement.textContent = `Error: ${error.message}`;
                }
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)