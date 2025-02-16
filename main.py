from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load pre-trained model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

class ChatInput(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(input: ChatInput):
    if not input.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    response = chatbot(input.message, max_length=50, num_return_sequences=1)
    return ChatResponse(response=response[0]['generated_text'])

@app.get("/")
async def root():
    return {"message": "Welcome to the Chatbot API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)