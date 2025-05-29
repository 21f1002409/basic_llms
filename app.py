from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from google import genai
from google.genai import types
import base64
import os
import requests
from typing import List, Optional
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="API Service", description="Configured APIs")

# Initialize Gemini client
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

# Sarvam AI API key
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
if not SARVAM_API_KEY:
    raise ValueError("SARVAM_API_KEY environment variable not set")

client = genai.Client(api_key=GOOGLE_API_KEY)

# Pydantic models for request/response
class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: str

class ImageRequest(BaseModel):
    image_base64: str

class ImageResponse(BaseModel):
    transcription: str

class TTSRequest(BaseModel):
    text: str
    target_language_code: str = "en-IN"  # Default to English India
    speaker: Optional[str] = "Meera"  # Default speaker
    pitch: Optional[float] = 0.0  # Range: -0.75 to 0.75
    pace: Optional[float] = 1.0  # Range: 0.3 to 3.0
    loudness: Optional[float] = 1.0  # Range: 0.1 to 3.0
    speech_sample_rate: Optional[int] = 22050  # 8000, 16000, 22050, 24000
    enable_preprocessing: Optional[bool] = False
    model: Optional[str] = "bulbul:v1"  # bulbul:v1 or bulbul:v2

class TTSResponse(BaseModel):
    audio_base64: str
    request_id: str

class EmbeddingRequest(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    embeddings: List[float]
    dimensions: int

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[dict]] = []  # [{"role": "user/assistant", "content": "message"}]

class ChatResponse(BaseModel):
    response: str
    conversation_history: List[dict]

# Endpoint 1: Sentiment Analysis
@app.post("/emo", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """
    Analyze sentiment of text and return positive or negative.
    """
    try:
        prompt = f"""
        Analyze the sentiment of the following text and respond with only "positive" or "negative":
        
        Text: "{request.text}"
        
        Response format: Just return "positive" or "negative" with a brief confidence level.
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-05-20",
            contents=prompt
        )
        
        result_text = response.text.lower().strip()
        
        # Extract sentiment and confidence
        if "positive" in result_text:
            sentiment = "positive"
        elif "negative" in result_text:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        return SentimentResponse(
            sentiment=sentiment,
            confidence=result_text
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")

# Endpoint 2: Image Transcription
@app.post("/img", response_model=ImageResponse)
async def transcribe_image(request: ImageRequest):
    """
    Transcribe/describe image content from base64 encoded image.
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        
        # Create image part for Gemini
        image_part = types.Part.from_bytes(
            data=image_data, 
            mime_type="image/jpeg"  # Adjust mime_type based on your image format
        )
        
        # Generate content with image
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-05-20",
            contents=[
                "Please transcribe any text you see in this image. If there's no text, describe what you see in the image in detail.",
                image_part
            ]
        )
        
        return ImageResponse(transcription=response.text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Endpoint 3: Text-to-Speech using Sarvam AI
@app.post("/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech using Sarvam AI and return base64 encoded audio.
    Supports multiple Indian languages and voice customization.
    """
    try:
        # Prepare headers
        headers = {
            "api-subscription-key": SARVAM_API_KEY,
            "Content-Type": "application/json"
        }
        
        # Prepare payload
        payload = {
            "text": request.text,
            "target_language_code": request.target_language_code,
            "speaker": request.speaker,
            "pitch": request.pitch,
            "pace": request.pace,
            "loudness": request.loudness,
            "speech_sample_rate": request.speech_sample_rate,
            "enable_preprocessing": request.enable_preprocessing,
            "model": request.model
        }
        
        # Make request to Sarvam AI
        response = requests.post(
            "https://api.sarvam.ai/text-to-speech",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Sarvam AI API error: {response.text}"
            )
        
        result = response.json()
        
        return TTSResponse(
            audio_base64=result["audios"][0],
            request_id=result.get("request_id", "")
        )
        
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling Sarvam AI API: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

# Endpoint for TTS with audio file response
@app.post("/tts/audio")
async def text_to_speech_audio(request: TTSRequest):
    """
    Convert text to speech and return audio file directly.
    """
    try:
        # Get TTS response
        tts_response = await text_to_speech(request)
        
        # Decode base64 audio
        audio_data = base64.b64decode(tts_response.audio_base64)
        
        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating audio file: {str(e)}")

# Endpoint 4: Text Embeddings
@app.post("/emb", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    """
    Generate embeddings for the input text using Gemini embedding model.
    """
    try:
        result = client.models.embed_content(
            model="gemini-embedding-exp-03-07",
            contents=request.text
        )
        
        embeddings = result.embeddings[0].values if result.embeddings else []
        
        return EmbeddingResponse(
            embeddings=embeddings,
            dimensions=len(embeddings)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

# Endpoint 5: Chat/Text-to-Text
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for text-to-text conversations using Gemini.
    Maintains conversation history for context.
    """
    try:
        # Build conversation context
        conversation_context = ""
        
        # Add previous conversation history
        for msg in request.conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                conversation_context += f"User: {content}\n"
            elif role == "assistant":
                conversation_context += f"Assistant: {content}\n"
        
        # Add current message
        conversation_context += f"User: {request.message}\nAssistant: "
        
        # Generate response using Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=conversation_context
        )
        
        assistant_response = response.text.strip()
        
        # Update conversation history
        updated_history = request.conversation_history.copy()
        updated_history.append({"role": "user", "content": request.message})
        updated_history.append({"role": "assistant", "content": assistant_response})
        
        return ChatResponse(
            response=assistant_response,
            conversation_history=updated_history
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating chat response: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "API Service"}

# Supported languages endpoint
@app.get("/languages")
async def get_supported_languages():
    """Get supported languages for TTS"""
    return {
        "supported_languages": [
            {"code": "bn-IN", "name": "Bengali (India)"},
            {"code": "en-IN", "name": "English (India)"},
            {"code": "gu-IN", "name": "Gujarati (India)"},
            {"code": "hi-IN", "name": "Hindi (India)"},
            {"code": "kn-IN", "name": "Kannada (India)"},
            {"code": "ml-IN", "name": "Malayalam (India)"},
            {"code": "mr-IN", "name": "Marathi (India)"},
            {"code": "od-IN", "name": "Odia (India)"},
            {"code": "pa-IN", "name": "Punjabi (India)"},
            {"code": "ta-IN", "name": "Tamil (India)"},
            {"code": "te-IN", "name": "Telugu (India)"}
        ],
        "speakers": {
            "bulbul:v1": {
                "female": ["Diya", "Maya", "Meera", "Pavithra", "Maitreyi", "Misha"],
                "male": ["Amol", "Arjun", "Amartya", "Arvind", "Neel", "Vian"]
            },
            "bulbul:v2": {
                "female": ["Anushka", "Manisha", "Vidya", "Arya"],
                "male": ["Abhilash", "Karun", "Hitesh"]
            }
        }
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "API Service",
        "endpoints": {
            "/emo": "POST - Sentiment analysis",
            "/img": "POST - Image transcription", 
            "/tts": "POST - Text to speech (returns base64 audio)",
            "/tts/audio": "POST - Text to speech (returns audio file)",
            "/emb": "POST - Text embeddings",
            "/chat": "POST - Text-to-text chat",
            "/health": "GET - Health check",
            "/languages": "GET - Supported languages and speakers"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
