from urllib import response
from xmlrpc import client
from fastapi import FastAPI,UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from textblob import TextBlob
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from haversian import haversine
from transformers import pipeline
from langchain_core.messages import HumanMessage
import pandas as pd
import google.generativeai as genai
import torch
from PIL import Image
from huggingface_hub import InferenceClient
import base64
import httpx
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer
import requests
import platform
API_TOKEN = "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"
import io
import json
import os

# Set ffmpeg path based on OS
if platform.system() == "Windows":
    AudioSegment.converter = r"C:\ffmpeg\ffmpeg-files\bin\ffmpeg.exe"
else:
    AudioSegment.converter = "ffmpeg"

# Load vosk model
modelvoice = Model("./vosk-model-small-en-us-0.15")
app = FastAPI()
genai.configure(api_key="AIzaSyBBx_cccbAKlqhW6ZNeieRgxnkomsDUv9Q")

def detect_language(text: str) -> str:
    try:
        blob = TextBlob(text)
        return blob.detect_language()
    except Exception:
        return "unknown"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://my-frontend-visionx.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Request schema ----
class ChatRequest(BaseModel):
    data: str

# ---- LLM ----
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-235B-A22B-Instruct-2507",
    huggingfacehub_api_token="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",
    task="text-generation",
    temperature=0.3,
    max_new_tokens=500
)

model = ChatHuggingFace(llm=llm)
model2 = genai.GenerativeModel("gemini-2.0-flash")

@app.post("/api/chatbot")
async def create_item(request: ChatRequest):
    input = request.data
    lang = detect_language(request.data)
    prompt = f"""
You are a medical assistant .
Predict best and ONLY one possible  diagnosis  and Provide treatment plan on the diagnosis for the following symptoms {input} and also
keep the writing clean with spaces and professional."""
    response = model.invoke(prompt)
    blob=TextBlob(response.content)
    return {
        "reply": str(blob.translate(to=lang)) if lang != 'en' and lang != 'unknown' else response.content
    }

################################
@app.post("/api/voicebot")
async def create_voice_item(file: UploadFile = File(...)):
     audio_bytes = await file.read()
     ext = file.filename.split(".")[-1].lower()
     if ext in ["mp3", "wav", "m4a", "ogg", "flac", "aac"]:
        audio_format = ext
     else:
        audio_format = "wav"  # fallback

    
     
     audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)
              

   
     audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)

    # Initialize recognizer
     rec = KaldiRecognizer(modelvoice, 16000)

    # Feed entire audio to recognizer in chunks of 4000ms
     chunk_length_ms = 4000
     for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i+chunk_length_ms]
        rec.AcceptWaveform(chunk.raw_data)

     result = json.loads(rec.FinalResult())
    
     input=result["text"]
     print(input)
     lang = detect_language(input)
     prompt = f"""
You are a medical assistant.
Predict best and ONLY one possible  diagnosis  and Provide treatment plan on the diagnosis for the following symptoms{input} and also
keep the writing clean with spaces and professional."""
     response = model.invoke(prompt)
     
     blob=TextBlob(response.content)
     return {
        "reply": str(blob.translate(to=lang)) if lang != 'en' and lang != 'unknown' else response.content
     }

class LocationRequest(BaseModel):
    latitude: float
    longitude: float

       
#############################################
@app.post("/api/nearest-hospital")
async def nearest_hospital(request: LocationRequest):
    latitude = request.latitude
    longitude = request.longitude
    df=pd.read_csv("C:\\Users\\rishe\\Downloads\\pynext\\my-app\\backend\\Hospitals In India (Anonymized).csv")
    df["distance_km"]=0.0
    df["distance_km"] = df.apply(
        lambda row: haversine(latitude, longitude, row["Latitude"], row["Longitude"]),
        axis=1
    )

    return df.sort_values("distance_km").head(5).to_dict(orient="records")

      
@app.post("/api/image-analysis")
async def image_analysis(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image=Image.open(io.BytesIO(image_bytes)).convert("RGB")
    response = model2.generate_content([
    "Analyze this medical image and list abnormalities and its treatment options.",
    image
])
    print(response.text)
    return {"reply":response.text}

    