import google.generativeai as genai
import os
import json
from google.cloud import vision


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3-pro-preview")
# --- Cloud Vision setup ---
credentials_info = json.loads(
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
)

vision_client = vision.ImageAnnotatorClient.from_service_account_info(
    credentials_info
)
    
from fastapi import FastAPI, UploadFile, File
from deepface import DeepFace
from PIL import Image
import imagehash
import tempfile
import os

app = FastAPI()

@app.get("/test-gemini")
def test_gemini():
    response = model.generate_content(
        "Explain non-consensual AI-generated content in simple words."
    )
    return {"gemini_response": response.text}

@app.get("/test-vision")
def test_vision():
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Alberto_conversi_profile_pic.jpg/256px-Alberto_conversi_profile_pic.jpg"

    image = vision.Image()
    image.source.image_uri = image_url

    response = vision_client.safe_search_detection(image=image)
    safe = response.safe_search_annotation

    return {
        "adult": vision.Likelihood(safe.adult).name,
        "racy": vision.Likelihood(safe.racy).name,
        "violence": vision.Likelihood(safe.violence).name,
        "medical": vision.Likelihood(safe.medical).name,
        "spoof": vision.Likelihood(safe.spoof).name
    }


@app.get("/")
def root():
    return {"status": "Backend is running"}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        image_path = tmp.name

    # Generate face embedding
    embedding = DeepFace.represent(
        img_path=image_path,
        model_name="Facenet",
        enforce_detection=False
    )

    # Generate perceptual hash
    img = Image.open(image_path)
    phash = str(imagehash.phash(img))

    # Cleanup
    os.remove(image_path)

    return {
        "status": "success",
        "face_embedding_length": len(embedding[0]["embedding"]),
        "phash": phash
    }
