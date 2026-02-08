import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3-pro-preview")
    
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
