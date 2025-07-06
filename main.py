# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from faster_whisper import WhisperModel
# import os
# from uuid import uuid4

# # Initialize FastAPI app
# app = FastAPI()

# # Allow frontend access
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Load the model once at startup
# model = WhisperModel("small", compute_type="int8", device="cpu")
#   # use 'small' or 'medium' for better quality

# @app.post("/transcribe")
# async def transcribe_audio(audio: UploadFile = File(...)):
#     # Save uploaded audio file
#     file_id = uuid4().hex
#     ext = os.path.splitext(audio.filename)[1]
#     path = os.path.join(UPLOAD_FOLDER, f"{file_id}{ext}")

#     with open(path, "wb") as f:
#         f.write(await audio.read())

#     # Transcribe the audio
#     segments, _ = model.transcribe(path, beam_size=5)

#     # Format result
#     formatted = []
#     for seg in segments:
#         formatted.append({
#             "start": round(seg.start, 2),
#             "end": round(seg.end, 2),
#             "text": seg.text.strip()
#         })

#     # Clean up
#     os.remove(path)

#     return {"segments": formatted}
# #uvicorn main:app --reload



























from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from pydub import AudioSegment
from uuid import uuid4
import os

# Initialize FastAPI app
app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once at startup (use "medium" or "large" for better accuracy)
model = WhisperModel("medium", compute_type="int8", device="cpu")

CHUNK_DURATION_MS = 30 * 1000  # 30 seconds


def transcribe_with_chunks(audio_path: str):
    audio = AudioSegment.from_file(audio_path)
    total_duration = len(audio)  # in ms

    all_segments = []
    offset = 0  # in seconds

    for start in range(0, total_duration, CHUNK_DURATION_MS):
        end = min(start + CHUNK_DURATION_MS, total_duration)
        chunk = audio[start:end]

        chunk_filename = f"chunk_{uuid4().hex}.wav"
        chunk.export(chunk_filename, format="wav")

        try:
            segments, _ = model.transcribe(
                chunk_filename,
                beam_size=5,
                vad_filter=True,
                language="en"  # 🚨 Force English
            )
            for seg in segments:
                all_segments.append({
                    "start": round(seg.start + offset, 2),
                    "end": round(seg.end + offset, 2),
                    "text": seg.text.strip()
                })
        finally:
            os.remove(chunk_filename)

        offset += (end - start) / 1000

    return all_segments



@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    # Save uploaded audio file
    file_id = uuid4().hex
    ext = os.path.splitext(audio.filename)[1]
    path = os.path.join(UPLOAD_FOLDER, f"{file_id}{ext}")

    with open(path, "wb") as f:
        f.write(await audio.read())

    try:
        result = transcribe_with_chunks(path)
        return {"segments": result}
    except Exception as e:
        return {"error": str(e)}
    finally:
        os.remove(path)
