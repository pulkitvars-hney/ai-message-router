from dotenv import load_dotenv
from google import genai
from pathlib import Path
import base64
import os

load_dotenv()

gemini_client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))
GEMINI_MODEL = 'gemini-3.8-flash'

# Load Whisper once globally
_whisper_model = None

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        print("   Loading Whisper model (one time)...")
        import whisper
        _whisper_model = whisper.load_model("base")
    return _whisper_model

def describe_image(file_path: str) -> str:
    full_path = Path('dataset') / file_path
    if not full_path.exists():
        return f"Image file not found: {file_path}"
    try:
        with open(full_path, 'rb') as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        ext = full_path.suffix.lower()
        mime_map = {'.jpg':'image/jpeg','.jpeg':'image/jpeg','.png':'image/png','.webp':'image/webp'}
        mime_type = mime_map.get(ext, 'image/jpeg')
        prompt = """Analyze this WhatsApp message image in 2-3 sentences.
State: what type of content it is (sale poster, event flyer, payment QR, scam, screenshot etc),
what the main message or offer is, and any suspicious elements, amounts, or urgency signals."""
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[{"parts": [
                {"inline_data": {"mime_type": mime_type, "data": image_b64}},
                {"text": prompt}
            ]}]
        )
        return response.text
    except Exception as e:
        err = str(e)
        if '429' in err:
            return "Image quota exceeded — classify based on text context only"
        return f"Image read error: {err[:100]}"

def transcribe_voice(file_path: str) -> str:
    full_path = Path('dataset') / file_path
    if not full_path.exists():
        return f"Audio file not found: {file_path}"
    try:
        model = get_whisper()
        result = model.transcribe(str(full_path))
        transcript = result['text'].strip()
        if not transcript:
            return "Voice note transcription was empty"
        return transcript
    except Exception as e:
        err = str(e)
        if 'WinError 2' in err or 'ffmpeg' in err.lower():
            return "Voice note: ffmpeg not available — classify based on context only"
        return f"Transcription error: {err[:100]}"

def get_media_description(media_type: str, media_path: str) -> str:
    if not media_path or media_path == '':
        return ''
    if media_type == 'image':
        print(f"   Reading image: {media_path}")
        return describe_image(media_path)
    elif media_type == 'voice':
        print(f"   Transcribing voice: {media_path}")
        return transcribe_voice(media_path)
    return ''