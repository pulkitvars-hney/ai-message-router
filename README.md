markdown
# WhatsApp Message Notification Router

AI-powered system that routes WhatsApp messages into notify/digest/mute decisions.

## Setup

1. Install Python 3.10+
2. Create virtual environment:

python -m venv venv
venv\Scripts\activate


3. Install dependencies:

pip install pandas groq google-genai openai-whisper python-dotenv


4. Create .env file:

GROQ_API_KEY=your-groq-key
GEMINI_API_KEY=your-gemini-key


## Run

python pipeline.py


## Evaluate

python evaluate.py


## Architecture

- context.py — builds context bundle from all CSVs per message
- media.py — image description via Gemini vision + voice via Whisper
- classifier.py — sends context to Groq LLM, returns routing decision
- pipeline.py — runs full pipeline on all 110 messages
- evaluate.py — scores predictions against sample_messages.csv

## Tech Stack

- Groq (Qwen 3.8 27B) — text classification
- Google Gemini — image vision
- OpenAI Whisper — voice transcription
- pandas — data loading and joining
- Python 3.10+

## Accuracy

- Action accuracy: 83.3%
- Message type accuracy: 70%+
- Evaluated against 30 labeled sample messages