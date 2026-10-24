# WhatsApp Message Notification Router

An AI-powered system that reads incoming WhatsApp messages and decides how each one should be handled for a specific user — notify, digest, or mute — based on message content, user behavior, sender trust, and conversation history.

Built with Python, Groq LLM, Google Gemini vision, and OpenAI Whisper. Achieves 86.7% routing accuracy on labeled test data.

---

## What It Does

WhatsApp is noisy. Family chats, society notices, school updates, business promotions, scams, and personal messages all arrive in the same stream. This system acts as a smart filter:

- **notify** — interrupt the user now (urgent, personal, time-sensitive)
- **digest** — useful but not urgent, show later in a summary
- **mute** — spam, scam, unwanted, or low-value content

Every decision is personalized. The same message can get a different routing for different users depending on their engagement history, preferences, and relationship with the sender.

---

## Features

- Handles text, image, and voice note messages
- Reads image posters and screenshots using Gemini vision
- Transcribes voice notes using OpenAI Whisper
- Joins 9 data sources per message to build personalized context
- Detects scams and phishing with safety-first routing
- Respects user quiet hours (do not disturb windows)
- Uses past message history and user reactions as evidence
- Outputs structured CSV with action, type, reason, confidence, and evidence

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| pandas | Loading and joining 13 CSV files |
| Groq (Qwen 3.8 27B) | Text classification and routing decisions |
| Google Gemini | Image vision — reading posters and screenshots |
| OpenAI Whisper | Voice note transcription |
| python-dotenv | Secure API key management |

---

## Project Structure

ai-message-router/
├── dataset/
│ ├── messages.csv ← 110 messages to route
│ ├── sample_messages.csv ← 30 labeled examples for evaluation
│ ├── users.csv ← user behavior profiles
│ ├── groups.csv ← group metadata
│ ├── group_members.csv ← user roles and mute status per group
│ ├── business_accounts.csv ← business trust signals
│ ├── user_business_history.csv ← user-business relationships
│ ├── message_history.csv ← past messages received
│ ├── message_events.csv ← how user reacted to past messages
│ ├── images.csv ← image file references
│ ├── voice_notes.csv ← voice note file references
│ ├── daily_notification_summary.csv
│ ├── output.csv ← generated predictions
│ └── media/
│ ├── images/
│ └── audio/
├── explore.py ← data exploration
├── context.py ← builds context bundle per message
├── media.py ← image vision + voice transcription
├── classifier.py ← LLM routing with structured JSON output
├── pipeline.py ← runs full pipeline on all messages
├── evaluate.py ← accuracy measurement against labeled data
├── README.md
└── .env ← API keys (not committed)


---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/ai-message-router.git
cd ai-message-router
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install pandas groq google-genai openai-whisper python-dotenv
```

### 4. Create .env file

GROQ_API_KEY=your-groq-key-here
GEMINI_API_KEY=your-gemini-key-here


Get free API keys:
- Groq: https://console.groq.com
- Gemini: https://aistudio.google.com

### 5. (Optional) Install ffmpeg for voice notes
```bash
# Windows
winget install ffmpeg

# Mac
brew install ffmpeg
```

---

## Usage

### Run the full pipeline
```bash
python pipeline.py
```

Processes all 110 messages and writes predictions to `dataset/output.csv`.

### Evaluate accuracy
```bash
python evaluate.py
```

Runs the system on 30 labeled examples and prints accuracy scores.

### Explore the data
```bash
python explore.py
```

Prints shapes and sample rows from all CSV files.

---

## Output Format

Each row in `output.csv` contains:

| Column | Description |
|---|---|
| message_id | Unique message identifier |
| action | notify / digest / mute |
| message_type | personal / urgent / event / payment / business_update / promotion / greeting / forward / spam / scam / unknown |
| reason | One sentence explanation |
| confidence | 0.0 to 1.0 |
| evidence_message_ids | Semicolon-separated past message IDs used as evidence |

---

## How It Works

For every incoming message the system:

1. **Builds context** — joins user profile, group info, business trust signals, and past message history from 9 CSV files
2. **Reads media** — sends images to Gemini vision for description, transcribes voice notes with Whisper
3. **Classifies** — sends the full context to Groq LLM with a structured prompt and gets a JSON routing decision back
4. **Validates** — checks that output values match allowed categories
5. **Saves** — writes the decision to output.csv with evidence

---

## Results

| Metric | Score |
|---|---|
| Action accuracy | 86.7% |
| Message type accuracy | 66.7% |
| Total messages processed | 110 |
| Unknown predictions | 1 / 110 |
| Evaluated against | 30 labeled examples |

Action distribution across 110 messages:
- mute: 49
- notify: 34
- digest: 27

---

## Architecture Decisions

**Why Groq instead of Gemini for text?**
Gemini free tier allows 20 requests per day. Groq allows 14,400. With 110 messages to process, Gemini would take 6 days on the free tier. Groq completes the full dataset in under 5 minutes.

**Why keep Gemini for images?**
Groq does not support vision input. Gemini does. With only 15 image messages in the dataset, 15 Gemini calls fits comfortably within the daily free quota.

**Why not train a custom model?**
Training requires labeled data at scale, GPU compute, and ML expertise. Using an existing LLM with a well-designed prompt achieves strong accuracy without any of these requirements. The real engineering work is context assembly, not model training.

**Why build context from 9 CSVs instead of just using message text?**
Message text alone is not enough for personalized routing. The same promotional message should be notified to a user who regularly buys from that business, but muted for a user who has never interacted with them. Context is what makes routing personalized rather than generic.

---

## Known Limitations

- Voice note transcription requires ffmpeg installed and on PATH
- Gemini image quota is limited to 20 requests per day on free tier
- Groq daily token limit is 200,000 tokens — sufficient for one full run
- 30 labeled examples is a small evaluation set; accuracy may vary on larger samples

---

## Future Improvements

- Add ffmpeg auto-detection with clear setup instructions
- Implement retry logic with exponential backoff for rate limits
- Cache image descriptions to avoid re-processing the same image twice
- Expand labeled evaluation set beyond 30 examples
- Add confidence calibration analysis
- Build a simple web interface for testing individual messages

---

## License
