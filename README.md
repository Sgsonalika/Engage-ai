# ⚡ EngageAI — AI-Powered Social Media Content Analyzer

Analyze social media posts from screenshots, images, or PDFs and receive AI-generated engagement insights, content quality metrics, and actionable optimization recommendations.

EngageAI combines OCR, NLP, rule-based analytics, and Google Gemini to help creators, marketers, freelancers, and businesses improve the performance of their content.

---

## Features

### Content Extraction

* Upload screenshots, images, or PDFs
* OCR-powered text extraction using Tesseract
* PDF text extraction using PyMuPDF
* Automatic preprocessing for improved OCR accuracy

### Content Analytics

Automatically extracts:

* Word count
* Average sentence length
* Sentiment analysis
* Readability classification
* Hashtag usage
* Emoji usage
* Call-to-action (CTA) detection

### AI-Powered Insights

Using Google Gemini:

* Engagement score (0–100)
* Discoverability recommendations
* Readability improvements
* Interaction optimization suggestions
* Tone and messaging feedback

### Flexible Usage

* FastAPI web interface
* Command-line interface (CLI)
* JSON-based AI responses
* Async processing for lower latency

---

## Tech Stack

| Layer          | Technology                 |
| -------------- | -------------------------- |
| Backend        | Python                     |
| API            | FastAPI                    |
| OCR            | Tesseract OCR              |
| PDF Processing | PyMuPDF                    |
| AI Model       | Gemini 2.5 Flash           |
| NLP            | Custom rule-based analysis |
| Async Requests | HTTPX + asyncio            |
| Testing        | Pytest                     |

---

## Project Architecture

```text
User Upload
     │
     ▼
File Validation
     │
     ▼
Text Extraction
 ├─ PDF → PyMuPDF
 └─ Image → OCR
     │
     ▼
Content Analysis
 ├─ Sentiment
 ├─ Readability
 ├─ Hashtags
 ├─ Emojis
 └─ CTA Detection
     │
     ▼
Parallel AI Processing
 ├─ Engagement Scoring
 └─ Improvement Suggestions
     │
     ▼
Result Generation
     │
     ▼
Web UI / CLI Output
```

---

## Project Structure

```text
engage_ai/
├── main.py
├── requirements.txt
├── .env.example
│
├── sample_files/
│   └── sample_post.txt
│
├── tests/
│   └── test_engage_ai.py
│
└── modules/
    ├── api.py
    ├── extractor.py
    ├── analyzer.py
    ├── scorer.py
    ├── suggestions.py
    └── display.py
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/engage_ai.git
cd engage_ai
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate:

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Gemini API

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

Get an API key from Google AI Studio.

---

## Run the Application

### Web Interface

```bash
python main.py
```

Open:

```text
http://127.0.0.1:8000
```

### CLI Mode

```bash
python main.py --file sample_post.png
```

or

```bash
python main.py --file sample_post.pdf
```

---

## Example Output

```json
{
  "score": 82,
  "discoverability": [
    "Add niche hashtags",
    "Place keywords earlier"
  ],
  "interaction": [
    "Ask a direct question",
    "Encourage comments"
  ]
}
```

---

## Future Improvements

* Instagram-specific scoring model
* OCR cleanup pipeline
* Post category classification
* Content benchmarking
* Multi-image carousel analysis
* Historical performance tracking
* Dashboard with analytics visualizations
* AI-generated caption rewriting

---

## Testing

```bash
pytest tests/ -v
```

---

## Why This Project?

Social media content is often evaluated manually.

EngageAI automates the process by combining:

* OCR
* NLP
* AI reasoning
* Content analytics

to provide fast, actionable feedback for creators and marketers.

---

## License

MIT License
