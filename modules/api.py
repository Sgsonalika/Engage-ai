"""
modules/api.py
---------------
FastAPI application exposing:
  POST /api/analyze   – accepts a multipart file upload, runs the full pipeline
  GET  /              – serves the single-page HTML frontend
  GET  /health        – simple health check
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from modules.extractor import extract_text, ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES
from modules.analyzer import analyze_characteristics
from modules.scorer import generate_engagement_score
from modules.suggestions import provide_improvement_suggestions

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="EngageAI",
    description="Social media content analyzer powered by Google Gemini",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Simple liveness check."""
    return {"status": "ok", "service": "EngageAI"}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Full analysis pipeline:
      1. Validate & save the uploaded file
      2. Extract text (PDF → PyMuPDF, image → Tesseract OCR)
      3. Analyze content characteristics (rule-based)
      4. Generate engagement score + improvement suggestions (Gemini AI, parallel)
      5. Return combined JSON result
    """
    # --- Validate file type ---
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # --- Read & validate file size ---
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        size_mb = len(contents) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum allowed size is 10 MB.",
        )

    # --- Save to a temp file so extraction libraries can open it ---
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Step 1 – Extract text
        extracted_text, error = extract_text(tmp_path)
        if error:
            raise HTTPException(status_code=422, detail=error)

        # Step 2 – Analyze characteristics (fast, synchronous)
        characteristics = analyze_characteristics(extracted_text)

        # Step 3 – Score + Suggestions in parallel (both are async API calls)
        score_result, suggestions_result = await asyncio.gather(
            generate_engagement_score(extracted_text, characteristics),
            provide_improvement_suggestions(extracted_text, characteristics),
        )

    except HTTPException:
        raise
    except RuntimeError as exc:
        # Catches API key missing, Gemini errors, etc.
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")
    finally:
        # Always clean up the temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # --- Build response ---
    ch = characteristics
    return JSONResponse({
        "extracted_text": extracted_text,
        "characteristics": {
            "word_count": ch.word_count,
            "average_sentence_length": ch.average_sentence_length,
            "sentiment": ch.sentiment,
            "hashtag_usage": {
                "count": ch.hashtag_usage.count,
                "hashtags": ch.hashtag_usage.hashtags,
            },
            "emoji_usage": {
                "count": ch.emoji_usage.count,
                "emojis": ch.emoji_usage.emojis,
            },
            "call_to_action": {
                "detected": ch.call_to_action.detected,
                "cta_text": ch.call_to_action.cta_text,
            },
            "readability_level": ch.readability_level,
        },
        "score": score_result["score"],
        "suggestions": suggestions_result,
    })


# ---------------------------------------------------------------------------
# Frontend – served as a single HTML page
# ---------------------------------------------------------------------------

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>EngageAI – Content Analyzer</title>
  <style>
    :root {
      --blue: #4285F4;
      --green: #34A853;
      --red: #EA4335;
      --gray-bg: #F5F5F5;
      --card-bg: #FFFFFF;
      --text: #212121;
      --muted: #666;
      --radius: 12px;
      --shadow: 0 2px 12px rgba(0,0,0,0.08);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', system-ui, sans-serif;
      background: var(--gray-bg);
      color: var(--text);
      min-height: 100vh;
    }
    header {
      background: var(--blue);
      color: #fff;
      padding: 18px 32px;
      display: flex;
      align-items: center;
      gap: 12px;
    }
    header h1 { font-size: 1.4rem; font-weight: 700; }
    header span { font-size: 0.95rem; opacity: 0.85; }
    main { max-width: 960px; margin: 0 auto; padding: 32px 16px; }
    .hero { text-align: center; margin-bottom: 32px; }
    .hero h2 { font-size: 2rem; font-weight: 800; color: var(--blue); }
    .hero p { color: var(--muted); margin-top: 8px; }

    /* Upload area */
    .upload-area {
      border: 2px dashed var(--blue);
      border-radius: var(--radius);
      padding: 40px;
      text-align: center;
      background: var(--card-bg);
      cursor: pointer;
      transition: background 0.2s;
    }
    .upload-area:hover { background: #EBF1FD; }
    .upload-area input { display: none; }
    .upload-icon { font-size: 2.5rem; }
    .upload-area p { color: var(--muted); margin-top: 8px; font-size: 0.9rem; }
    .upload-area .file-name { margin-top: 10px; font-weight: 600; color: var(--blue); }
    .btn {
      display: inline-block;
      margin-top: 20px;
      padding: 12px 32px;
      background: var(--blue);
      color: #fff;
      border: none;
      border-radius: 8px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.2s;
    }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .btn:hover:not(:disabled) { opacity: 0.9; }
    .error-msg { color: var(--red); margin-top: 12px; font-size: 0.9rem; }

    /* Spinner */
    .spinner-wrap { text-align: center; padding: 40px; display: none; }
    .spinner {
      width: 48px; height: 48px;
      border: 5px solid #ddd;
      border-top-color: var(--blue);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      margin: 0 auto 16px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* Results grid */
    #results { display: none; margin-top: 32px; }
    .score-card {
      background: var(--blue);
      color: #fff;
      border-radius: var(--radius);
      padding: 28px 32px;
      text-align: center;
      box-shadow: var(--shadow);
      margin-bottom: 24px;
    }
    .score-num { font-size: 4rem; font-weight: 900; }
    .score-label { font-size: 1rem; opacity: 0.85; margin-top: 4px; }

    .cards-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }
    .metric-card {
      background: var(--card-bg);
      border-radius: var(--radius);
      padding: 20px;
      box-shadow: var(--shadow);
    }
    .metric-card .label { font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .value { font-size: 1.5rem; font-weight: 700; margin-top: 4px; }
    .metric-card .sub { font-size: 0.82rem; color: var(--muted); margin-top: 4px; }
    .tag { display: inline-block; background: #EBF1FD; color: var(--blue); border-radius: 4px; padding: 2px 8px; font-size: 0.8rem; margin: 2px; }

    .card {
      background: var(--card-bg);
      border-radius: var(--radius);
      padding: 24px;
      box-shadow: var(--shadow);
      margin-bottom: 24px;
    }
    .card h3 { font-size: 1rem; font-weight: 700; margin-bottom: 14px; display: flex; align-items: center; gap: 8px; }
    .card textarea {
      width: 100%;
      height: 140px;
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 12px;
      font-size: 0.88rem;
      resize: vertical;
      color: var(--text);
      font-family: inherit;
    }

    /* Suggestions */
    .suggestions-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }
    .sug-group { background: var(--card-bg); border-radius: var(--radius); padding: 18px; box-shadow: var(--shadow); }
    .sug-group h4 { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--blue); margin-bottom: 10px; }
    .sug-group ul { list-style: none; }
    .sug-group li { font-size: 0.88rem; color: var(--text); padding: 6px 0; border-bottom: 1px solid #f0f0f0; }
    .sug-group li:last-child { border-bottom: none; }
    .sug-group li::before { content: "✓ "; color: var(--green); font-weight: 700; }

    .sentiment-positive { color: var(--green); }
    .sentiment-neutral   { color: var(--blue);  }
    .sentiment-negative  { color: var(--red);   }

    @media (max-width: 600px) {
      header h1 { font-size: 1.1rem; }
      .hero h2 { font-size: 1.4rem; }
      .score-num { font-size: 3rem; }
    }
  </style>
</head>
<body>

<header>
  <div>
    <h1>⚡ EngageAI</h1>
    <span>Social Media Content Analyzer</span>
  </div>
</header>

<main>
  <div class="hero">
    <h2>Analyze Your Content's Engagement Potential</h2>
    <p>Upload a PDF or image of your social media post and get an instant AI-powered engagement score with actionable suggestions.</p>
  </div>

  <!-- Upload form -->
  <div class="upload-area" onclick="document.getElementById('fileInput').click()">
    <input type="file" id="fileInput" accept=".pdf,.png,.jpg,.jpeg,.webp" onchange="onFileSelect(event)" />
    <div class="upload-icon">📄</div>
    <p>Drag &amp; drop or <strong>click to upload</strong></p>
    <p>Supports PDF, PNG, JPG, WEBP · Max 10 MB</p>
    <div class="file-name" id="fileName"></div>
  </div>

  <div style="text-align:center">
    <button class="btn" id="analyzeBtn" onclick="analyze()" disabled>Analyze Content</button>
    <div class="error-msg" id="errorMsg"></div>
  </div>

  <!-- Loading spinner -->
  <div class="spinner-wrap" id="spinnerWrap">
    <div class="spinner"></div>
    <p>Analyzing your content with AI…</p>
  </div>

  <!-- Results -->
  <div id="results">

    <div class="score-card">
      <div class="score-num" id="scoreNum">–</div>
      <div class="score-label">Engagement Score (out of 100)</div>
    </div>

    <div class="cards-row" id="metricsRow"></div>

    <div class="card">
      <h3>📝 Extracted Text</h3>
      <textarea id="extractedText" readonly></textarea>
    </div>

    <div class="card">
      <h3>💡 Improvement Suggestions</h3>
      <div class="suggestions-grid" id="suggestionsGrid"></div>
    </div>

  </div>
</main>

<script>
  let selectedFile = null;

  // Drag-and-drop support
  const uploadArea = document.querySelector('.upload-area');
  uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.style.background = '#EBF1FD'; });
  uploadArea.addEventListener('dragleave', () => { uploadArea.style.background = ''; });
  uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    uploadArea.style.background = '';
    const file = e.dataTransfer.files[0];
    if (file) setFile(file);
  });

  function onFileSelect(event) {
    const file = event.target.files[0];
    if (file) setFile(file);
  }

  function setFile(file) {
    selectedFile = file;
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('analyzeBtn').disabled = false;
    document.getElementById('errorMsg').textContent = '';
    document.getElementById('results').style.display = 'none';
  }

  async function analyze() {
    if (!selectedFile) return;
    document.getElementById('errorMsg').textContent = '';
    document.getElementById('analyzeBtn').disabled = true;
    document.getElementById('spinnerWrap').style.display = 'block';
    document.getElementById('results').style.display = 'none';

    const form = new FormData();
    form.append('file', selectedFile);

    try {
      const resp = await fetch('/api/analyze', { method: 'POST', body: form });
      const data = await resp.json();

      if (!resp.ok) {
        throw new Error(data.detail || 'Analysis failed. Please try again.');
      }
      renderResults(data);
    } catch (err) {
      document.getElementById('errorMsg').textContent = err.message;
    } finally {
      document.getElementById('analyzeBtn').disabled = false;
      document.getElementById('spinnerWrap').style.display = 'none';
    }
  }

  function renderResults(data) {
    const ch = data.characteristics;

    // Score
    document.getElementById('scoreNum').textContent = data.score;

    // Metrics cards
    const sentClass = ch.sentiment === 'Positive' ? 'sentiment-positive'
                    : ch.sentiment === 'Negative' ? 'sentiment-negative'
                    : 'sentiment-neutral';

    const hashtagsSub = ch.hashtag_usage.hashtags.length
      ? ch.hashtag_usage.hashtags.slice(0,8).map(h => `<span class="tag">${h}</span>`).join('')
      : '<span style="color:#aaa">none found</span>';

    const emojisSub = ch.emoji_usage.emojis.length
      ? ch.emoji_usage.emojis.slice(0,10).join(' ')
      : 'none found';

    const ctaText = ch.call_to_action.detected
      ? `<span style="color:var(--green)">✓ Detected</span>`
      : `<span style="color:var(--muted)">✗ None found</span>`;

    document.getElementById('metricsRow').innerHTML = `
      <div class="metric-card">
        <div class="label">Word Count</div>
        <div class="value">${ch.word_count}</div>
        <div class="sub">Avg sentence: ${ch.average_sentence_length} words</div>
      </div>
      <div class="metric-card">
        <div class="label">Sentiment</div>
        <div class="value ${sentClass}">${ch.sentiment}</div>
        <div class="sub">Readability: ${ch.readability_level}</div>
      </div>
      <div class="metric-card">
        <div class="label">Hashtags (${ch.hashtag_usage.count})</div>
        <div class="value" style="font-size:1rem;margin-top:8px">${hashtagsSub}</div>
      </div>
      <div class="metric-card">
        <div class="label">Emojis (${ch.emoji_usage.count})</div>
        <div class="value" style="font-size:1.2rem;margin-top:8px">${emojisSub}</div>
      </div>
      <div class="metric-card">
        <div class="label">Call to Action</div>
        <div class="value" style="font-size:1rem;margin-top:8px">${ctaText}</div>
        <div class="sub">${ch.call_to_action.cta_text ? '"' + ch.call_to_action.cta_text.substring(0,40) + '…"' : ''}</div>
      </div>
    `;

    // Extracted text
    document.getElementById('extractedText').value = data.extracted_text;

    // Suggestions
    const icons = { discoverability: '🔍', readability: '📖', interaction: '💬', tone: '🎙️' };
    const sugg = data.suggestions;
    document.getElementById('suggestionsGrid').innerHTML = Object.entries(sugg).map(([cat, items]) => `
      <div class="sug-group">
        <h4>${icons[cat] || ''} ${cat.charAt(0).toUpperCase() + cat.slice(1)}</h4>
        <ul>${items.map(s => `<li>${s}</li>`).join('')}</ul>
      </div>
    `).join('');

    document.getElementById('results').style.display = 'block';
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
  }
</script>

</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the single-page frontend."""
    return HTML
