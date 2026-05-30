"""
EngageAI - Social Media Content Analyzer
=========================================
Main entry point. Run with:
    uvicorn main:app --reload
Or for CLI usage:
    python main.py --file path/to/file.pdf
"""

import argparse
import asyncio
import sys
from pathlib import Path

import uvicorn

from modules.api import app  # FastAPI app


def run_cli(file_path: str) -> None:
    """
    Run the analysis pipeline from the command line without the web server.
    Useful for quick batch analysis or testing.
    """
    from modules.extractor import extract_text
    from modules.analyzer import analyze_characteristics
    from modules.scorer import generate_engagement_score
    from modules.suggestions import provide_improvement_suggestions
    from modules.display import print_report

    path = Path(file_path)
    if not path.exists():
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    print(f"\n🔍  Analyzing: {path.name}\n")

    # Step 1 – Extract text
    extracted_text, error = extract_text(str(path))
    if error:
        print(f"[ERROR] {error}")
        sys.exit(1)

    # Step 2 – Analyze characteristics
    characteristics = analyze_characteristics(extracted_text)

    # Step 3 – Generate score + suggestions concurrently (using asyncio)
    async def _parallel():
        score_task = generate_engagement_score(extracted_text, characteristics)
        suggestions_task = provide_improvement_suggestions(extracted_text, characteristics)
        return await asyncio.gather(score_task, suggestions_task)

    score, suggestions = asyncio.run(_parallel())

    # Step 4 – Print full report
    print_report(extracted_text, characteristics, score, suggestions)


def main():
    parser = argparse.ArgumentParser(
        description="EngageAI – Social Media Content Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Start web server:
    python main.py

  Analyze a file from CLI:
    python main.py --file post.pdf
    python main.py --file screenshot.png
        """,
    )
    parser.add_argument("--file", type=str, help="Path to a PDF, PNG, or JPG file to analyze (CLI mode)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for the web server (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port for the web server (default: 8000)")
    args = parser.parse_args()

    if args.file:
        run_cli(args.file)
    else:
        print("🚀  Starting EngageAI web server ...")
        print(f"    Open http://{args.host}:{args.port} in your browser.\n")
        uvicorn.run("modules.api:app", host=args.host, port=args.port, reload=True)


if __name__ == "__main__":
    main()
