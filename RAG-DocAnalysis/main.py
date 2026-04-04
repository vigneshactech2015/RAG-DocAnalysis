"""RAG-DocAnalysis CLI entry-point.

Usage
-----
    python main.py ingest                        # embed docs into ChromaDB
    python main.py evaluate                      # run 15-question eval suite
    python main.py evaluate --template detailed  # eval with a specific template
    python main.py api                           # start FastAPI on :8000
    python main.py ui                            # start Gradio UI on :7860
    python main.py all                           # start API + UI together
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> None:
    from app.ingestion import ingest
    ingest()
    print("\nIngestion complete. You can now run  python main.py api  or  python main.py ui.")


def cmd_evaluate(args: argparse.Namespace) -> None:
    import json
    from app.evaluation import run_evaluation

    print(f"\nRunning evaluation | template={args.template!r} preset={args.preset!r}\n")
    summary = run_evaluation(template_name=args.template, preset_name=args.preset)

    # Print aggregate metrics; detailed per-question JSON is saved to disk
    display = {k: v for k, v in summary.items() if k != "results"}
    print(json.dumps(display, indent=2))
    print(f"\nPer-question results saved to  data/eval_results/eval_{args.template}_{args.preset}.json")


def cmd_api(args: argparse.Namespace) -> None:
    import uvicorn
    from api.server import app

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def cmd_ui(args: argparse.Namespace) -> None:
    import os
    os.environ.setdefault("GRADIO_PORT", str(args.port))

    from ui.gradio_app import demo
    demo.launch(server_name=args.host, server_port=args.port, share=False)


def cmd_all(args: argparse.Namespace) -> None:
    """Start the FastAPI server in a daemon thread, then launch Gradio."""
    import threading
    import uvicorn
    from api.server import app as fastapi_app

    api_thread = threading.Thread(
        target=lambda: uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="info"),
        daemon=True,
        name="fastapi",
    )
    api_thread.start()
    print("FastAPI started at http://localhost:8000  (docs: http://localhost:8000/docs)")

    from ui.gradio_app import demo
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-docanalysis",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ingest
    p = sub.add_parser("ingest", help="Load docs, embed them, and store in ChromaDB")
    p.set_defaults(func=cmd_ingest)

    # evaluate
    p = sub.add_parser("evaluate", help="Run the 15-question evaluation suite")
    p.add_argument(
        "--template",
        default="concise",
        choices=["concise", "detailed", "conversational"],
        help="Prompt template to test (default: concise)",
    )
    p.add_argument(
        "--preset",
        default="precise",
        choices=["precise", "creative"],
        help="LLM parameter preset to test (default: precise)",
    )
    p.set_defaults(func=cmd_evaluate)

    # api
    p = sub.add_parser("api", help="Start the FastAPI REST server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_api)

    # ui
    p = sub.add_parser("ui", help="Start the Gradio chat interface")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.set_defaults(func=cmd_ui)

    # all
    p = sub.add_parser("all", help="Start FastAPI + Gradio together")
    p.set_defaults(func=cmd_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
