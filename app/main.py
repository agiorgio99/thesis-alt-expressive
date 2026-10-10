"""
main.py — ExpressiveALT showcase app (local Gradio).

Tabs:
* Dataset           — real GTSinger takes (control / technique / speech) with plots
* Augmentation lab  — synthesise a technique on a control take, compare to real
* ASR               — transcribe with zero-shot or fine-tuned Whisper, score it
* Listening test    — blind naturalness / technique ratings of the pilot clips
* Results           — baselines, fine-tuning ablation, calibration, pilot

Run from the repository root:

    python -m app.main                  # http://127.0.0.1:7860
    python -m app.main --host 0.0.0.0   # reachable from the LAN / a container

Audio is read from ``data/`` at runtime and never bundled (GTSinger is
CC BY-NC-SA 4.0 — non-commercial use with attribution).
"""

from __future__ import annotations

import argparse
import warnings

import gradio as gr

from app import state
from app.tabs import asr, aug_lab, explorer, listening, results

# pyworld imports the deprecated pkg_resources; the warning is noise here.
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")


def build_app() -> gr.Blocks:
    """Assemble all tabs into one Blocks app."""
    with gr.Blocks(title="ExpressiveALT") as demo:
        gr.Markdown("# ExpressiveALT — expressive singing, augmentation & ASR\n"
                    "GTSinger data © original authors, CC BY-NC-SA 4.0.")
        with gr.Tab("Dataset"):
            explorer.build(demo)
        with gr.Tab("Augmentation lab"):
            aug_lab.build(demo)
        with gr.Tab("ASR"):
            asr.build(demo)
        with gr.Tab("Listening test"):
            listening.build(demo)
        with gr.Tab("Results"):
            results.build(demo)
    return demo


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    state.index()  # build/load the data index before the first request
    build_app().launch(
        server_name=args.host, server_port=args.port,
        # Gradio only serves files from allowed paths: the dataset trees.
        allowed_paths=[str(state.DATA_ROOT), str(state.AUG_ROOT),
                       str(state.REPO / "results")],
    )


if __name__ == "__main__":
    main()
