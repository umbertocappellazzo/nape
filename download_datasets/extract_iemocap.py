#!/usr/bin/env python3
"""
extract_iemocap.py

Parse the IEMOCAP dataset (USC SAIL release, official layout) and produce a
single manifest.json + classes.json suitable for the SUPERB ER protocol:

  - 4 emotion classes: neutral, happy, sad, angry
  - 'excited' (exc) is merged into 'happy' (hap)
  - All other emotions (fru, fea, sur, dis, oth, xxx) are dropped
  - Each utterance is tagged with its session number (1-5) in the 'fold' field;
    cross-validation is then a leave-one-session-out loop, matching ESC-50.

Expected input layout:
    $SOURCE_DIR/
      Session1/
        dialog/EmoEvaluation/*.txt
        sentences/wav/<dialog_id>/*.wav
      Session2/
      ... up to Session5/

Output:
    $OUTPUT_DIR/manifest.json
    $OUTPUT_DIR/classes.json

Manifest entry format (matches WavManifestClassificationDataset's expectations):
    {"wav": "/abs/path/to/file.wav", "label": "neu", "fold": 1}

Usage:
    python extract_iemocap.py \\
        --source_dir /path/to/IEMOCAP_full_release \\
        --output_dir /ucappell/datasets/iemocap
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# SUPERB ER protocol: 4 final classes after merging exc -> hap
# Keep order stable so classes.json indices are deterministic.
SUPERB_CLASSES = ["neu", "hap", "sad", "ang"]

# Original emotion codes we accept from EmoEvaluation files.
# Each maps to one of the 4 SUPERB classes.
EMOTION_REMAP = {
    "neu": "neu",
    "hap": "hap",
    "exc": "hap",   # excited merged into happy (SUPERB convention)
    "sad": "sad",
    "ang": "ang",
    # Everything else (fru, fea, sur, dis, oth, xxx) is dropped.
}

# Regex to match an utterance line in an EmoEvaluation .txt file.
# Example line:
#   [6.2901 - 8.2357]\tSes01F_impro01_F000\tneu\t[2.5000, 2.5000, 2.5000]
UTT_RE = re.compile(
    r"^\[\s*[\d.]+\s*-\s*[\d.]+\s*\]\s+"   # time bracket
    r"(?P<wav_id>\S+)\s+"                  # wav identifier
    r"(?P<emotion>\w+)\s+"                 # emotion code
    r"\[.*\]\s*$"                          # VAD values (we don't use them)
)


def parse_eval_file(eval_path: Path):
    """Yield (wav_id, emotion_code) for each utterance line in an EmoEvaluation .txt."""
    with eval_path.open("r", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.startswith("["):
                continue
            m = UTT_RE.match(line)
            if m is None:
                continue
            yield m.group("wav_id"), m.group("emotion")


def find_wav_path(session_dir: Path, wav_id: str) -> Path:
    """
    Locate the wav file for a given utterance id.

    Wav files live at: Session{N}/sentences/wav/<dialog_id>/<wav_id>.wav
    where <dialog_id> is the portion of <wav_id> before the final '_' segment
    (i.e., dropping the speaker+turn suffix like '_F000').

    Example: wav_id = 'Ses01F_impro01_F000'
             dialog_id = 'Ses01F_impro01'
             path = 'Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav'
    """
    # wav_id like 'Ses01F_impro01_F000' or 'Ses02M_script03_2_M001'
    # The "dialog id" is everything up to the final '_<speaker_turn>' token.
    # The robust rule: drop the last underscore-separated piece.
    parts = wav_id.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot infer dialog id from wav_id={wav_id!r}")
    dialog_id = parts[0]
    return session_dir / "sentences" / "wav" / dialog_id / f"{wav_id}.wav"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True,
                        help="Path to the IEMOCAP_full_release root "
                             "(must contain Session1/.../Session5/).")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write manifest.json and classes.json.")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_dir.is_dir():
        logger.error(f"--source_dir not found: {source_dir}")
        sys.exit(1)

    # Collect all utterances across the 5 sessions.
    all_entries = []
    raw_emotion_counts = Counter()
    dropped_emotion_counts = Counter()
    missing_wav_count = 0

    for session_idx in range(1, 6):
        session_dir = source_dir / f"Session{session_idx}"
        eval_dir = session_dir / "dialog" / "EmoEvaluation"
        if not eval_dir.is_dir():
            logger.error(f"Missing EmoEvaluation dir: {eval_dir}")
            sys.exit(1)

        eval_files = sorted(eval_dir.glob("*.txt"))
        # The EmoEvaluation directory contains some extra files (e.g.
        # Categorical/, Self-evaluation/ subdirs handled by glob being
        # non-recursive). Filter to only top-level dialog labels:
        # standard names look like 'Ses01F_impro01.txt' etc.
        eval_files = [p for p in eval_files if p.is_file()]

        n_kept_in_session = 0
        for eval_path in eval_files:
            for wav_id, emotion_code in parse_eval_file(eval_path):
                raw_emotion_counts[emotion_code] += 1
                if emotion_code not in EMOTION_REMAP:
                    dropped_emotion_counts[emotion_code] += 1
                    continue

                label = EMOTION_REMAP[emotion_code]
                wav_path = find_wav_path(session_dir, wav_id)
                if not wav_path.exists():
                    missing_wav_count += 1
                    logger.warning(f"Missing wav file: {wav_path}")
                    continue

                all_entries.append({
                    "wav": str(wav_path),
                    "labels": [label],
                    "fold": session_idx,
                })
                n_kept_in_session += 1

        logger.info(f"Session{session_idx}: kept {n_kept_in_session} utterances "
                    f"from {len(eval_files)} EmoEvaluation files")

    # ---- Diagnostics ----
    logger.info("=== Summary ===")
    logger.info(f"Total kept utterances: {len(all_entries)}")
    logger.info(f"Raw emotion counts (all codes seen): "
                f"{dict(raw_emotion_counts.most_common())}")
    logger.info(f"Dropped emotion counts: "
                f"{dict(dropped_emotion_counts.most_common())}")
    if missing_wav_count > 0:
        logger.warning(f"Missing wav files: {missing_wav_count} "
                       f"(these utterances were skipped)")

    # Per-class counts after remap
    final_counts = Counter(e["labels"][0] for e in all_entries)
    logger.info(f"Final per-class counts: "
                f"{dict(sorted(final_counts.items()))}")
    fold_counts = Counter(e["fold"] for e in all_entries)
    logger.info(f"Per-fold (session) counts: "
                f"{dict(sorted(fold_counts.items()))}")

    # ---- Write outputs ----
    classes_path = output_dir / "classes.json"
    with classes_path.open("w") as f:
        json.dump({"classes": SUPERB_CLASSES}, f, indent=2)
    logger.info(f"Wrote classes file: {classes_path}")

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump({"data": all_entries}, f, indent=2)
    logger.info(f"Wrote manifest: {manifest_path} ({len(all_entries)} entries)")

    logger.info("Next step: compute normalization stats")
    logger.info("  python compute_norm_stats.py \\")
    logger.info(f"      --manifest {manifest_path} \\")
    logger.info(f"      --output {output_dir / 'norm_stats.json'} \\")
    logger.info("      --audio_duration 10.0")


if __name__ == "__main__":
    main()
