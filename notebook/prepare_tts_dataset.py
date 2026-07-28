"""Builds an LJSpeech-format metadata.csv (Coqui TTS's VITS recipe format:
wav_filename|transcription|normalized_transcription, no header) from the
davit312/Armenian-speech-hy audiobook corpus (tts_data/raw/{book}/*.wav +
matching *.txt), for training a single-voice Eastern Armenian VITS model to
replace the robotic gTTS voice in src/services/voice.py.

Copies every wav into a flat tts_data/wavs/ dir (Coqui's LJSpeech formatter
expects all wavs in one directory next to metadata.csv), skipping any file
whose transcript is empty/whitespace-only after stripping.

Usage:
    ./tts_env/bin/python notebook/prepare_tts_dataset.py
Writes tts_data/metadata.csv and prints corpus stats.
"""
import glob
import os
import shutil

_ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(_ROOT, "tts_data", "raw")
WAVS_DIR = os.path.join(_ROOT, "tts_data", "wavs")
OUT_PATH = os.path.join(_ROOT, "tts_data", "metadata.csv")


def main():
    wav_paths = sorted(glob.glob(os.path.join(RAW_DIR, "**", "*.wav"), recursive=True))
    os.makedirs(WAVS_DIR, exist_ok=True)

    rows = []
    skipped_no_text = 0
    skipped_empty = 0
    for wav_path in wav_paths:
        txt_path = os.path.splitext(wav_path)[0] + ".txt"
        if not os.path.exists(txt_path):
            skipped_no_text += 1
            continue
        text = open(txt_path, encoding="utf-8").read().strip()
        if not text:
            skipped_empty += 1
            continue

        # Flatten into tts_data/wavs/ with a unique name (book dir + original
        # filename) since filenames like "gm003-11" repeat across book dirs.
        book = os.path.basename(os.path.dirname(wav_path))
        clip_id = f"{book}-{os.path.splitext(os.path.basename(wav_path))[0]}"
        dest = os.path.join(WAVS_DIR, f"{clip_id}.wav")
        if not os.path.exists(dest):
            shutil.copyfile(wav_path, dest)
        rows.append((clip_id, text))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for clip_id, text in rows:
            f.write(f"{clip_id}|{text}|{text}\n")

    print(f"Wrote {len(rows)} rows to {OUT_PATH}")
    print(f"Skipped: {skipped_no_text} missing .txt, {skipped_empty} empty transcript")


if __name__ == "__main__":
    main()
