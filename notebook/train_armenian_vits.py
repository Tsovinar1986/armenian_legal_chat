"""Trains a single-speaker Eastern Armenian VITS TTS model on the
davit312/Armenian-speech-hy audiobook corpus (see prepare_tts_dataset.py),
to replace the robotic gTTS voice in src/services/voice.py.

Character-based (no phonemizer) since espeak-ng's Armenian phoneme support
is unconfirmed/unavailable in this environment -- VITS trains fine on raw
characters, just needs a bit more data/steps to learn pronunciation than
phoneme-based input would.

Runs on MPS (Apple Silicon GPU) if available -- confirmed present on this
machine's PyTorch install (see the M1 GPU vs. CPU timing lesson from the
legal-model fine-tune: always prefer GPU/MPS over CPU where available).

Usage:
    ./tts_env/bin/python notebook/train_armenian_vits.py
"""
import glob
import os

import torch
from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

_ROOT = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_PATH = os.path.join(_ROOT, "tts_data", "runs")


def main():
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata.csv",
        path=os.path.join(_ROOT, "tts_data"),
    )

    # Exact character set present in tts_data/metadata.csv (see the inspection
    # that produced this list) -- full Armenian alphabet (upper/lower), Armenian
    # punctuation (։ ՞ ՛ ՜ ՝ և — the "ev" ligature), plus the small amount of
    # Latin punctuation/digits that appear in the source audiobook transcripts.
    armenian_characters = CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters=(
            "ԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՒՓՔՕՖ"
            "աբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքօֆև"
            "'(),-.:`«»´ ՛՜՝՞։֊—…0123456789"
        ),
        punctuations="'(),-.:`«»´ ՛՜՝՞։֊—…",
    )

    audio_config = VitsAudioConfig(
        sample_rate=22050,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=None,
    )

    config = VitsConfig(
        audio=audio_config,
        run_name="armenian_vits",
        batch_size=16,
        eval_batch_size=8,
        batch_group_size=5,
        num_loader_workers=4,
        num_eval_loader_workers=2,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        text_cleaner="basic_cleaners",  # no English-specific normalization
        use_phonemes=False,
        compute_input_seq_cache=True,
        print_step=25,
        print_eval=False,
        mixed_precision=False,  # MPS doesn't support AMP the way CUDA does
        output_path=OUTPUT_PATH,
        datasets=[dataset_config],
        characters=armenian_characters,
        # ~150 steps is roughly 2 hours at this model/hardware's measured
        # ~48s/step -- deliberately frequent (not Coqui's usual 1000-step
        # default) because this machine has already rebooted twice in one
        # session, wiping in-progress work each time. Losing up to ~2 hours
        # to a reboot is an acceptable cost; losing many hours is not.
        # save_n_checkpoints=5 keeps a bit of history in case the very
        # latest checkpoint was written mid-corruption by a reboot.
        save_step=150,
        save_n_checkpoints=5,
        test_sentences=[
            # Held out sanity-check sentences (not literal training transcripts)
            # to eyeball pronunciation/naturalness on ordinary Armenian text
            # rather than just memorized training clips.
            ["Ողջույն, ինչպե՞ս եք։"],
            ["Ես ուրախ եմ ձեզ օգնել իրավական հարցերում։"],
        ],
    )

    ap = AudioProcessor.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_size=0.02,
    )

    tokenizer, config = TTSTokenizer.init_from_config(config)
    model = Vits(config, ap, tokenizer=tokenizer, speaker_manager=None)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Auto-resume: this is meant to run unattended across reboots (see the
    # save_step comment above), so every invocation must pick up from the
    # latest run's latest checkpoint instead of silently starting a fresh
    # run from scratch each time it's relaunched. continue_path finds and
    # restores the latest checkpoint (model + optimizer + scheduler state,
    # plus the epoch/step counters) from an existing run dir on its own.
    existing_runs = sorted(
        d for d in glob.glob(os.path.join(OUTPUT_PATH, "armenian_vits-*")) if os.path.isdir(d)
    )
    continue_path = existing_runs[-1] if existing_runs else ""
    if continue_path:
        print(f"Resuming from latest run: {continue_path}")

    trainer = Trainer(
        TrainerArgs(continue_path=continue_path),
        config,
        OUTPUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


# Required on macOS: DataLoader workers use the 'spawn' multiprocessing start
# method (unlike Linux's 'fork'), which re-imports this file as __main__ in
# each worker process. Without this guard, every worker re-ran the entire
# training setup (including trainer.fit() itself) instead of just importing
# definitions -- visible as duplicate "EPOCH: 0/999" blocks in the log from
# concurrent trainer instances stepping on each other.
if __name__ == "__main__":
    main()
