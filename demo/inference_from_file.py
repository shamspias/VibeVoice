import argparse
import os
import re
import sys
from typing import List, Tuple, Dict
import time
import difflib
import shutil
import tempfile

import torch

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# --------------------------
# Colab / CUDA helpers
# --------------------------
def prefer_fp16_on_this_gpu() -> bool:
    """Return True if this GPU benefits from fp16 over bf16 (pre-Ampere)."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major < 8  # Ampere (8.0+) has solid bf16; below that prefer fp16


def set_cuda_allocator_env():
    # Reduces fragmentation on large graphs & long runs
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def fa2_available() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def print_gpu_info():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"CUDA available: True | GPU: {name} | CC: {cap[0]}.{cap[1]} | VRAM: {total:.2f} GiB")
    else:
        print("CUDA available: False (CPU mode)")


# --------------------------
# Voice mapping utilities
# --------------------------
class VoiceMapper:
    """Maps speaker names to voice file paths"""

    def __init__(self):
        self.setup_voice_presets()

        # Generate simple aliases from filenames:
        # "Ava_West" -> "Ava", "West"
        # "Andrew-West" -> "West"
        new_dict = {}
        for name, path in self.voice_presets.items():
            alias = name
            if "_" in alias:
                alias = alias.split("_")[0]
                new_dict[alias] = path
            alias2 = name
            if "-" in alias2:
                alias2 = alias2.split("-")[-1]
                new_dict[alias2] = path
        self.voice_presets.update(new_dict)

    def setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory."""
        voices_dir = os.path.join(os.path.dirname(__file__), "voices")

        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            self.voice_presets = {}
            self.available_voices = {}
            return

        self.voice_presets = {}
        wav_files = [
            f for f in os.listdir(voices_dir)
            if f.lower().endswith(".wav") and os.path.isfile(os.path.join(voices_dir, f))
        ]
        for wav_file in wav_files:
            name = os.path.splitext(wav_file)[0]
            full_path = os.path.join(voices_dir, wav_file)
            self.voice_presets[name] = full_path

        self.voice_presets = dict(sorted(self.voice_presets.items()))
        self.available_voices = {
            name: path for name, path in self.voice_presets.items() if os.path.exists(path)
        }

        print(f"Found {len(self.available_voices)} voice files in {voices_dir}")
        if self.available_voices:
            print(f"Available voices: {', '.join(self.available_voices.keys())}")

    def list_speakers(self) -> List[str]:
        """All recognizable speaker names (including normalized aliases created in __init__)."""
        return sorted(self.voice_presets.keys())

    def best_match(self, speaker_name: str) -> Tuple[str, str, str]:
        """
        Return (matched_key, path, match_type) where match_type is 'exact' | 'partial' | 'fallback'.
        """
        if speaker_name in self.voice_presets:
            return speaker_name, self.voice_presets[speaker_name], "exact"

        speaker_lower = speaker_name.lower()
        for preset_name, path in self.voice_presets.items():
            if preset_name.lower() in speaker_lower or speaker_lower in preset_name.lower():
                return preset_name, path, "partial"

        if not self.voice_presets:
            return "", "", "fallback"
        default_key = next(iter(self.voice_presets))
        return default_key, self.voice_presets[default_key], "fallback"

    def get_voice_path(self, speaker_name: str) -> str:
        """Get voice file path for a given speaker name; falls back if no match."""
        key, path, match_type = self.best_match(speaker_name)
        if match_type == "fallback":
            print(
                f"Warning: No voice preset found for '{speaker_name}', using default voice: {os.path.basename(path)}"
            )
        elif match_type == "partial" and key != speaker_name:
            print(f"Info: '{speaker_name}' matched preset '{key}'.")
        return path


# --------------------------
# Text script parsing
# --------------------------
def parse_txt_script(txt_content: str) -> Tuple[List[str], List[str]]:
    """
    Parse txt script content and extract speakers and their text
    Fixed pattern: Speaker 1, Speaker 2, Speaker 3, Speaker 4
    Returns: (scripts, speaker_numbers)
    """
    lines = txt_content.strip().split("\n")
    scripts = []
    speaker_numbers = []

    speaker_pattern = r"^Speaker\s+(\d+):\s*(.*)$"

    current_speaker = None
    current_text = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(speaker_pattern, line, re.IGNORECASE)
        if match:
            if current_speaker and current_text:
                scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
                speaker_numbers.append(current_speaker)

            current_speaker = match.group(1).strip()
            current_text = match.group(2).strip()
        else:
            if current_text:
                current_text += " " + line
            else:
                current_text = line

    if current_speaker and current_text:
        scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
        speaker_numbers.append(current_speaker)

    return scripts, speaker_numbers


# --------------------------
# CLI helper outputs
# --------------------------
def print_available_voices(vm: VoiceMapper):
    names = vm.list_speakers()
    print(f"\nAvailable speaker names ({len(names)}):")
    for n in names:
        print(f"  - {n}")
    print()


def check_speaker_names(vm: VoiceMapper, speaker_names: List[str]) -> int:
    """
    Validate the given names. Returns 0 if all are 'exact' or 'partial',
    otherwise prints details and returns 1.
    """
    print("\nChecking requested speaker names...\n")
    exit_code = 0
    for name in speaker_names:
        matched_key, path, match_type = vm.best_match(name)
        if match_type == "fallback":
            exit_code = 1
            suggestions = difflib.get_close_matches(name, vm.list_speakers(), n=5, cutoff=0.5)
            msg = f"[MISSING] '{name}' → no match"
            if matched_key:
                msg += f" (would fallback to '{matched_key}')"
            print(msg + ".")
            if suggestions:
                print(f"          Did you mean: {', '.join(suggestions)}")
        else:
            label = "OK (exact)" if match_type == "exact" else f"OK (partial → '{matched_key}')"
            print(f"[{label}] '{name}' → {os.path.basename(path)}")
    print()
    return exit_code


# --------------------------
# (Optional) Preprocess voice samples for Colab
#   - Convert to mono, 24kHz
#   - Trim to max_ref_seconds
#   - Returns new file paths in a temp dir (cleaned up at process end)
# --------------------------
def maybe_preprocess_voices(voice_paths: List[str], target_sr: int, max_seconds: float, enable: bool) -> Tuple[List[str], str]:
    """
    Returns (processed_paths, temp_dir). Caller should delete temp_dir when done (we do it at the end).
    If preprocessing is disabled or torchaudio missing, returns original paths and empty temp_dir.
    """
    if not enable:
        return voice_paths, ""

    try:
        import torchaudio
    except Exception:
        print("torchaudio not available; skipping voice preprocessing.")
        return voice_paths, ""

    tmpdir = tempfile.mkdtemp(prefix="vibevoice_tmp_voices_")
    processed = []
    for i, p in enumerate(voice_paths):
        try:
            wav, sr = torchaudio.load(p)  # [channels, samples]
            # to mono
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            # resample
            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)
                sr = target_sr
            # trim/cap duration
            max_len = int(max(1, max_seconds) * sr)
            if wav.size(1) > max_len:
                wav = wav[:, :max_len]
            # ensure contiguous
            wav = wav.contiguous()
            out_path = os.path.join(tmpdir, f"voice_{i:02d}.wav")
            torchaudio.save(out_path, wav, sr)
            processed.append(out_path)
        except Exception as e:
            print(f"Preprocess failed for {p}: {e}. Using original.")
            processed.append(p)
    return processed, tmpdir


# --------------------------
# CLI
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice Processor TXT Input (Colab friendly)")
    parser.add_argument(
        "--model_path",
        type=str,
        default="microsoft/VibeVoice-1.5B",
        help="Path or HF id of the model (default: microsoft/VibeVoice-1.5B)",
    )
    parser.add_argument(
        "--txt_path",
        type=str,
        default="demo/text_examples/1p_abs.txt",
        help="Path to the txt file containing the script",
    )
    parser.add_argument(
        "--speaker_names",
        type=str,
        nargs="+",
        default=["Andrew"],
        help="Speaker names in order (e.g., --speaker_names Andrew Ava 'Bill Gates')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save output audio files",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=2.0,
        help="CFG (Classifier-Free Guidance) scale for generation",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
        help="DDPM inference steps (lower for lower VRAM; default 10)",
    )
    parser.add_argument(
        "--ddpm_batch_mul",
        type=int,
        default=1,
        help="Micro-batch multiplier for diffusion head (1 lowers VRAM; default 1)",
    )
    parser.add_argument(
        "--max_ref_seconds",
        type=float,
        default=8.0,
        help="Cap reference voice duration in seconds (preprocessing).",
    )
    parser.add_argument(
        "--resample_voices",
        action="store_true",
        help="If set, preprocess reference wavs to mono/24kHz and cap duration.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=24000,
        help="Assumed/generation sample rate (for duration metrics & preprocessing).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "fp16", "bf16"],
        default="auto",
        help="Computation dtype. 'auto' = fp16 on pre-Ampere, bf16 otherwise.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Transformers device_map. 'auto' can offload to CPU to save VRAM.",
    )
    parser.add_argument(
        "--force_sdpa",
        action="store_true",
        help="Force SDPA attention (ignore FlashAttention2 even if installed).",
    )
    parser.add_argument(
        "--list_voices",
        action="store_true",
        help="List all available speaker names and exit",
    )
    parser.add_argument(
        "--check_speaker_names",
        action="store_true",
        help="Validate provided --speaker_names and exit with non-zero on missing",
    )

    return parser.parse_args()


# --------------------------
# Main
# --------------------------
def main():
    set_cuda_allocator_env()
    # Speed-friendly default in Colab
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    args = parse_args()
    print_gpu_info()

    # Initialize voice mapper / utilities
    voice_mapper = VoiceMapper()

    if args.list_voices:
        print_available_voices(voice_mapper)
        sys.exit(0)

    if args.check_speaker_names:
        names = args.speaker_names if isinstance(args.speaker_names, list) else [args.speaker_names]
        rc = check_speaker_names(voice_mapper, names)
        if rc != 0:
            print("One or more speaker names were not found. See suggestions above.")
        sys.exit(rc)

    # Validate txt file
    if not os.path.exists(args.txt_path):
        print(f"Error: txt file not found: {args.txt_path}")
        sys.exit(1)

    # Read script
    print(f"Reading script from: {args.txt_path}")
    with open(args.txt_path, "r", encoding="utf-8") as f:
        txt_content = f.read()

    scripts, speaker_numbers = parse_txt_script(txt_content)
    if not scripts:
        print("Error: No valid speaker scripts found in the txt file")
        sys.exit(1)

    print(f"Found {len(scripts)} speaker segments:")
    for i, (script, speaker_num) in enumerate(zip(scripts, speaker_numbers)):
        print(f"  {i+1}. Speaker {speaker_num}")
        print(f"     Text preview: {script[:100]}...")

    # Map number -> provided name
    speaker_name_mapping: Dict[str, str] = {}
    speaker_names_list = args.speaker_names if isinstance(args.speaker_names, list) else [args.speaker_names]
    for i, name in enumerate(speaker_names_list, 1):
        speaker_name_mapping[str(i)] = name

    print("\nSpeaker mapping:")
    for speaker_num in sorted(set(speaker_numbers), key=lambda x: int(x)):
        mapped_name = speaker_name_mapping.get(speaker_num, f"Speaker {speaker_num}")
        print(f"  Speaker {speaker_num} -> {mapped_name}")

    # Unique speakers in order of appearance
    unique_speaker_numbers: List[str] = []
    seen = set()
    for s in speaker_numbers:
        if s not in seen:
            unique_speaker_numbers.append(s)
            seen.add(s)

    # Resolve voices
    voice_paths = []
    for speaker_num in unique_speaker_numbers:
        normalized_name = speaker_name_mapping.get(speaker_num, f"Speaker {speaker_num}")
        vp = voice_mapper.get_voice_path(normalized_name)
        voice_paths.append(vp)
        print(f"Speaker {speaker_num} ('{normalized_name}') -> Voice: {os.path.basename(vp)}")

    # Optional preprocessing of voice WAVs (trim/resample)
    tmp_voice_dir = ""
    processed_voice_paths = voice_paths
    if args.resample_voices:
        processed_voice_paths, tmp_voice_dir = maybe_preprocess_voices(
            voice_paths, target_sr=args.sample_rate, max_seconds=args.max_ref_seconds, enable=True
        )
        if tmp_voice_dir:
            print(f"Preprocessed voices saved to temp dir: {tmp_voice_dir}")

    # Prepare script for model
    full_script = "\n".join(scripts).replace("’", "'")

    # Processor
    print(f"Loading processor from {args.model_path}")
    processor = VibeVoiceProcessor.from_pretrained(args.model_path)

    # Choose dtype
    if args.dtype == "fp16":
        chosen_dtype = torch.float16
    elif args.dtype == "bf16":
        chosen_dtype = torch.bfloat16
    else:
        chosen_dtype = torch.float16 if prefer_fp16_on_this_gpu() else torch.bfloat16

    # Choose attention impl
    use_flash = (not args.force_sdpa) and fa2_available()

    # Device map (auto can help reduce VRAM by CPU offload)
    device_map = args.device_map

    # Model
    print(f"Using dtype: {chosen_dtype}, device_map: {device_map}, attention: {'flash_attn_2' if use_flash else 'sdpa'}")
    try:
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            args.model_path,
            torch_dtype=chosen_dtype,
            device_map=device_map,
            attn_implementation="flash_attention_2" if use_flash else "sdpa",
        )
    except Exception as e:
        print(f"[WARN] Primary load failed ({type(e).__name__}): {e}")
        print("Falling back to SDPA.")
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            args.model_path,
            torch_dtype=chosen_dtype,
            device_map=device_map,
            attn_implementation="sdpa",
        )

    model.eval()
    # Diffusion settings for lower VRAM
    model.set_ddpm_inference_steps(num_steps=int(args.num_steps))
    try:
        model.model.diffusion_head.config.ddpm_batch_mul = int(args.ddpm_batch_mul)
        print(f"Set diffusion ddpm_batch_mul={args.ddpm_batch_mul}")
    except Exception:
        pass

    if hasattr(model.model, "language_model"):
        print(f"Language model attention: {model.model.language_model.config._attn_implementation}")

    # Build inputs
    inputs = processor(
        text=[full_script],                  # batch of 1
        voice_samples=[processed_voice_paths],  # batch of 1: list of reference paths
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    # Move to CUDA if needed (when device_map is not 'auto')
    # With device_map='auto', HF handles placement; otherwise, ensure tensors on cuda
    if device_map != "auto" and torch.cuda.is_available():
        inputs = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inputs.items()}

    print(f"Starting generation with cfg_scale: {args.cfg_scale}, steps: {args.num_steps}")

    torch.cuda.empty_cache()
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=args.cfg_scale,
        tokenizer=processor.tokenizer,
        generation_config={"do_sample": False},
        verbose=True,
    )
    generation_time = time.time() - start_time
    print(f"Generation time: {generation_time:.2f} seconds")

    # Metrics
    sample_rate = args.sample_rate
    audio_duration = 0.0
    rtf = float("inf")

    if getattr(outputs, "speech_outputs", None) is not None and outputs.speech_outputs[0] is not None:
        audio = outputs.speech_outputs[0]
        audio_samples = audio.shape[-1] if hasattr(audio, "shape") and len(audio.shape) > 0 else len(audio)
        audio_duration = audio_samples / float(sample_rate)
        rtf = generation_time / audio_duration if audio_duration > 0 else float("inf")
        print(f"Generated audio duration: {audio_duration:.2f} seconds")
        print(f"RTF (Real Time Factor): {rtf:.2f}x")
    else:
        print("No audio output generated.")

    input_tokens = inputs["input_ids"].shape[1]
    output_tokens = outputs.sequences.shape[1]
    generated_tokens = output_tokens - input_tokens
    print(f"Prefilling tokens: {input_tokens}")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Total tokens: {output_tokens}")

    # Save audio
    txt_filename = os.path.splitext(os.path.basename(args.txt_path))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{txt_filename}_generated.wav")
    processor.save_audio(outputs.speech_outputs[0], output_path=out_path)
    print(f"Saved output to {out_path}")

    # Summary
    print("\n" + "=" * 50)
    print("GENERATION SUMMARY")
    print("=" * 50)
    print(f"Input file: {args.txt_path}")
    print(f"Output file: {out_path}")
    print(f"Speaker names: {args.speaker_names}")
    print(f"Number of unique speakers: {len(set(speaker_numbers))}")
    print(f"Number of segments: {len(scripts)}")
    print(f"Prefilling tokens: {input_tokens}")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Total tokens: {output_tokens}")
    print(f"Generation time: {generation_time:.2f} seconds")
    print(f"Audio duration: {audio_duration:.2f} seconds")
    print(f"RTF (Real Time Factor): {rtf:.2f}x")
    print("=" * 50)

    # Cleanup temp voice dir
    if tmp_voice_dir and os.path.isdir(tmp_voice_dir):
        shutil.rmtree(tmp_voice_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
