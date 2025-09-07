import argparse
import os
import re
import sys
import traceback
from typing import List, Tuple, Union, Dict, Any, Optional
import time
import torch
import difflib
import tempfile
import shutil

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# =========================
# CUDA / Colab helpers
# =========================
def prefer_fp16_on_this_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major < 8  # pre-Ampere: prefer fp16


def set_cuda_allocator_env():
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
        print(f"CUDA: True | GPU: {name} | CC: {cap[0]}.{cap[1]} | VRAM: {total:.2f} GiB")
    else:
        print("CUDA: False (CPU mode)")


# =========================
# Optional voice preprocess
# =========================
def maybe_preprocess_voices(voice_paths: List[str], target_sr: int, max_seconds: float, enable: bool) -> Tuple[
    List[str], str]:
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
            wav, sr = torchaudio.load(p)  # [C, T]
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)
                sr = target_sr
            max_len = int(max(1, max_seconds) * sr)
            if wav.size(1) > max_len:
                wav = wav[:, :max_len]
            out_path = os.path.join(tmpdir, f"voice_{i:02d}.wav")
            torchaudio.save(out_path, wav.contiguous(), sr)
            processed.append(out_path)
        except Exception as e:
            print(f"Preprocess failed for {p}: {e}. Using original.")
            processed.append(p)
    return processed, tmpdir


# =========================
# Voice mapping utilities
# =========================
class VoiceMapper:
    """Maps speaker names to voice file paths and provides utilities to inspect availability."""

    def __init__(self):
        self.setup_voice_presets()
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
        voices_dir = os.path.join(os.path.dirname(__file__), "voices")
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            self.voice_presets: Dict[str, str] = {}
            self.available_voices: Dict[str, str] = {}
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
        self.available_voices = {n: p for n, p in self.voice_presets.items() if os.path.exists(p)}
        print(f"Found {len(self.available_voices)} voice files in {voices_dir}")
        if self.available_voices:
            print(f"Available voices: {', '.join(self.available_voices.keys())}")

    def list_speakers(self) -> List[str]:
        return sorted(self.voice_presets.keys())

    def best_match(self, speaker_name: str) -> Tuple[str, str, str]:
        if not self.voice_presets:
            return "", "", "none"
        if speaker_name in self.voice_presets:
            return speaker_name, self.voice_presets[speaker_name], "exact"
        speaker_lower = speaker_name.lower()
        for preset_name, path in self.voice_presets.items():
            if preset_name.lower() in speaker_lower or speaker_lower in preset_name.lower():
                return preset_name, path, "partial"
        default_key = next(iter(self.voice_presets))
        return default_key, self.voice_presets[default_key], "fallback"

    def get_voice_path(self, speaker_name: str) -> str:
        matched_key, path, match_type = self.best_match(speaker_name)
        if match_type == "none":
            raise RuntimeError("No voice .wav files found in 'voices' directory.")
        if match_type == "fallback":
            print(f"Warning: No voice preset found for '{speaker_name}', using default: {path}")
        elif match_type == "partial" and matched_key != speaker_name:
            print(f"Info: '{speaker_name}' matched preset '{matched_key}'.")
        return path


# =========================
# Script parsing
# =========================
def parse_txt_script(txt_content: str) -> Tuple[List[str], List[str]]:
    lines = txt_content.strip().split("\n")
    scripts: List[str] = []
    speaker_numbers: List[str] = []
    pat = r"^Speaker\s+(\d+):\s*(.*)$"
    current_speaker = None
    current_text = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = re.match(pat, line, re.IGNORECASE)
        if m:
            if current_speaker and current_text:
                scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
                speaker_numbers.append(current_speaker)
            current_speaker = m.group(1).strip()
            current_text = m.group(2).strip()
        else:
            current_text = (current_text + " " + line).strip() if current_text else line
    if current_speaker and current_text:
        scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
        speaker_numbers.append(current_speaker)
    return scripts, speaker_numbers


# =========================
# Utilities (listing/check)
# =========================
def print_available_voices(vm: VoiceMapper):
    names = vm.list_speakers()
    print(f"\nAvailable speaker names ({len(names)}):")
    for n in names:
        print(f"  - {n}")
    print()


def check_speaker_names(vm: VoiceMapper, speaker_names: List[str]) -> int:
    if not vm.list_speakers():
        print("No voice .wav files found in 'voices'. Add at least one .wav and try again.")
        return 2
    print("\nChecking requested speaker names...\n")
    exit_code = 0
    all_known = vm.list_speakers()
    for name in speaker_names:
        matched_key, path, match_type = vm.best_match(name)
        if match_type == "fallback":
            exit_code = 1
            suggestions = difflib.get_close_matches(name, all_known, n=5, cutoff=0.5)
            print(f"[MISSING] '{name}' → no match (would fallback to '{matched_key}').")
            if suggestions:
                print(f"          Did you mean: {', '.join(suggestions)}")
        elif match_type == "none":
            exit_code = 2
            print(f"[ERROR] No voices available for '{name}'.")
        else:
            label = "OK (exact)" if match_type == "exact" else f"OK (partial → '{matched_key}')"
            print(f"[{label}] '{name}' → {os.path.basename(path)}")
    print()
    return exit_code


# =========================
# Chunking & robust generate
# =========================
def sentence_chunks(text: str, max_chars: int) -> List[str]:
    # Split by sentence end but keep delimiters; then pack greedily up to max_chars.
    parts = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    chunks, cur = [], ""
    for p in parts:
        if not p:
            continue
        if len(cur) + (1 if cur else 0) + len(p) <= max_chars:
            cur = p if not cur else f"{cur} {p}"
        else:
            if cur:
                chunks.append(cur)
            if len(p) <= max_chars:
                cur = p
            else:
                # very long sentence: hard wrap
                for i in range(0, len(p), max_chars):
                    seg = p[i:i + max_chars]
                    if seg:
                        chunks.append(seg)
                cur = ""
    if cur:
        chunks.append(cur)
    return chunks


def try_generate_once(
        model,
        processor,
        text_chunk: str,
        voice_paths: List[str],
        cfg_scale: float,
        device_map: str,
) -> Optional[torch.Tensor]:
    inputs = processor(
        text=[text_chunk],
        voice_samples=[voice_paths],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    if device_map != "auto" and torch.cuda.is_available():
        inputs = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=cfg_scale,
        tokenizer=processor.tokenizer,
        generation_config={"do_sample": False},
        verbose=False,
    )
    if getattr(out, "speech_outputs", None) is None or out.speech_outputs[0] is None:
        return None
    return out.speech_outputs[0].detach().cpu()


def safe_generate_with_backoff(
        model,
        processor,
        full_text: str,
        voice_paths: List[str],
        cfg_scale: float,
        sample_rate: int,
        device_map: str,
        initial_max_chars: int,
        min_max_chars: int,
) -> List[torch.Tensor]:
    """
    Robust generation that:
      - splits text into chunks (max_chars)
      - on OOM: halves chunk size and retries
      - further on OOM: reduces steps and forces SDPA
    Returns list of audio tensors to be concatenated.
    """
    audio_pieces: List[torch.Tensor] = []
    max_chars = initial_max_chars

    while True:
        try:
            chunks = sentence_chunks(full_text, max_chars)
            print(f"Generating in {len(chunks)} chunk(s) (max_chars={max_chars}).")
            for idx, ch in enumerate(chunks, 1):
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                t0 = time.time()
                audio = try_generate_once(model, processor, ch, voice_paths, cfg_scale, device_map)
                dt = time.time() - t0
                if audio is None:
                    print(f"[WARN] No audio for chunk {idx}/{len(chunks)}; skipping.")
                else:
                    print(f"Chunk {idx}/{len(chunks)} done ({dt:.2f}s, {audio.shape[-1] / sample_rate:.2f}s audio).")
                    audio_pieces.append(audio)
            break  # success
        except torch.cuda.OutOfMemoryError:
            # Back off strategy: shrink chunk size
            torch.cuda.empty_cache()
            if max_chars > min_max_chars:
                max_chars = max(min_max_chars, max_chars // 2)
                print(f"[OOM] Reducing chunk size. Retrying with max_chars={max_chars}...")
                audio_pieces = []
                continue
            else:
                raise  # let caller handle next-stage backoff
    return audio_pieces


def concat_audio(pieces: List[torch.Tensor]) -> Optional[torch.Tensor]:
    if not pieces:
        return None
    return torch.cat(pieces, dim=-1)


# =========================
# Args
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice TXT → Audio (OOM-resistant)")
    parser.add_argument("--model_path", type=str, default="microsoft/VibeVoice-1.5b")
    parser.add_argument("--txt_path", type=str, default="demo/text_examples/1p_abs.txt")
    parser.add_argument("--speaker_names", type=str, nargs="+", default=["Andrew"])
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--num_steps", type=int, default=10, help="DDPM steps (lower = less VRAM)")
    parser.add_argument("--ddpm_batch_mul", type=int, default=1, help="Diffusion micro-batch (1 = safest)")
    parser.add_argument("--dtype", type=str, choices=["auto", "fp16", "bf16"], default="auto")
    parser.add_argument("--device_map", type=str, default="auto", help="'auto' enables CPU offload")
    parser.add_argument("--force_sdpa", action="store_true", help="Ignore FlashAttention2 even if installed")
    parser.add_argument("--resample_voices", action="store_true", help="Preprocess voices to mono/24kHz & trim")
    parser.add_argument("--max_ref_seconds", type=float, default=8.0)
    parser.add_argument("--sample_rate", type=int, default=24000)

    # NEW: hard OOM safety
    parser.add_argument("--max_chars_per_call", type=int, default=1200, help="Max characters per generation call")
    parser.add_argument("--min_chars_per_call", type=int, default=300, help="Lower bound for backoff")
    parser.add_argument("--retry_steps", type=int, nargs="*", default=[10, 8, 6, 4], help="Steps tried on OOM")
    parser.add_argument("--force_sdpa_on_retry", action="store_true", help="Force SDPA when backoff retried")
    parser.add_argument("--downgrade_dtype_on_retry", action="store_true",
                        help="If bf16 used and OOM persists, retry with fp16")

    # Utilities
    parser.add_argument("--list_voices", action="store_true")
    parser.add_argument("--check_speaker_names", action="store_true")
    return parser.parse_args()


# =========================
# Main
# =========================
def main():
    set_cuda_allocator_env()
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    args = parse_args()
    print_gpu_info()

    # Voice mapper / utilities
    voice_mapper = VoiceMapper()
    if args.list_voices:
        print_available_voices(voice_mapper);
        return
    if args.check_speaker_names:
        names = args.speaker_names if isinstance(args.speaker_names, list) else [args.speaker_names]
        rc = check_speaker_names(voice_mapper, names)
        if rc != 0:
            print("One or more speaker names were not found or no voices exist.");
        return

    # Validate txt
    if not os.path.exists(args.txt_path):
        print(f"Error: txt file not found: {args.txt_path}")
        return

    # Read & parse script
    print(f"Reading script from: {args.txt_path}")
    with open(args.txt_path, "r", encoding="utf-8") as f:
        txt_content = f.read()
    scripts, speaker_numbers = parse_txt_script(txt_content)
    if not scripts:
        print("Error: No valid speaker scripts found.");
        return
    print(f"Found {len(scripts)} segments.")
    for i, (script, sp) in enumerate(zip(scripts, speaker_numbers), 1):
        print(f"  {i}. Speaker {sp} | {min(100, len(script))} chars preview: {script[:100]}...")

    # Map speaker numbers -> names
    speaker_name_mapping: Dict[str, str] = {}
    speaker_names_list = args.speaker_names if isinstance(args.speaker_names, list) else [args.speaker_names]
    for i, name in enumerate(speaker_names_list, 1):
        speaker_name_mapping[str(i)] = name
    print("\nSpeaker mapping:")
    for sp in sorted(set(speaker_numbers), key=int):
        print(f"  Speaker {sp} -> {speaker_name_mapping.get(sp, f'Speaker {sp}')}")

    # Resolve voices
    unique_speakers, seen = [], set()
    for sp in speaker_numbers:
        if sp not in seen:
            unique_speakers.append(sp);
            seen.add(sp)
    voice_paths = []
    try:
        for sp in unique_speakers:
            nm = speaker_name_mapping.get(sp, f"Speaker {sp}")
            vp = voice_mapper.get_voice_path(nm)
            voice_paths.append(vp)
            print(f"Speaker {sp} ('{nm}') -> {os.path.basename(vp)}")
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return

    # Optional preprocess refs
    tmp_voice_dir = ""
    if args.resample_voices:
        voice_paths, tmp_voice_dir = maybe_preprocess_voices(
            voice_paths, target_sr=args.sample_rate, max_seconds=args.max_ref_seconds, enable=True
        )
        if tmp_voice_dir:
            print(f"Preprocessed voices in: {tmp_voice_dir}")

    # Build full text
    full_script = "\n".join(scripts).replace("’", "'")

    # Processor
    print(f"Loading processor from {args.model_path}")
    processor = VibeVoiceProcessor.from_pretrained(args.model_path)

    # Dtype
    if args.dtype == "fp16":
        chosen_dtype = torch.float16
    elif args.dtype == "bf16":
        chosen_dtype = torch.bfloat16
    else:
        chosen_dtype = torch.float16 if prefer_fp16_on_this_gpu() else torch.bfloat16

    # Attention
    use_flash = (not args.force_sdpa) and fa2_available()
    attn_impl = "flash_attention_2" if use_flash else "sdpa"

    # Device map for offloading
    device_map = args.device_map

    print(f"Using dtype={chosen_dtype}, attn={attn_impl}, device_map={device_map}")
    try:
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            args.model_path,
            torch_dtype=chosen_dtype,
            device_map=device_map,
            attn_implementation=attn_impl,
        )
    except Exception as e:
        print(f"[WARN] Initial load failed ({type(e).__name__}): {e}\nFalling back to SDPA.")
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            args.model_path,
            torch_dtype=chosen_dtype,
            device_map=device_map,
            attn_implementation="sdpa",
        )

    model.eval()
    model.set_ddpm_inference_steps(num_steps=int(args.num_steps))
    try:
        model.model.diffusion_head.config.ddpm_batch_mul = int(args.ddpm_batch_mul)
        print(f"Set ddpm_batch_mul={args.ddpm_batch_mul}")
    except Exception:
        pass
    if hasattr(model.model, "language_model"):
        print(f"LM attention: {model.model.language_model.config._attn_implementation}")

    # -------- Robust, OOM-resistant generation pipeline --------
    t_start = time.time()
    audio_pieces: List[torch.Tensor] = []

    steps_sequence = args.retry_steps if args.retry_steps else [args.num_steps]
    force_sdpa_used = False
    dtype_downgraded = False

    for attempt_steps in steps_sequence:
        # Apply steps for this attempt
        try:
            model.set_ddpm_inference_steps(num_steps=int(attempt_steps))
        except Exception:
            pass

        # Optionally force SDPA on retry
        if attempt_steps != steps_sequence[0] and args.force_sdpa_on_retry and not force_sdpa_used:
            try:
                print("[Backoff] Forcing SDPA attention for retry.")
                model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    args.model_path,
                    torch_dtype=chosen_dtype,
                    device_map=device_map,
                    attn_implementation="sdpa",
                )
                model.eval()
                model.set_ddpm_inference_steps(num_steps=int(attempt_steps))
                try:
                    model.model.diffusion_head.config.ddpm_batch_mul = int(args.ddpm_batch_mul)
                except Exception:
                    pass
                force_sdpa_used = True
            except Exception as e:
                print(f"[Backoff] Forcing SDPA failed: {e}. Continuing with current model.")

        try:
            audio_pieces = safe_generate_with_backoff(
                model=model,
                processor=processor,
                full_text=full_script,
                voice_paths=voice_paths,
                cfg_scale=args.cfg_scale,
                sample_rate=args.sample_rate,
                device_map=device_map,
                initial_max_chars=args.max_chars_per_call,
                min_max_chars=args.min_chars_per_call,
            )
            break  # success
        except torch.cuda.OutOfMemoryError as e:
            print(f"[OOM] after steps={attempt_steps}: {e}")

            # Try dtype downgrade if allowed and we're on bf16
            if args.downgrade_dtype_on_retry and chosen_dtype == torch.bfloat16 and not dtype_downgraded:
                try:
                    print("[Backoff] Downgrading dtype to fp16 and retrying.")
                    chosen_dtype = torch.float16
                    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        args.model_path,
                        torch_dtype=chosen_dtype,
                        device_map=device_map,
                        attn_implementation="sdpa",  # safest
                    )
                    model.eval()
                    model.set_ddpm_inference_steps(num_steps=int(attempt_steps))
                    try:
                        model.model.diffusion_head.config.ddpm_batch_mul = int(args.ddpm_batch_mul)
                    except Exception:
                        pass
                    dtype_downgraded = True
                    continue
                except Exception as e2:
                    print(f"[Backoff] Dtype downgrade failed: {e2}")

            # else move to next attempt with fewer steps
            continue
        except Exception as e:
            print(f"[ERROR] Generation attempt with steps={attempt_steps} failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

    if not audio_pieces:
        print(
            "[FATAL] Could not generate audio even after backoff. Try shorter references, --resample_voices, or smaller text.")
        return

    # Concatenate and save
    final_audio = concat_audio(audio_pieces)
    if final_audio is None:
        print("[ERROR] No audio was produced.");
        return

    gen_time = time.time() - t_start
    duration_sec = final_audio.shape[-1] / float(args.sample_rate)
    rtf = gen_time / duration_sec if duration_sec > 0 else float("inf")
    print(f"Total audio duration: {duration_sec:.2f}s | Wall time: {gen_time:.2f}s | RTF: {rtf:.2f}x")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    txt_filename = os.path.splitext(os.path.basename(args.txt_path))[0]
    out_path = os.path.join(args.output_dir, f"{txt_filename}_generated.wav")
    processor.save_audio(final_audio, output_path=out_path)
    print(f"Saved output to {out_path}")

    # Metrics (tokens from a small dummy pack; optional)
    # Skipped: token counts across chunked calls aren’t trivially comparable.

    # Cleanup
    if 'tmp_voice_dir' in locals() and tmp_voice_dir and os.path.isdir(tmp_voice_dir):
        shutil.rmtree(tmp_voice_dir, ignore_errors=True)

    print("\n" + "=" * 50)
    print("GENERATION SUMMARY (OOM-resistant)")
    print("=" * 50)
    print(f"Input: {args.txt_path}")
    print(f"Output: {out_path}")
    print(f"Steps tried: {args.retry_steps}")
    print(f"Dtype final: {chosen_dtype}")
    print(f"Attn forced SDPA: {force_sdpa_used}")
    print(f"Dtype downgraded: {dtype_downgraded}")
    print(f"Total duration: {duration_sec:.2f}s | RTF: {rtf:.2f}x")
    print("=" * 50)


if __name__ == "__main__":
    main()
