import argparse
import os
import re
import traceback
from typing import List, Tuple, Union, Dict, Any
import time
import torch
import difflib

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class VoiceMapper:
    """Maps speaker names to voice file paths and provides utilities to inspect availability."""

    def __init__(self):
        self.setup_voice_presets()

        # Create normalized aliases from filenames to improve matching.
        # Example: "Ava_West" -> alias "Ava"; "John-Doe" -> alias "Doe"
        new_dict = {}
        for name, path in self.voice_presets.items():
            alias = name
            if "_" in alias:
                alias = alias.split("_")[0]
                new_dict[alias] = path
            # add a second alias using the tail after '-' if present
            alias2 = name
            if "-" in alias2:
                alias2 = alias2.split("-")[-1]
                new_dict[alias2] = path

        # Merge aliases into the presets (later keys may override earlier ones; that's OK)
        self.voice_presets.update(new_dict)

    def setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory for .wav files."""
        voices_dir = os.path.join(os.path.dirname(__file__), "voices")

        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            self.voice_presets: Dict[str, str] = {}
            self.available_voices: Dict[str, str] = {}
            return

        # Scan for all WAV files in the voices directory
        self.voice_presets = {}

        wav_files = [
            f
            for f in os.listdir(voices_dir)
            if f.lower().endswith(".wav") and os.path.isfile(os.path.join(voices_dir, f))
        ]

        for wav_file in wav_files:
            name = os.path.splitext(wav_file)[0]
            full_path = os.path.join(voices_dir, wav_file)
            self.voice_presets[name] = full_path

        # Sort alphabetically for a nicer listing
        self.voice_presets = dict(sorted(self.voice_presets.items()))

        # Filter out voices that don't exist (redundant, but safe)
        self.available_voices = {
            name: path for name, path in self.voice_presets.items() if os.path.exists(path)
        }

        print(f"Found {len(self.available_voices)} voice files in {voices_dir}")
        if self.available_voices:
            print(f"Available voices: {', '.join(self.available_voices.keys())}")

    def list_speakers(self) -> List[str]:
        """Return all recognizable speaker names (including normalized aliases)."""
        return sorted(self.voice_presets.keys())

    def best_match(self, speaker_name: str) -> Tuple[str, str, str]:
        """
        Return (matched_key, path, match_type) where match_type ∈
        {'exact', 'partial', 'fallback', 'none'}.

        - 'exact': key exists exactly as provided
        - 'partial': case-insensitive substring match (both directions)
        - 'fallback': no match found, would fallback to the first available voice
        - 'none': there are no voices at all
        """
        if not self.voice_presets:
            return "", "", "none"

        # exact
        if speaker_name in self.voice_presets:
            return speaker_name, self.voice_presets[speaker_name], "exact"

        # partial (case-insensitive, substring either way)
        speaker_lower = speaker_name.lower()
        for preset_name, path in self.voice_presets.items():
            if preset_name.lower() in speaker_lower or speaker_lower in preset_name.lower():
                return preset_name, path, "partial"

        # fallback
        default_key = next(iter(self.voice_presets))
        return default_key, self.voice_presets[default_key], "fallback"

    def get_voice_path(self, speaker_name: str) -> str:
        """Get voice file path for a given speaker name (with fallback)."""
        matched_key, path, match_type = self.best_match(speaker_name)
        if match_type == "none":
            raise RuntimeError(
                "No voice .wav files were found. Please add .wav files to the 'voices' directory."
            )
        if match_type == "fallback":
            print(
                f"Warning: No voice preset found for '{speaker_name}', using default voice: {path}"
            )
        return path


def parse_txt_script(txt_content: str) -> Tuple[List[str], List[str]]:
    """
    Parse txt script content and extract speakers and their text.
    Expected format per line: 'Speaker X: text...'
    Returns: (scripts, speaker_numbers)
    """
    lines = txt_content.strip().split("\n")
    scripts: List[str] = []
    speaker_numbers: List[str] = []

    speaker_pattern = r"^Speaker\s+(\d+):\s*(.*)$"

    current_speaker = None
    current_text = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(speaker_pattern, line, re.IGNORECASE)
        if match:
            # flush previous
            if current_speaker and current_text:
                scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
                speaker_numbers.append(current_speaker)

            current_speaker = match.group(1).strip()
            current_text = match.group(2).strip()
        else:
            # continuation
            if current_text:
                current_text += " " + line
            else:
                current_text = line

    # flush last
    if current_speaker and current_text:
        scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
        speaker_numbers.append(current_speaker)

    return scripts, speaker_numbers


def print_available_voices(vm: VoiceMapper):
    names = vm.list_speakers()
    print(f"\nAvailable speaker names ({len(names)}):")
    for n in names:
        print(f"  - {n}")
    print()


def check_speaker_names(vm: VoiceMapper, speaker_names: List[str]) -> int:
    """
    Validate the given names. Returns 0 if all are 'exact' or 'partial',
    otherwise prints details and returns 1. If no voices exist, returns 2.
    """
    if not vm.list_speakers():
        print("No voice .wav files were found in the 'voices' directory.")
        print("Please add at least one .wav file and try again.")
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
            # handled above, but keep for completeness
            exit_code = 2
            print(f"[ERROR] No voices are available to match '{name}'.")
        else:
            label = "OK (exact)" if match_type == "exact" else f"OK (partial → '{matched_key}')"
            print(f"[{label}] '{name}' → {os.path.basename(path)}")
    print()
    return exit_code


def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice Processor TXT Input Test")
    parser.add_argument(
        "--model_path",
        type=str,
        default="microsoft/VibeVoice-1.5b",
        help="Path to the HuggingFace model directory",
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
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for tensor tests",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=2,
        help="CFG (Classifier-Free Guidance) scale for generation (default: 2)",
    )
    # NEW: voice inspection utilities
    parser.add_argument(
        "--list_voices",
        action="store_true",
        help="List all available speaker names and exit",
    )
    parser.add_argument(
        "--check_speaker_names",
        action="store_true",
        help="Validate provided --speaker_names and exit with non-zero if any are unavailable",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize voice mapper
    voice_mapper = VoiceMapper()

    # Utility modes (exit early)
    if args.list_voices:
        print_available_voices(voice_mapper)
        return

    if args.check_speaker_names:
        speaker_names_list = (
            args.speaker_names if isinstance(args.speaker_names, list) else [args.speaker_names]
        )
        rc = check_speaker_names(voice_mapper, speaker_names_list)
        if rc != 0:
            print("One or more speaker names were not found or no voices exist. See details above.")
        return

    # Normal generation flow below

    # Check if txt file exists
    if not os.path.exists(args.txt_path):
        print(f"Error: txt file not found: {args.txt_path}")
        return

    # Read and parse txt file
    print(f"Reading script from: {args.txt_path}")
    with open(args.txt_path, "r", encoding="utf-8") as f:
        txt_content = f.read()

    # Parse the txt content to get speaker numbers
    scripts, speaker_numbers = parse_txt_script(txt_content)

    if not scripts:
        print("Error: No valid speaker scripts found in the txt file")
        return

    print(f"Found {len(scripts)} speaker segments:")
    for i, (script, speaker_num) in enumerate(zip(scripts, speaker_numbers)):
        print(f"  {i + 1}. Speaker {speaker_num}")
        print(f"     Text preview: {script[:100]}...")

    # Map speaker numbers to provided speaker names
    speaker_name_mapping: Dict[str, str] = {}
    speaker_names_list = args.speaker_names if isinstance(args.speaker_names, list) else [args.speaker_names]
    for i, name in enumerate(speaker_names_list, 1):
        speaker_name_mapping[str(i)] = name

    print(f"\nSpeaker mapping:")
    for speaker_num in sorted(set(speaker_numbers), key=int):
        mapped_name = speaker_name_mapping.get(speaker_num, f"Speaker {speaker_num}")
        print(f"  Speaker {speaker_num} -> {mapped_name}")

    # Map speakers to voice files using the provided speaker names
    voice_samples: List[str] = []
    actual_speakers: List[str] = []

    # Get unique speaker numbers in order of first appearance
    unique_speaker_numbers: List[str] = []
    seen = set()
    for speaker_num in speaker_numbers:
        if speaker_num not in seen:
            unique_speaker_numbers.append(speaker_num)
            seen.add(speaker_num)

    try:
        for speaker_num in unique_speaker_numbers:
            speaker_name = speaker_name_mapping.get(speaker_num, f"Speaker {speaker_num}")
            voice_path = voice_mapper.get_voice_path(speaker_name)
            voice_samples.append(voice_path)
            actual_speakers.append(speaker_name)
            print(f"Speaker {speaker_num} ('{speaker_name}') -> Voice: {os.path.basename(voice_path)}")
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return

    # Prepare data for model
    full_script = "\n".join(scripts)
    full_script = full_script.replace("’", "'")

    # Load processor
    print(f"Loading processor & model from {args.model_path}")
    processor = VibeVoiceProcessor.from_pretrained(args.model_path)

    # Load model
    try:
        # Keep using CUDA map here as in original code; adjust if needed based on args.device
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2",  # flash_attention_2 is recommended
        )
    except Exception as e:
        print(f"[ERROR] : {type(e).__name__}: {e}")
        print(traceback.format_exc())
        print(
            "Error loading the model. Trying to use SDPA. However, note that only flash_attention_2 has been fully "
            "tested, and using SDPA may result in lower audio quality."
        )
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="sdpa",
        )

    model.eval()
    model.set_ddpm_inference_steps(num_steps=10)

    if hasattr(model.model, "language_model"):
        print(f"Language model attention: {model.model.language_model.config._attn_implementation}")

    # Prepare inputs for the model
    inputs = processor(
        text=[full_script],  # Wrap in list for batch processing
        voice_samples=[voice_samples],  # Wrap in list for batch processing
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    print(f"Starting generation with cfg_scale: {args.cfg_scale}")

    # Generate audio
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

    # Calculate audio duration and additional metrics
    audio_duration = 0.0
    rtf = float("inf")
    if hasattr(outputs, "speech_outputs") and outputs.speech_outputs and outputs.speech_outputs[0] is not None:
        sample_rate = 24000  # Assuming 24kHz sample rate
        audio_samples = (
            outputs.speech_outputs[0].shape[-1]
            if len(outputs.speech_outputs[0].shape) > 0
            else len(outputs.speech_outputs[0])
        )
        audio_duration = audio_samples / sample_rate
        rtf = generation_time / audio_duration if audio_duration > 0 else float("inf")

        print(f"Generated audio duration: {audio_duration:.2f} seconds")
        print(f"RTF (Real Time Factor): {rtf:.2f}x")
    else:
        print("No audio output generated")

    # Calculate token metrics
    input_tokens = inputs["input_ids"].shape[1]  # Number of input tokens
    output_tokens = outputs.sequences.shape[1]  # Total tokens (input + generated)
    generated_tokens = output_tokens - input_tokens

    print(f"Prefilling tokens: {input_tokens}")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Total tokens: {output_tokens}")

    # Save output
    txt_filename = os.path.splitext(os.path.basename(args.txt_path))[0]
    output_path = os.path.join(args.output_dir, f"{txt_filename}_generated.wav")
    os.makedirs(args.output_dir, exist_ok=True)

    processor.save_audio(
        outputs.speech_outputs[0],  # First (and only) batch item
        output_path=output_path,
    )
    print(f"Saved output to {output_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("GENERATION SUMMARY")
    print("=" * 50)
    print(f"Input file: {args.txt_path}")
    print(f"Output file: {output_path}")
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


if __name__ == "__main__":
    main()
