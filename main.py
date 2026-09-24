"""Score candidate actions with Cua-S1 without interacting with the screen."""

import argparse
import json
import os
from pathlib import Path

# Set before torch initializes so unsupported MPS operations can fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import transformers
from cua_s1.four_b import FourBModel, Option, OptionProbability
from huggingface_hub import snapshot_download

transformers.utils.logging.set_verbosity_error()


def parse_args() -> argparse.Namespace:
    """Parse CLI options and reject missing screenshot files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check dependencies and the selected device without downloading weights",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="Login screen image; use the text example if omitted",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default="mps",
        help="Inference device: CPU, Apple GPU, or NVIDIA GPU (default: mps)",
    )
    args = parser.parse_args()
    if args.screenshot is not None and not args.screenshot.is_file():
        parser.error(f"Image not found: {args.screenshot}")

    return args


def check_environment(device: str) -> str:
    """Verify device computation and return its supported inference dtype.

    Uses float32 for CPU and CUDA without bfloat16 support; otherwise bfloat16.
    Exits if the selected device is unavailable or returns incorrect results.
    """
    from transformers import (  # noqa: F401
        Qwen3_5ForCausalLM,
        Qwen3_5ForConditionalGeneration,
    )

    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable. Use --device cpu instead.")
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Use --device cpu instead.")

    dtype = "bfloat16"
    if device == "cpu" or (device == "cuda" and not torch.cuda.is_bf16_supported()):
        dtype = "float32"

    # Verify operations with the inference dtype, beyond the availability flag.
    probe = torch.ones(2, device=device, dtype=getattr(torch, dtype))
    if (probe + probe).sum().item() != 4:
        raise SystemExit(f"{device} computation check failed.")

    print(f"{device} / {dtype}: OK")
    return dtype


def create_model(modality: str, device: str, dtype: str) -> FourBModel:
    """Download the text or multimodal adapter and configure lazy model loading.

    Pass a device and dtype validated by check_environment(). The base weights
    load on the first forward call, or explicitly through model.load().
    """
    print(
        f"Loading model ({modality}). Weights will be downloaded on the first run.",
        flush=True,
    )

    adapter_root = snapshot_download(
        repo_id="cua-ai/cua-s1-4b-0.2",
        allow_patterns=[f"{modality}/*"],
    )

    return FourBModel(
        base_model="Qwen/Qwen3.5-4B",
        lora_adapter_path=Path(adapter_root) / modality,
        modality=modality,
        device=device,
        dtype=dtype,
    )


def score_actions(
    model: FourBModel, sample: dict, screenshot: Path | None = None
) -> list[OptionProbability]:
    """Score the supplied GUI candidates without executing actions.

    Args:
        model: Cua model configured for multimodal input when a screenshot is
            supplied, or text input otherwise. Weights load on first inference.
        sample: Model inputs with app, task_family, goal, ax_tree, and options.
            Each option is a dictionary accepted by Option.
        screenshot: Image replacing the sample's screen text, or None for text.

    Returns:
        Candidate actions with their predicted probabilities, without sorting.
    """
    inputs = {
        **sample,
        "options": [Option(**option) for option in sample["options"]],
        "ax_tree": None if screenshot is not None else sample["ax_tree"],
    }
    return model.forward(**inputs, screenshot=screenshot)


def main() -> None:
    """Check the device or score the login sample without executing GUI actions."""
    args = parse_args()
    dtype = check_environment(args.device)
    if args.check:
        print("No model weights were downloaded.")
        return

    modality = "multimodal" if args.screenshot else "text"
    model = create_model(modality, args.device, dtype)
    sample_path = Path(__file__).parent / "tasks/login/environment/app/login.json"
    sample = json.loads(sample_path.read_text())
    results = score_actions(model, sample, args.screenshot)
    for result in sorted(results, key=lambda item: item.probability, reverse=True):
        print(f"{result.label}: {result.action} {result.probability:.3f}")


if __name__ == "__main__":
    main()
