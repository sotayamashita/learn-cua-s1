"""Score candidate actions with Cua-S1 without interacting with the screen."""

import argparse
import os
from pathlib import Path

# Set before torch initializes so unsupported MPS operations can fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import transformers
from cua_s1.four_b import FourBModel, Option, OptionProbability
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)


class CuaModel(FourBModel):
    """Use the Transformers dtype argument with Cua's inference implementation."""

    def load(self) -> None:
        """Load the base model and optional LoRA adapter for inference.

        Called automatically on the first forward call. Uses the configured
        modality, device, and dtype, and downloads uncached model files.
        Stores the tokenizer, image processor when needed, and model in
        evaluation mode on this instance.

        Overrides the pinned dependency's loader to replace its deprecated
        torch_dtype argument with dtype.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        if self.modality == "multimodal":
            self._processor = AutoProcessor.from_pretrained(self.base_model)
            model_cls = AutoModelForImageTextToText
        else:
            self._processor = None
            model_cls = AutoModelForCausalLM

        model = model_cls.from_pretrained(
            self.base_model, dtype=getattr(torch, self.dtype), device_map=self.device
        )

        if self.lora_adapter_path:
            model = PeftModel.from_pretrained(model, str(self._resolve_adapter_path()))

        self._model = model.eval()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments and validate the screenshot path.

    Returns:
        Arguments containing the device, check flag, and optional screenshot Path.

    Raises:
        SystemExit: On a help request, invalid arguments, or a screenshot path
            that does not point to a file.
    """
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
    """Check the selected device without downloading model weights.

    Args:
        device: "cpu", "mps", or "cuda".

    Returns:
        The verified dtype name: float32 for CPU and bfloat16 for GPU.
        CUDA devices without bfloat16 support use float32.

    Raises:
        ImportError: If the required Qwen model classes cannot be imported.
        SystemExit: If the device is unavailable or the computation is incorrect.
        RuntimeError: If the selected device cannot run the computation.
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


def create_model(modality: str, device: str, dtype: str) -> CuaModel:
    """Download the selected adapter and configure the model for inference.

    Args:
        modality: "text" for screen text or "multimodal" for screenshots.
        device: "cpu", "mps", or "cuda".
        dtype: Dtype name returned by check_environment.

    Returns:
        A Cua model configured for the selected device with a local adapter path.
        Base model weights load on the first forward call.
    """
    print(
        f"Loading model ({modality}). Weights will be downloaded on the first run.",
        flush=True,
    )

    adapter_root = snapshot_download(
        repo_id="cua-ai/cua-s1-4b-0.2",
        allow_patterns=[f"{modality}/*"],
    )

    return CuaModel(
        base_model="Qwen/Qwen3.5-4B",
        lora_adapter_path=Path(adapter_root) / modality,
        modality=modality,
        device=device,
        dtype=dtype,
    )


def score_actions(model: CuaModel, screenshot: Path | None) -> list[OptionProbability]:
    """Score the candidate actions in a fixed login example.

    The sample goal is to submit a completed login form. The candidates are
    clicks on "Login" and "Login with Google"; no actions are executed.

    Args:
        model: Cua model configured for multimodal input when a screenshot is
            supplied, or text input otherwise. Weights load on first inference.
        screenshot: Path to a completed login form image, or None to use the
            built-in text describing filled fields and the two buttons.

    Returns:
        Candidate actions with their predicted probabilities, without sorting.
    """
    return model.forward(
        options=[
            Option(element_id="login", role="button", label="Login", action="click"),
            Option(
                element_id="login-google",
                role="button",
                label="Login with Google",
                action="click",
            ),
        ],
        app="Browser",
        task_family="login",
        goal="Submit the completed login form.",
        ax_tree=(
            "Login form\nEmail: filled\nPassword: filled\n"
            "Button: Login\nButton: Login with Google"
            if screenshot is None
            else None
        ),
        screenshot=screenshot,
    )


def main() -> None:
    """Run environment checks or score the sample login actions from the CLI.

    With --check, exits after checking the environment without downloading
    weights. Otherwise, selects the input modality and prints candidate
    probabilities in descending order. Does not execute GUI actions.
    """
    args = parse_args()
    dtype = check_environment(args.device)
    if args.check:
        print("No model weights were downloaded.")
        return

    modality = "multimodal" if args.screenshot else "text"
    model = create_model(modality, args.device, dtype)
    results = score_actions(model, args.screenshot)
    for result in sorted(results, key=lambda item: item.probability, reverse=True):
        print(f"{result.label}: {result.action} {result.probability:.3f}")


if __name__ == "__main__":
    main()
