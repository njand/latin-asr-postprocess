"""CLI entrypoint for running ITN inference on raw Latin text."""

import argparse
from latin_itn import LatinITNPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ITN inference on raw Latin text.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path or HF ID to fine-tuned model checkpoint.",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Raw input text to process.",
    )
    parser.add_argument(
        "--window_tokens",
        type=int,
        default=120,
        help="Sliding window size in tokens (default: 120).",
    )
    parser.add_argument(
        "--stride_tokens",
        type=int,
        default=60,
        help="Stride step size in tokens (default: 60).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run model on ('cuda', 'cpu'). Auto-detected if omitted.",
    )

    args = parser.parse_args()

    predictor = LatinITNPredictor.from_pretrained(
        args.model_path, device=args.device
    )

    formatted = predictor.predict(
        args.text,
        window_tokens=args.window_tokens,
        stride_tokens=args.stride_tokens,
    )

    print("\nFormatted ITN Output:")
    print(formatted)


if __name__ == "__main__":
    main()