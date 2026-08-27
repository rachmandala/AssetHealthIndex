from __future__ import annotations

import argparse
import json

from .pipeline import AHIPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Asset Health Index pipeline")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--config", default="config/default.yaml", help="YAML config path")
    parser.add_argument("--output-dir", default="artifacts", help="Output directory")
    args = parser.parse_args()

    pipeline = AHIPipeline(config_path=args.config)
    result = pipeline.run(input_csv=args.input, output_dir=args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
