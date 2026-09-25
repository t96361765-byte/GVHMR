"""Create/edit reusable constraint files for controlled ablation experiments."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hmr4d.utils.mushroom_config import (
    PRESETS,
    merge_constraints,
    read_constraints,
    resolve_constraints,
    set_parameter,
    validate_constraints,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Existing constraint file to modify; omitted uses current defaults")
    parser.add_argument("--preset", choices=PRESETS, default="default")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=JSON_VALUE",
        help="For example hands.weights.orientation=0 or legs.enabled=false; repeatable",
    )
    parser.add_argument("--output", help="Write validated JSON here; omitted prints it")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        cfg = resolve_constraints(overrides=read_constraints(args.input) if args.input else None)
        merge_constraints(cfg, PRESETS[args.preset])
        for item in args.set:
            if "=" not in item:
                raise ValueError("--set requires PATH=JSON_VALUE")
            path, value = item.split("=", 1)
            set_parameter(cfg, path, json.loads(value))
        validate_constraints(cfg)
        text = json.dumps(cfg, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            output = Path(args.output)
            if output.exists() and not args.overwrite:
                raise ValueError(f"File exists: {output}; use --overwrite")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
            print(output.resolve())
        else:
            print(text, end="")
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
