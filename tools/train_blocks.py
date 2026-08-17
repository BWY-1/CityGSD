#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Train all blocks sequentially, forwarding remaining arguments to train.py"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model_root", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    with open(args.manifest, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if args.train_args and args.train_args[0] == "--":
        args.train_args = args.train_args[1:]
    forbidden = {"--block_manifest", "--block_id", "--model_path"}
    conflicts = forbidden.intersection(args.train_args)
    if conflicts:
        parser.error(f"These arguments are managed by train_blocks.py: {sorted(conflicts)}")

    for block in manifest["blocks"]:
        block_id = block["id"]
        model_path = os.path.join(args.model_root, block_id)
        final_iteration = None
        for index, value in enumerate(args.train_args):
            if value == "--iterations" and index + 1 < len(args.train_args):
                final_iteration = args.train_args[index + 1]
        if final_iteration:
            final_dir = os.path.join(model_path, "point_cloud", f"iteration_{final_iteration}")
            if os.path.isdir(final_dir):
                print(f"Skipping completed block {block_id}: {final_dir}")
                continue
        command = [
            args.python,
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "train.py"),
            *args.train_args,
            "--block_manifest", os.path.abspath(args.manifest),
            "--block_id", block_id,
            "--model_path", model_path,
        ]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
