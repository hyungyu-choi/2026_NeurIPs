#!/usr/bin/env python3
"""
Patch all training scripts to add --dataset argument support.

Run from the project root:
    python3 patch_add_dataset_arg.py

This will modify the following files in-place:
    - train.py
    - train_hyperbolic.py
    - train_hyperbolic_entail_and_pl.py
    - train_hyperbolic_entail_and_pl_mat.py
    - train_hyperbolic_pl_only.py

Changes:
    1. Adds --dataset argument (choices: cholec80, lemon)
    2. Changes --data_root default to None (resolved from dataset config in get_loader)

Make sure you've also replaced utils/data_utils.py with the updated version
that includes DATASET_CONFIGS and get_dataset_config().
"""
import os
import re
import sys


DATASET_ARG_LINES = '''\
    parser.add_argument("--dataset", type=str, default="cholec80",
                        choices=["cholec80", "lemon"],
                        help="Dataset to use for training (default: cholec80).")
'''

SCRIPTS = [
    "train.py",
    "train_hyperbolic.py",
    "train_hyperbolic_entail_and_pl.py",
    "train_hyperbolic_entail_and_pl_mat.py",
    "train_hyperbolic_pl_only.py",
]


def patch_file(filepath: str) -> bool:
    """Patch a single training script. Returns True if modified."""
    with open(filepath, 'r') as f:
        content = f.read()

    original = content

    # -- 1. Add --dataset argument after "# Dataset" comment --
    if '--dataset' not in content:
        pattern = r'([ \t]*# Dataset\n)'
        replacement = r'\g<1>' + DATASET_ARG_LINES
        content = re.sub(pattern, replacement, content, count=1)

    # -- 2. Change --data_root default to None --
    # Pattern A: with help text on next line
    content = re.sub(
        r'parser\.add_argument\("--data_root",\s*type=str,\s*\n'
        r'\s*default="[^"]*cholec80[^"]*",\s*\n'
        r'\s*help="[^"]*"\)',
        'parser.add_argument("--data_root", type=str, default=None,\n'
        '                        help="Training data root. If None, uses default for --dataset.")',
        content,
        count=1,
    )

    # Pattern B: without help text (just default on next line, closing paren)
    content = re.sub(
        r'parser\.add_argument\("--data_root",\s*type=str,\s*\n'
        r'\s*default="[^"]*cholec80[^"]*"\)',
        'parser.add_argument("--data_root", type=str, default=None,\n'
        '                        help="Training data root. If None, uses default for --dataset.")',
        content,
        count=1,
    )

    if content == original:
        return False

    with open(filepath, 'w') as f:
        f.write(content)
    return True


def main():
    modified = []
    skipped = []
    not_found = []

    print("Patching training scripts to add --dataset support...")
    print()

    for script in SCRIPTS:
        if not os.path.isfile(script):
            not_found.append(script)
            print(f"  [NOT FOUND] {script}")
            continue

        if patch_file(script):
            modified.append(script)
            print(f"  [PATCHED]   {script}")
        else:
            skipped.append(script)
            print(f"  [SKIP]      {script} (already patched or pattern not found)")

    print()
    print(f"Results: {len(modified)} patched, {len(skipped)} skipped, {len(not_found)} not found")

    if not_found:
        print(f"  Not found: {', '.join(not_found)}")
        print("  -> Make sure you run this from the project root directory.")

    if modified:
        print()
        print("Done! You can now use --dataset lemon (or cholec80) in training commands.")
        print()
        print("Examples:")
        print("  # LEMON dataset (uses default path ../Dataset/LEMON_frames)")
        print("  python3 train_hyperbolic_entail_and_pl_mat.py --dataset lemon --name lemon-mat ...")
        print()
        print("  # LEMON with custom path")
        print("  python3 train_hyperbolic_entail_and_pl_mat.py --dataset lemon --data_root /my/path ...")
        print()
        print("  # Cholec80 (default, same as before)")
        print("  python3 train_hyperbolic_entail_and_pl_mat.py --dataset cholec80 --name cholec-mat ...")


if __name__ == "__main__":
    main()