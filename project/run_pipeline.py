"""
run_pipeline.py

Runs the complete pipeline (Week 2 + Week 3) in one command.

Pipeline:

0. curate_dataset.py (optional)
1. run_parsers.py
2. comparative_logger.py
3. batch_runner.py
4. evaluation_runner.py

----- Week 3 -----
5. interpretability.tensor_normalizer
6. interpretability.sensitivity_variance
7. interpretability.token_heatmap
8. interpretability.entity_importance_review
9. interpretability.plausibility_trap

Automatically captures run_id from comparative_logger and forwards it
to every script that requires it.
"""

import argparse
import re
import subprocess
import sys

RUN_ID_PATTERN = re.compile(r"Started run_id=(\d+)")


def run_step(cmd, capture=False):
    """
    Execute a subprocess.

    If capture=True:
        - stream output live
        - return captured stdout
    """

    print("\n" + "=" * 70)
    print("RUNNING:")
    print(" ".join(cmd))
    print("=" * 70)

    if not capture:
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"\nStep failed ({result.returncode})")
            sys.exit(result.returncode)

        return None

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output = []

    for line in process.stdout:
        print(line, end="")
        output.append(line)

    process.wait()

    if process.returncode != 0:
        print(f"\nStep failed ({process.returncode})")
        sys.exit(process.returncode)

    return "".join(output)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--curate",
        action="store_true",
        help="Rebuild curated dataset."
    )

    parser.add_argument(
        "--skip_parsing",
        action="store_true",
        help="Skip run_parsers.py"
    )

    parser.add_argument(
        "--num_questions",
        type=int,
        default=20
    )

    parser.add_argument(
        "--model",
        default="gpt2"
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50
    )

    parser.add_argument(
        "--top_n",
        type=int,
        default=10
    )

    args = parser.parse_args()

    python = sys.executable

    ###########################################################
    # Step 0
    ###########################################################

    if args.curate:
        run_step([python, "curate_dataset.py"])
    else:
        print("\nSkipping curate_dataset.py")

    ###########################################################
    # Step 1
    ###########################################################

    if not args.skip_parsing:
        run_step([python, "run_parsers.py"])
    else:
        print("\nSkipping run_parsers.py")

    ###########################################################
    # Step 2
    ###########################################################

    output = run_step(
        [
            python,
            "comparative_logger.py",
            "--num_questions",
            str(args.num_questions),
            "--model",
            args.model,
            "--max_new_tokens",
            str(args.max_new_tokens),
        ],
        capture=True,
    )

    match = RUN_ID_PATTERN.search(output)

    if not match:
        print("Could not determine run_id.")
        sys.exit(1)

    run_id = match.group(1)

    print(f"\nCaptured run_id = {run_id}")

    ###########################################################
    # Step 3
    ###########################################################

    run_step(
        [
            python,
            "batch_runner.py",
            "--run_id",
            run_id,
        ]
    )

    ###########################################################
    # Step 4
    ###########################################################

    run_step(
        [
            python,
            "evaluation_runner.py",
            "--run_id",
            run_id,
            "--top_n",
            str(args.top_n),
        ]
    )

    ###########################################################
    # Week 3
    ###########################################################

    print("\n")
    print("=" * 70)
    print("STARTING WEEK 3 INTERPRETABILITY")
    print("=" * 70)

    ###########################################################
    # Tensor Normalizer
    ###########################################################

    run_step(
        [
            python,
            "-m",
            "interpretability.tensor_normalizer",
        ]
    )

    ###########################################################
    # Sensitivity Variance
    ###########################################################

    run_step(
        [
            python,
            "-m",
            "interpretability.sensitivity_variance",
        ]
    )

    ###########################################################
    # Token Heatmap
    ###########################################################

    run_step(
        [
            python,
            "-m",
            "interpretability.token_heatmap",
        ]
    )

    ###########################################################
    # Entity Importance Review
    ###########################################################

    run_step(
        [
            python,
            "-m",
            "interpretability.entity_importance_review",
            "--run_id",
            run_id,
        ]
    )

    ###########################################################
    # Plausibility Trap
    ###########################################################

    run_step(
        [
            python,
            "-m",
            "interpretability.plausibility_trap",
            "--run_id",
            run_id,
        ]
    )

    ###########################################################
    # Finished
    ###########################################################

    print("\n")
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Run ID : {run_id}")
    print("=" * 70)


if __name__ == "__main__":
    main()