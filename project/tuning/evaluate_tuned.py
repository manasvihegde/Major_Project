import argparse
from comprehensive_evaluator import run_comprehensive_evaluation

def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned LoRA checkpoint.")
    parser.add_argument("--adapter_path", type=str, default="checkpoints/lora_v1")
    parser.add_argument("--run_name", type=str, default="gpt2+lora_v1")
    parser.add_argument("--output_csv", type=str, default="data/outputs/truth_score_report_tuned.csv")
    args = parser.parse_args()

    run_comprehensive_evaluation(
        adapter_path=args.adapter_path,
        run_name=args.run_name,
        output_csv=args.output_csv
    )

if __name__ == "__main__":
    main()