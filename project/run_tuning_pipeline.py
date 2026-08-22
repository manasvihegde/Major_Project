import subprocess
import sys

def run_command(command_list):
    print(f"🚀 Running: {' '.join(command_list)}")
    result = subprocess.run(command_list)
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    print("==================================================")
    print("🌟 STARTING END-TO-END XAI-GUIDED TUNING PIPELINE 🌟")
    print("==================================================")
    
    # 1. Train the LoRA adapter
    run_command([sys.executable, "-m", "tuning.train_lora"])
    
    # 2. Run comparative report metrics script
    run_command([sys.executable, "-m", "tuning.compare_runs_report"])
    
    print("\n" + "="*50)
    print("🎉 End-to-End Tuning & Reporting Pipeline Complete!")
    print("="*50)

if __name__ == "__main__":
    main()