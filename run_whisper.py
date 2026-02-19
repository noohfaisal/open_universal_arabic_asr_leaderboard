
import sys
import os
import argparse

# Ensure we can import from models/
sys.path.append(os.getcwd())

try:
    from models.whisper import run_whisper
    # Import eval for optional on-the-fly eval, though we usually run leaderboard_score separately
    from eval import calculate_wer 
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run Whisper ASR Inference")
    parser.add_argument("--model_id", type=str, default="openai/whisper-large-v3", help="Model ID")
    parser.add_argument("--data_manifest", type=str, required=True, help="Path to input manifest")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to audio files")
    parser.add_argument("--output_manifest", type=str, default="results/whisper/output.json", help="Path to output manifest")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_manifest), exist_ok=True)
    
    print(f"Running Inference with {args.model_id}...")
    print(f"Input: {args.data_manifest}")
    print(f"Output: {args.output_manifest}")
    
    try:
        run_whisper(args.model_id, args.data_manifest, args.data_folder, args.output_manifest)
        print("Inference Complete.")
    except Exception as e:
        print(f"Inference Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Optional: Run quick eval
    try:
        print("Running Quick Eval...")
        calculate_wer(args.output_manifest)
    except Exception as e:
        print(f"Eval Failed: {e}")

if __name__ == "__main__":
    main()
