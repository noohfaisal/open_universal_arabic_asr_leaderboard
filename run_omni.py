
import sys
import os
sys.path.append(os.getcwd())
try:
    from models.omnilingual_asr import run_omnilingual
    from eval import calculate_wer
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Omnilingual ASR Inference")
    parser.add_argument("--model_id", type=str, default="omniASR_LLM_1B", help="Model ID")
    parser.add_argument("--data_manifest", type=str, required=True, help="Path to input manifest")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to audio files")
    parser.add_argument("--output_manifest", type=str, default="results/omnilingual_asr/output.json", help="Path to output manifest")
    args = parser.parse_args()

    # Create the test file if it doesn't exist (using the available script)
    if not os.path.exists("datasets/test_audio.wav"):
         os.system(".venv/bin/python scripts/create_dummy_wav.py datasets/test_audio.wav")
         
    if not os.path.exists("datasets/test_manifest.json"):
         with open("datasets/test_manifest.json", "w") as f:
             f.write('{"audio_filepath": "{data_folder}/test_audio.wav", "duration": 1.0, "text": "تجربة"}\n')

    os.makedirs(os.path.dirname(args.output_manifest), exist_ok=True)
    
    print(f"Running Inference with {args.model_id}...")
    try:
        run_omnilingual(args.model_id, args.data_manifest, args.data_folder, args.output_manifest)
        print("Inference Complete.")
    except Exception as e:
        print(f"Inference Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Running Eval...")
    calculate_wer(args.output_manifest)
    print("Eval Complete.")

if __name__ == "__main__":
    main()
