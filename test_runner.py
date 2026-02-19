
import sys
import os
sys.path.append(os.getcwd())
try:
    from models.speechbrain import run_speechbrain
    from eval import calculate_wer
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def main():
    model_id = "speechbrain/asr-wav2vec2-commonvoice-14-ar"
    # model_id = "asafaya/hubert-large-arabic-transcribe" # Larger
    data_manifest = "datasets/test_manifest.json"
    data_folder = "datasets" # Contains test_audio.wav
    output_manifest = "results/test_output.json"
    
    os.makedirs("results", exist_ok=True)
    
    print("Running Model Inference...")
    try:
        run_speechbrain(model_id, data_manifest, data_folder, output_manifest)
        print("Inference Complete.")
    except Exception as e:
        print(f"Inference Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Running Eval...")
    calculate_wer(output_manifest)
    print("Eval Complete.")

if __name__ == "__main__":
    main()
