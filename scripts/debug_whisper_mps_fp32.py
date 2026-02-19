
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sys

def run_debug():
    model_id = "openai/whisper-large-v3"
    # Target the problematic file found at line 2183 (new)
    audio_file = "datasets/sada_segments/6k_v2ms_SBA_6_segmented_1-seg_39_380-44_010.wav"
    
    print(f"Testing on {audio_file}")
    
    # Test MPS + FP32
    print("\n--- Test 3: MPS + FP32 ---")
    if torch.backends.mps.is_available():
        try:
            device = "mps"
            torch_dtype = torch.float32 
            
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            model.to(device)
            
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=30,
                batch_size=1,
                return_timestamps=False,
                torch_dtype=torch_dtype,
                device=device,
            )
            
            # Added timeout mechanism or just print checking
            print("Running inference...")
            result = pipe(audio_file, generate_kwargs={"language": "ar", "task": "transcribe"})
            print(f"Result: {result['text']}")
        except Exception as e:
            print(f"Failed: {e}")
    else:
        print("MPS not available.")

if __name__ == "__main__":
    run_debug()
