
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sys

def run_debug():
    model_id = "openai/whisper-large-v3"
    audio_file = "datasets/arabic_diacritized_audio/audio_0.flac"
    
    print(f"Testing on {audio_file}")
    
    # 1. Test CPU + FP32 (Baseline)
    print("\n--- Test 1: CPU + FP32 ---")
    try:
        device = "cpu"
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
        
        result = pipe(audio_file, generate_kwargs={"language": "ar", "task": "transcribe"})
        print(f"Result: {result['text']}")
        
        del model
        del pipe
    except Exception as e:
        print(f"Failed: {e}")

    # 2. Test MPS + FP16 (Current Setup)
    print("\n--- Test 2: MPS + FP16 ---")
    if torch.backends.mps.is_available():
        try:
            device = "mps"
            torch_dtype = torch.float16
            
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
            
            result = pipe(audio_file, generate_kwargs={"language": "ar", "task": "transcribe"})
            print(f"Result: {result['text']}")
        except Exception as e:
            print(f"Failed: {e}")
    else:
        print("MPS not available.")

if __name__ == "__main__":
    run_debug()
