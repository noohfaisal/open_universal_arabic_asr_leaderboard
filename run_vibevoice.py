
import sys
import os
import argparse
import time
import json
import torch
import numpy as np
from tqdm import tqdm

# Add VibeVoice to path if installed via pip -e, otherwise append manually
try:
    from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
except ImportError:
    # If not installed, try adding the repo path manually
    sys.path.append(os.path.join(os.getcwd(), "models/VibeVoice_repo"))
    from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

def run_vibevoice(model_id, data_manifest, data_folder, output_manifest):
    print(f"Loading VibeVoice model: {model_id}")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # mps prefers float32
    dtype = torch.float32 if device == "mps" else torch.float16
    
    print(f"Using device: {device}, dtype: {dtype}")

    processor = VibeVoiceASRProcessor.from_pretrained(
        model_id,
        language_model_pretrained_name="Qwen/Qwen2.5-7B"
    )

    # Attention implementation: 'sdpa' is standard for latest torch on consumer hardware/Mac
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=None, # Manual device placement
        attn_implementation="sdpa",
        trust_remote_code=True
    ).to(device)
    
    model.eval()
    
    print("Model loaded.")

    # Load data
    with open(data_manifest, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Check for existing results to resume
    existing_results = {}
    if os.path.exists(output_manifest):
        with open(output_manifest, 'r') as f:
            for line in f:
                try:
                    res = json.loads(line)
                    existing_results[res['audio_filepath']] = res
                except:
                    pass
    
    print(f"Found {len(existing_results)} existing results. Resuming...")
    
    # Normalize paths if needed? The user's env var NORMALIZE_PATH might be set.
    should_normalize = os.environ.get("NORMALIZE_PATH", "False") == "True"

    results_file = open(output_manifest, 'a' if len(existing_results) > 0 else 'w', encoding='utf-8')
    
    for item in tqdm(data):
        audio_path = item['audio_filepath']
        
        # Check resume condition
        normalized_audio_path = os.path.normpath(audio_path) if should_normalize else audio_path
        
        # Check against existing keys (which might need normalization too on check)
        # For simplicity, let's assume keys in output match input manifest exactly
        if audio_path in existing_results:
            continue
            
        text_gt = item['text']
        duration = item.get('duration', 0.0)
        
        try:
            start_time = time.time()
            
            # Prepare input
            # VibeVoice processor handles audio loading? Or expects array?
            # The demo script passes file path string list.
            
            # Batch size 1 for simplicity and safety on Mac
            inputs = processor(
                audio=[audio_path],
                sampling_rate=None, # Processor handles loading?
                return_tensors="pt",
                padding=True,
                add_generation_prompt=True
            )
            
            # Move to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            generation_config = {
                "max_new_tokens": 512, # Adjust based on expected length?
                "pad_token_id": processor.pad_id,
                "eos_token_id": processor.tokenizer.eos_token_id,
                "do_sample": False, # Greedy
                "num_beams": 1
            }
            
            with torch.no_grad():
                output_ids = model.generate(**inputs, **generation_config)
            
            # Decode
            input_length = inputs['input_ids'].shape[1]
            generated_ids = output_ids[0, input_length:]
            
            # Post-process eos
            eos_positions = (generated_ids == processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                generated_ids = generated_ids[:eos_positions[0] + 1]
                
            pred_text = processor.decode(generated_ids, skip_special_tokens=True)
            
            # Optionally parse structured output? 
            # The model outputs raw text which is usually the transcript directly if trained on it?
            # Or does it output structured XML-like tags?
            # The demo uses `processor.post_process_transcription(generated_text)`.
            # Let's save both raw and parsed if possible, but for eval, we just need text.
            # Usually strict text.
            
            # Post-process for leaderboard format
            final_pred_text = pred_text
            # Try to extract text if it has tags (e.g. <|startoftext|><|speaker|>...)
            # The processor `post_process_transcription` creates segments.
            try:
                segments = processor.post_process_transcription(pred_text)
                # Join segments text
                final_pred_text = " ".join([seg['text'] for seg in segments])
            except:
                pass # Fallback to raw
            
            inference_time = time.time() - start_time
            

            def sanitize_for_json(obj):
                if isinstance(obj, (np.integer, np.int64, int)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64, float)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray, list, tuple)):
                    return [sanitize_for_json(x) for x in obj]
                elif isinstance(obj, dict):
                    return {k: sanitize_for_json(v) for k, v in obj.items()}
                elif hasattr(obj, 'item'): # Torch scalar
                    return obj.item()
                else:
                    return str(obj)

            result = {
                "audio_filepath": audio_path,
                "text": text_gt,
                "pred_text": final_pred_text,
                "inference_time": inference_time,
                "duration": duration,
                "raw_pred": pred_text
            }
            
            # Sanitize keys and values
            clean_result = sanitize_for_json(result)
            
            try:
                json.dump(clean_result, results_file, ensure_ascii=False)
                results_file.write('\n')
                results_file.flush()
            except TypeError as e:
                print(f"JSON Dump Failed: {e}")
                print(f"Offending Data: {clean_result}")
                # Try to print types
                for k, v in clean_result.items():
                    print(f"{k}: {type(v)}")

            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            # Write error entry? Or skip?
            # Skip for now to let resume handle it later if fixed.
    
    results_file.close()
    print("Inference Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="microsoft/VibeVoice-ASR")
    parser.add_argument("--data_manifest", type=str, required=True)
    parser.add_argument("--data_folder", type=str, required=True) # unused but kept for compatibility
    parser.add_argument("--output_manifest", type=str, required=True)
    args = parser.parse_args()
    
    run_vibevoice(args.model_id, args.data_manifest, args.data_folder, args.output_manifest)
