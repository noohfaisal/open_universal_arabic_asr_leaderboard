import argparse
import json
import torch
import os
from tqdm import tqdm
import time
import numpy as np
import logging
import sys
from transformers import logging as transformers_logging

# Ensure we can import from local modules
sys.path.append(os.getcwd())

try:
    from eval import calculate_wer
except ImportError:
    print("Warning: Could not import calculate_wer from eval.py. Evaluation will be skipped.")
    calculate_wer = None

# Suppress noisy warnings
transformers_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

def run_qwen(model_id, data_manifest, output_manifest, data_folder=None, language=None):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # Qwen-ASR on MPS: float16 often works best for memory/speed
    dtype = torch.float16 if device == "mps" else torch.float32
    
    print(f"Loading Qwen3-ASR model: {model_id} on {device} with {dtype}")
    
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError:
        print("Error: qwen-asr package not installed. Run: pip install qwen-asr")
        return

    try:
        # device_map="auto" works well if accelerate is installed
        # But for MPS, sometimes explicit is needed.
        # qwen-asr library might wrap transformers.
        model = Qwen3ASRModel.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device, 
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load data
    try:
        with open(data_manifest, 'r') as f:
            # Try loading as standard JSON list or object first (handles multiline JSON)
            content = json.load(f)
            if isinstance(content, dict):
                data = [content]
            elif isinstance(content, list):
                data = content
            else:
                raise ValueError("JSON content must be list or dict")
    except json.JSONDecodeError:
        # Fallback to JSONL (line by line)
        with open(data_manifest, 'r') as f:
            data = [json.loads(line) for line in f if line.strip()]
        
    print(f"Loaded {len(data)} samples from manifest")
    
    # Check resume
    existing_results = set()
    if os.path.exists(output_manifest):
         with open(output_manifest, 'r') as f:
            for line in f:
                try:
                    res = json.loads(line)
                    existing_results.add(res['audio_filepath'])
                except: pass
    
    print(f"Resuming {len(existing_results)} existing results")
    
    os.makedirs(os.path.dirname(output_manifest), exist_ok=True)
    results_file = open(output_manifest, 'a' if len(existing_results) > 0 else 'w', encoding='utf-8')
    
    all_inference_time = 0
    all_audio_duration = 0
    all_inference_memory = []
    
    # Initial memory
    initial_memory = 0
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        initial_memory = torch.mps.current_allocated_memory()/(1024 ** 3)
    
    count = 0
    for item in tqdm(data):
        audio_path = item['audio_filepath']
        
        # Replace {data_folder} placeholder if present
        if "{data_folder}" in audio_path:
            if data_folder:
                audio_path = audio_path.replace("{data_folder}", data_folder)
            else:
                print(f"Warning: {{data_folder}} placeholder found in {audio_path} but --data_folder arg not provided.")
                # Continue anyway, might fail
        
        # Original item path key remains for resume tracking coherence
        if item['audio_filepath'] in existing_results:
            continue
            
        start_t = time.time()
        try:
            # model.transcribe accepts path string or list
            # It returns a list of results
            results = model.transcribe(audio=audio_path, language=language)
            
            # The result object attributes: text, language, etc.
            pred_text = results[0].text
            
            inference_time = time.time() - start_t
            
            # Metric accumulation (exclude first few warmups if desired, but simple avg here)
            duration = item.get('duration', 0.0)
            if count > 4: # Skip warmup for RTF calc
                all_inference_time += inference_time
                all_audio_duration += duration
            
            # Peak memory check (approximate)
            peak_memory = 0
            if torch.backends.mps.is_available():
                 peak_memory = torch.mps.current_allocated_memory()/(1024 ** 3)
            all_inference_memory.append(peak_memory - initial_memory)
            
            res = {
                "audio_filepath": item['audio_filepath'], # Keep original path key
                "text": item['text'],
                "pred_text": pred_text,
                "inference_time": inference_time,
                "duration": duration
            }
            json.dump(res, results_file, ensure_ascii=False)
            results_file.write('\n')
            results_file.flush()
            
            count += 1
            if count % 50 == 0 and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            if "out of memory" in str(e).lower():
                torch.mps.empty_cache()
            
    results_file.close()
    print("Inference Complete.")
    
    # Metrics Output
    if all_audio_duration > 0:
        print("average rtf : ", all_inference_time/all_audio_duration)
    else:
        print("average rtf : N/A (insufficient data)")
        
    print("model memory : ", initial_memory)
    if all_inference_memory:
        print("average inference-only memory : ", sum(all_inference_memory)/len(all_inference_memory))
        
    # Run Eval if available
    if calculate_wer:
        print("Running Evaluation...")
        try:
            calculate_wer(output_manifest)
        except Exception as e:
            print(f"Evaluation Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--data_manifest", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--data_folder", default=None, help="Path to replace {data_folder} placeholder in manifest")
    parser.add_argument("--language", default=None, help="Language to force (e.g. Arabic)")
    args = parser.parse_args()
    
    run_qwen(args.model_id, args.data_manifest, args.output_manifest, args.data_folder, args.language)
