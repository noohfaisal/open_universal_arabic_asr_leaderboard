import json
import os
import time
import torch
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

if torch.cuda.is_available():
    device = "cuda:0"
    torch_dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "mps"
    # Use float32 for MPS as float16 produces invalid output ("!")
    torch_dtype = torch.float32
else:
    device = "cpu"
    torch_dtype = torch.float32


def run_whisper(model_id, data_manifest, data_folder, output_manifest):
    """
    Arguments
    ---------
    model_id: str
        HuggingFace whisper model name
    data_manifest: str
        Path of a data manifest under datasets/
    data_folder: str
        The path to the test set
    output_manifest: str
        The output manifest path

    Output
    ---------
    Create an output manifest containing ground truths and predictions
    """
    print(f"Loading model {model_id} on {device} with {torch_dtype}...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=1, # Reduced from 16 to avoid MPS hangs
        return_timestamps=False,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    # Calculate dataset size for tqdm
    with open(data_manifest, "r") as f:
        total_lines = sum(1 for _ in f)

    # Check for existing progress to resume
    processed_files = set()
    if os.path.exists(output_manifest):
        with open(output_manifest, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Normalize path to ensure consistency
                    p_file = os.path.normpath(data.get("audio_filepath", ""))
                    processed_files.add(p_file)
                except json.JSONDecodeError:
                    pass
        print(f"Found {len(processed_files)} processed items. Resuming...")

    with open(data_manifest, "r") as f:
        # Open in append mode to resume
        with open(output_manifest, 'a') as fout:
            all_inference_time = 0
            all_audio_duration = 0
            all_inference_memory = []
            count = 0
            for line in tqdm(f, total=total_lines, desc="Whisper Inference"):
                item = json.loads(line)
                in_path = item["audio_filepath"].format(data_folder=data_folder)
                # Normalize path for comparison
                in_path_norm = os.path.normpath(in_path)
                
                # Resume Check
                if in_path_norm in processed_files:
                    continue

                duration = item["duration"]

                initial_memory = 0
                if torch.cuda.is_available():
                    torch.cuda.reset_max_memory_allocated(torch.device("cuda"))
                    initial_memory = torch.cuda.max_memory_allocated(torch.device("cuda"))/(1024 ** 3)
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                    initial_memory = torch.mps.current_allocated_memory()/(1024 ** 3)

                start_time = time.time()
                # Ensure Language is Arabic
                transcription = pipe(in_path, generate_kwargs = {"language":"ar", "task": "transcribe"})["text"]
                end_time = time.time()
                
                inference_time = end_time - start_time
                if count > 4:
                    all_inference_time += inference_time
                    all_audio_duration += duration
                
                peak_memory = 0
                if torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda"))/(1024 ** 3)
                elif torch.backends.mps.is_available():
                    peak_memory = torch.mps.current_allocated_memory()/(1024 ** 3)

                all_inference_memory.append(peak_memory-initial_memory)        
                count += 1

                metadata = {
                    "audio_filepath": in_path,
                    "text": item["text"],
                    "pred_text": transcription,
                    "inference_time": inference_time,
                    "duration": duration
                }
                json.dump(metadata, fout, ensure_ascii=False)
                fout.write('\n')
                fout.flush() # Ensure data is written immediately

                # Periodic memory cleanup
                if count % 50 == 0:
                     if torch.backends.mps.is_available():
                         torch.mps.empty_cache()
                     elif torch.cuda.is_available():
                         torch.cuda.empty_cache()

    if all_audio_duration > 0:
        print("average rtf : ", all_inference_time/all_audio_duration)
    else:
        print("average rtf : N/A (insufficient data)")
    print("model memory : ", initial_memory)
    print("average inference-only memory : ", sum(all_inference_memory)/len(all_inference_memory))