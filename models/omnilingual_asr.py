import json
import time
import torch
from tqdm import tqdm
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


def run_omnilingual(model_id, data_manifest, data_folder, output_manifest):
    """
    Arguments
    ---------
    model_id: str
        ["omniASR_CTC_300M", "omniASR_LLM_300M", "omniASR_CTC_1B", "omniASR_LLM_1B", "omniASR_CTC_3B", "omniASR_LLM_3B", "omniASR_CTC_7B", "omniASR_LLM_7B"]
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
    pipeline = ASRInferencePipeline(model_card=model_id)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    num_lines = sum(1 for _ in open(data_manifest))
    print(f"Total items to process: {num_lines}")

    with open(data_manifest, "r") as f:
        with open(output_manifest, 'w') as fout:
            all_inference_time = 0
            all_audio_duration = 0
            all_inference_memory = []
            count = 0
            # Pass total to tqdm for ETA
            for line in tqdm(f, total=num_lines, unit="item", desc="Inference Progress"):
                item = json.loads(line)
                in_path = item["audio_filepath"].format(data_folder=data_folder)
                duration = item["duration"]

                initial_memory = 0
                if device == "cuda":
                    torch.cuda.reset_max_memory_allocated(torch.device("cuda"))
                    initial_memory = torch.cuda.max_memory_allocated(torch.device("cuda"))/(1024 ** 3)
                elif device == "mps":
                    initial_memory = 0 # MPS doesn't expose memory stats easily yet
                else:
                    initial_memory = 0
                
                # Warmup run is not explicitly done here, but usually fine
                start_time = time.time()
                try:
                    # pipeline.transcribe expects a list of inputs and returns a list of result strings
                    # We pass a single file [in_path] and get the first result [0]
                    transcription = pipeline.transcribe([in_path])[0]
                except Exception as err:
                    print(f"{err} with file {in_path}")
                    continue
                end_time = time.time()

                inference_time = end_time - start_time
                
                if count > 4:
                    all_inference_time += inference_time
                    all_audio_duration += duration
                    
                if device == "cuda":
                    peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda"))/(1024 ** 3)
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

    if all_audio_duration > 0:
        print("average rtf : ", all_inference_time/all_audio_duration)
    else:
        print("average rtf : N/A")
        
    print("model memory : ", initial_memory) 
    
    if all_inference_memory:
        print("average inference-only memory : ", sum(all_inference_memory)/len(all_inference_memory))
    else:
        print("average inference-only memory : N/A")
