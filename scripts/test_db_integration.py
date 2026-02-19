
import sys
import os
import random

# Add scripts dir to path
sys.path.append(os.path.join(os.getcwd(), "scripts"))

from db_manager import init_db, log_experiment, log_inferences, get_all_experiments, get_inferences_for_experiment

def main():
    # 1. Initialize DB
    print("--- 1. Initializing Database ---")
    if os.path.exists("metrics.db"):
        os.remove("metrics.db") # Clean start
    init_db()

    # 2. Simulate Experiment 1: Model A on Dataset 1
    print("\n--- 2. Simulating Experiment: Model A on Dataset 1 ---")
    
    # Dummy aggregated metrics
    global_wer = 0.15
    global_cer = 0.05
    avg_latency = 0.45
    
    exp_id = log_experiment("ModelA", "Dataset1", global_wer, global_cer, avg_latency)
    print(f"Logged Experiment with ID: {exp_id}")

    # Dummy inference data
    inferences = []
    dataset_size = 5
    for i in range(dataset_size):
        duration = random.uniform(2.0, 10.0)
        inference_time = duration * 0.1 # fast model
        
        item = {
            "audio_filepath": f"dataset1/file_{i}.wav",
            "text": "مرحبا بالعالم",
            "pred_text": "مرحبا عالم", # slight error
            "wer": 0.5 if i % 2 == 0 else 0.0, # dummy wer
            "cer": 0.1,
            "duration": duration,
            "inference_time": inference_time,
            "rtf": inference_time / duration
        }
        inferences.append(item)
    
    log_inferences(exp_id, inferences)
    
    # 3. Simulate Experiment 2: Model B on Dataset 2
    print("\n--- 3. Simulating Experiment: Model B on Dataset 2 ---")
    exp_id_2 = log_experiment("ModelB", "Dataset2", 0.05, 0.01, 1.2) # Slower but accurate
    
    inferences_b = []
    for i in range(3):
        duration = random.uniform(2.0, 10.0)
        inference_time = duration * 0.5 
        item = {
            "audio_filepath": f"dataset2/file_{i}.wav",
            "text": "تجربة ناجحة",
            "pred_text": "تجربة ناجحة",
            "wer": 0.0,
            "cer": 0.0,
            "duration": duration,
            "inference_time": inference_time,
            "rtf": inference_time / duration
        }
        inferences_b.append(item)
    log_inferences(exp_id_2, inferences_b)

    # 4. Verify / Retrieve
    print("\n--- 4. Verification: Querying Database ---")
    
    print("\n[Experiments Table]")
    exps = get_all_experiments()
    print("ID | Model | Dataset | WER | Latency")
    for e in exps:
        # Schema: id, model, dataset, time, wer, cer, lat
        print(f"{e[0]} | {e[1]} | {e[2]} | {e[4]} | {e[6]}")

    print(f"\n[Inferences for Experiment {exp_id} - Model A]")
    infs = get_inferences_for_experiment(exp_id)
    # Schema: id, exp_id, audio, gt, pred, wer, cer, dur, inf_time, rtf
    print("File | GT | Pred | WER | RTF")
    for i in infs:
        print(f"{i[2]} | {i[3]} | {i[4]} | {i[5]} | {i[9]:.2f}")

if __name__ == "__main__":
    main()
