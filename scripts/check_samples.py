import json
import random
import os

manifests = [
    "results/whisper/sada_output.json",
    "results/whisper/casablanca_output.json",
    "results/whisper/arabic_diacritized_output.json"
]

for manifest in manifests:
    print(f"\n--- Sampling from: {manifest} ---")
    if not os.path.exists(manifest):
        print("File not found.")
        continue
        
    with open(manifest, 'r') as f:
        lines = f.readlines()
        
    if not lines:
        print("File is empty.")
        continue

    samples = random.sample(lines, min(5, len(lines)))
    
    for i, line in enumerate(samples):
        try:
            data = json.loads(line)
            print(f"Sample {i+1}:")
            print(f"  File: {data.get('audio_filepath', 'N/A')}")
            print(f"  Ref:  {data.get('text', 'N/A')}")
            print(f"  Pred: {data.get('pred_text', 'N/A')}")
        except json.JSONDecodeError:
            print(f"Sample {i+1}: Invalid JSON")
