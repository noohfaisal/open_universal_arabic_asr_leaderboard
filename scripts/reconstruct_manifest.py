import json
import os

input_json = "results/omnilingual_asr/arabic_diacritized_output.json"
output_manifest = "datasets/arabic_diacritized_test.json"

print(f"Reconstructing {output_manifest} from {input_json}...")

entries = []
if not os.path.exists(input_json):
    print(f"Error: {input_json} not found.")
    exit(1)

with open(input_json, 'r') as f:
    for line in f:
        data = json.loads(line)
        # We need audio_filepath, text, duration
        entry = {
            "audio_filepath": data["audio_filepath"],
            "text": data["text"],
            "duration": data.get("duration", 0.0)
        }
        entries.append(entry)

print(f"Found {len(entries)} entries.")

with open(output_manifest, 'w', encoding='utf-8') as f:
    for entry in entries:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')

print("Reconstruction complete.")
