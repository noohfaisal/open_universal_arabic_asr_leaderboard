import json
import collections

manifest = "results/whisper/sada_output.json"
paths = []

with open(manifest, 'r') as f:
    for i, line in enumerate(f):
        try:
            item = json.loads(line)
            paths.append(item['audio_filepath'])
        except Exception as e:
            print(f"Line {i} error: {e}")

counts = collections.Counter(paths)
duplicates = {k: v for k, v in counts.items() if v > 1}

print(f"Total items: {len(paths)}")
print(f"Unique items: {len(counts)}")
print(f"Num Duplicates: {len(duplicates)}")

if duplicates:
    print("Example duplicates:")
    for k in list(duplicates.keys())[:5]:
        print(f"  {k}: {duplicates[k]} times")
