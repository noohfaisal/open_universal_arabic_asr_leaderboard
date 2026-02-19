
import os
import json
import soundfile as sf
from datasets import load_dataset, Audio
from tqdm import tqdm
import io

SUBSETS = ['Algeria', 'Egypt', 'Jordan', 'Mauritania', 'Morocco', 'Palestine', 'UAE', 'Yemen']
OUTPUT_DIR = "datasets/casablanca_audio"
OUTPUT_MANIFEST = "datasets/casablanca_test.json"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_entries = []
    
    print(f"Preparing Casablanca dataset (8 subsets)...")
    
    for subset in SUBSETS:
        print(f"Processing subset: {subset}")
        try:
            # Load the TEST split with streaming=True to avoid disk space issues
            # cast_column works on IterableDataset to disable decoding
            ds = load_dataset("UBC-NLP/Casablanca", subset, split="test", streaming=True)
            ds = ds.cast_column("audio", Audio(decode=False))
        except Exception as e:
            print(f"Failed to load subset {subset}: {e}")
            continue
            
        for i, item in enumerate(tqdm(ds)):
            # item['audio'] is now {'bytes': b'...', 'path': '...'}
            audio_blob = item['audio']
            
            # Unique filename: subset_index.flac
            filename = f"{subset}_{i}.flac"
            filepath = os.path.join(OUTPUT_DIR, filename)

            try:
                # Decode manually using soundfile
                if 'bytes' in audio_blob and audio_blob['bytes']:
                    audio_array, sample_rate = sf.read(io.BytesIO(audio_blob['bytes']))
                elif 'path' in audio_blob and audio_blob['path'] and os.path.exists(audio_blob['path']):
                     audio_array, sample_rate = sf.read(audio_blob['path'])
                else:
                    # Some streaming datasets provide URLs? Casablanca is likely parquet with bytes.
                    print(f"Skipping {subset} item {i}: No audio bytes or path found.")
                    continue

                # Save audio as FLAC locally (smaller size)
                sf.write(filepath, audio_array, sample_rate, format='FLAC')
                
                # Create manifest entry
                entry = {
                    "audio_filepath": filepath,
                    "text": item['transcription'],
                    "duration": item['duration'] if 'duration' in item and item['duration'] else len(audio_array)/sample_rate
                }
                all_entries.append(entry)
            except Exception as e:
                print(f"Error processing item {i} in {subset}: {e}")
                continue
            
    print(f"Total samples collected: {len(all_entries)}")
    
    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as f:
        for entry in all_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
            
    print(f"Manifest saved to {OUTPUT_MANIFEST}")

if __name__ == "__main__":
    main()
