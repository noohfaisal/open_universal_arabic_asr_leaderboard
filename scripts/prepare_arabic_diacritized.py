import os
import json
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

def prepare_arabic_diacritized():
    # Configuration
    dataset_name = "rFathi03/Arabic_Diacritized_Audio_Dataset"
    output_dir = "datasets/arabic_diacritized_audio"
    output_manifest = "datasets/arabic_diacritized_test.json"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading dataset: {dataset_name}...")
    # Load the train split as it's the only one available
    ds = load_dataset(dataset_name, split="train", streaming=True)
    # Disable auto-decoding to avoid dependency issues
    from datasets import Audio
    ds = ds.cast_column("audio", Audio(decode=False))
    
    manifest_entries = []
    
    print("Processing audio files...")
    # Using streaming mode, so we iterate carefully
    for i, item in enumerate(tqdm(ds)):
        try:
            audio_bytes = item['audio']['bytes']
            text = item['transcription']
            
            # Define output filename
            filename = f"audio_{i}.flac"
            file_path = os.path.join(output_dir, filename)
            
            # We need to decode the bytes to get duration and re-save as FLAC
            # Write temp file first
            temp_path = f"temp_{i}.wav" # Assumption
            with open(temp_path, "wb") as f:
                f.write(audio_bytes)
                
            # Read with soundfile
            try:
                data, samplerate = sf.read(temp_path)
                
                # Write to final FLAC
                sf.write(file_path, data, samplerate)
                
                # Calculate duration
                duration = len(data) / samplerate
                
                # Create manifest entry
                entry = {
                    "audio_filepath": file_path,
                    "text": text,
                    "duration": duration
                }
                manifest_entries.append(entry)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            continue
            
    # Write manifest
    print(f"Writing manifest to {output_manifest}...")
    with open(output_manifest, 'w', encoding='utf-8') as f:
        for entry in manifest_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
            
    print(f"Done! Processed {len(manifest_entries)} files.")

if __name__ == "__main__":
    prepare_arabic_diacritized()
