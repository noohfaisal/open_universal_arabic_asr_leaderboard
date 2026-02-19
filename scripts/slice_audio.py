
import pandas as pd
from pydub import AudioSegment
import os
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path to sada_test.csv")
    parser.add_argument("--source_dir", type=str, required=True, help="Root directory containing batch_X folders")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save segmented wavs")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    # Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # We need to reconstruct the expected path structure.
    # The JSON expects: {data_folder}/filename-seg-start-end.wav
    # The CSV has 'FileName': 'batch_1/6k_SBA_107_0.wav'
    # And 'SegmentID': '6k_SBA_107_0-seg_3_990-14_890' (This looks like the filename base)
    
    # We will preserve the 'batch_1' folder structure in output to match JSON?
    # Let's check JSON again.
    # JSON: "{data_folder}/6k_SBA_107_0-seg_3_990-14_890.wav"
    # It does NOT seem to include 'batch_1' in the filename part of the JSON path?
    # Wait, the JSON entry is: "audio_filepath": "{data_folder}/6k_SBA_107_0-seg_..."
    # If data_folder is "datasets/batch_1", then it expects them directly there.
    # BUT the CSV input files are in different batches (batch_1, batch_2 etc likely).
    # If the JSON flattens them, we should flatten them too?
    
    # Let's see if JSON has 'batch_1/' in the path.
    # Step 427: "audio_filepath": "{data_folder}/6k_SBA_107_0-seg_3_990-14_890.wav"
    # It does NOT have batch_X. 
    # The user is running with --data_folder datasets/batch_1
    # This implies the user thinks all audio is in batch_1.
    # BUT the CSV has 'FileName': 'batch_1/...'
    
    # Recommendation: We will output ALL segments into 'datasets/sada_segments' (flat or nested?)
    # And we will need to UPDATE the manifest or ensure usage matches.
    # Since I cannot easily change the JSON (it's 6k lines), I should match what the JSON expects.
    # The JSON expects "{data_folder}/FILENAME.wav".
    # So if I output to `datasets/batch_1/FILENAME.wav`, then data_folder=`datasets/batch_1` works.
    # BUT are there files from `batch_2`?
    
    # Let's check unique batches in CSV.
    unique_batches = df['FileName'].apply(lambda x: x.split('/')[0]).unique()
    print(f"Batches found: {unique_batches}")
    
    # If multiple batches, we might have a problem if JSON variable {data_folder} is static.
    # If JSON is static, maybe it expects all files in ONE folder.
    
    unique_sources = df['FileName'].unique()
    
    print(f"Found {len(unique_sources)} source audio files.")
    
    for source_rel_path in tqdm(unique_sources):
        full_source_path = os.path.join(args.source_dir, source_rel_path)
        
        if not os.path.exists(full_source_path):
            print(f"Warning: Source file not found: {full_source_path}")
            continue
            
        try:
            audio = AudioSegment.from_wav(full_source_path)
        except Exception as e:
            print(f"Failed to load {full_source_path}: {e}")
            continue

        # Get all segments for this file
        file_segments = df[df['FileName'] == source_rel_path]
        
        for _, row in file_segments.iterrows():
            start_sec = row['SegmentStart']
            end_sec = row['SegmentEnd']
            seg_id = row['SegmentID'] 
            
            # Pydub works in millis
            start_ms = int(start_sec * 1000)
            end_ms = int(end_sec * 1000)
            
            segment_audio = audio[start_ms:end_ms]
            
            # Construct output filename.
            # We want to match what is in the JSON 'id' or filename.
            # CSV SegmentID: 6k_SBA_107_0-seg_3_990-14_890
            # JSON filename: 6k_SBA_107_0-seg_3_990-14_890.wav
            
            out_name = f"{seg_id}.wav"
            
            # We will ignore the 'batch_X' folder for output and put everything in output_dir
            # This allows data_folder argument to just point to output_dir.
            out_path = os.path.join(args.output_dir, out_name)
            
            segment_audio.export(out_path, format="wav")
            
if __name__ == "__main__":
    main()
