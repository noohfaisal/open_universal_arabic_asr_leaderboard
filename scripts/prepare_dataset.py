
import argparse
import json
import os
import torchaudio
from tqdm import tqdm
from pathlib import Path

def create_manifest(audio_dir, transcript_path, output_path):
    audio_files = list(Path(audio_dir).rglob("*.wav"))
    print(f"Found {len(audio_files)} wav files in {audio_dir}")

    # Load transcripts
    transcripts = {}
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                # Assuming format: filename text
                key = os.path.splitext(os.path.basename(parts[0]))[0]
                transcripts[key] = parts[1]
            elif len(parts) == 1:
                # Maybe just id? Skip
                pass

    print(f"Loaded {len(transcripts)} transcripts")

    with open(output_path, 'w', encoding='utf-8') as fout:
        for audio_path in tqdm(audio_files):
            file_id = audio_path.stem
            
            if file_id in transcripts:
                try:
                    info = torchaudio.info(str(audio_path))
                    duration = info.num_frames / info.sample_rate
                    
                    entry = {
                        "audio_filepath": str(audio_path),
                        "duration": duration,
                        "text": transcripts[file_id]
                    }
                    json.dump(entry, fout, ensure_ascii=False)
                    fout.write('\n')
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}")

    print(f"Manifest saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ASR manifest from audio dir and transcript file")
    parser.add_argument("--audio_dir", required=True, help="Path to directory containing wav files")
    parser.add_argument("--transcript", required=True, help="Path to transcript text file")
    parser.add_argument("--output", required=True, help="Path to output json manifest")
    
    args = parser.parse_args()
    create_manifest(args.audio_dir, args.transcript, args.output)
