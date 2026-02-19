
import json
import os
import argparse

def check_files(manifest_path, data_folder):
    print(f"Checking {manifest_path} with data_folder={data_folder}")
    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        return

    with open(manifest_path, 'r') as f:
        line = f.readline()
        if not line:
            print("Empty manifest")
            return
        
        try:
            item = json.loads(line)
            audio_path = item["audio_filepath"].format(data_folder=data_folder)
            print(f"First entry path: {audio_path}")
            
            if os.path.exists(audio_path):
                print("SUCCESS: File exists.")
            else:
                print("FAILURE: File does NOT exist.")
                print(f"Current working directory: {os.getcwd()}")
                print("Please check if the 'data_folder' argument matches your actual data location.")
        except Exception as e:
            print(f"Error parsing JSON: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("data_folder")
    args = parser.parse_args()
    check_files(args.manifest, args.data_folder)
