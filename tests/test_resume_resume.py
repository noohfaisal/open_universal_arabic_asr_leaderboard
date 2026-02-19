import os
import shutil
import tempfile
import json
from unittest.mock import MagicMock, patch

# --- Mocking the dependencies to avoid running actual model inference ---

# We only need to import the function we want to test. 
# However, the module imports torch, transformers etc at top level.
# We can mock these imports or let them run if available. 
# Given the environment has them, we can import directly.
# But for safety and speed, let's just copy the logic we want to test or 
# assume the logic is simple enough we can just test the function 'run_whisper' 
# with mocked 'pipeline' and 'AutoModel' to avoid big downloads.

import sys
sys.path.append(os.getcwd())

# Mock torch and transformers to prevent loading models
with patch.dict(sys.modules, {
    'torch': MagicMock(),
    'transformers': MagicMock(),
    'transformers.pipeline': MagicMock(), 
}):
    # We need to reload/import after mocking if we want to ensure mocks are used
    # But since we're in the same process, we just need to make sure we haven't imported yet
    # or we patch where it is used.
    
    # Actually, simpler approach:
    # Just write a unit test for the specific PATH NORMALIZATION logic we added.
    # We can create a dummy file and check if logic holds.
    pass

from models.whisper import run_whisper

def test_resume_logic():
    # Setup
    tmp_dir = tempfile.mkdtemp()
    data_manifest = os.path.join(tmp_dir, "manifest.json")
    output_manifest = os.path.join(tmp_dir, "output.json")
    data_folder = os.path.join(tmp_dir, "audio")
    os.makedirs(data_folder)
    
    # Create dummy audio file
    audio_file = "test.wav"
    audio_path = os.path.join(data_folder, audio_file)
    with open(audio_path, "w") as f:
        f.write("dummy audio")

    # Create input manifest
    # Note: Using {data_folder} as expected by the script
    with open(data_manifest, "w") as f:
        # Item 1
        json.dump({
            "audio_filepath": "{data_folder}/test.wav", # Standard
            "duration": 1.0,
            "text": "test"
        }, f)
        f.write("\n")
        # Item 2
        json.dump({
            "audio_filepath": "{data_folder}/test2.wav", # Standard
            "duration": 1.0,
            "text": "test2"
        }, f)
        f.write("\n")

    # Create output manifest (Simulating previous run)
    # Here we simulate that 'test.wav' was already processed
    # BUT we use a SLIGHTLY DIFFERENT path string (e.g. double slash) to test normalization
    # The script uses: in_path = item["audio_filepath"].format(data_folder=data_folder)
    # The output manifest stores the formatted path
    
    processed_path = os.path.join(data_folder, "test.wav")
    # Add an extra slash or variation if possible, but os.path.join usually acts clean.
    # We manually construct a weird path
    weird_processed_path = processed_path.replace("audio/", "audio//") 
    
    with open(output_manifest, "w") as f:
        json.dump({
            "audio_filepath": weird_processed_path, 
            "text": "test",
            "pred_text": "pred",
            "inference_time": 0.1,
            "duration": 1.0
        }, f)
        f.write("\n")

    print(f"Created setup:\nManifest: {data_manifest}\nOutput: {output_manifest}\nData Folder: {data_folder}")
    print(f"Simulating resume. 'test.wav' is in output with path: {weird_processed_path}")

    # Mock the pipeline so it doesn't actually run inference (and tracking calls)
    # We patch 'models.whisper.pipeline' used inside run_whisper
    # AND 'models.whisper.AutoModelForSpeechSeq2Seq' etc to verify we skip
    
    with patch('models.whisper.pipeline') as mock_pipeline, \
         patch('models.whisper.AutoModelForSpeechSeq2Seq.from_pretrained'), \
         patch('models.whisper.AutoProcessor.from_pretrained'):
        
        mock_pipe_instance = MagicMock()
        mock_pipe_instance.return_value = {"text": "prediction"}
        mock_pipeline.return_value = mock_pipe_instance

        # Run inference
        print("Running inference...")
        run_whisper("dummy_model", data_manifest, data_folder, output_manifest)
        
        # Verification
        # We expect pipeline to be called ONCE (for test2.wav)
        # Because test.wav should be skipped despite the weird path in output
        
        print("\nVerifying calls...")
        if mock_pipe_instance.call_count == 1:
            print("SUCCESS: Pipeline called exactly once (skipped 1, processed 1).")
            # Verify it was test2.wav
            args, _ = mock_pipe_instance.call_args
            called_path = args[0]
            if "test2.wav" in called_path:
                print("SUCCESS: Processed test2.wav as expected.")
            else:
                print(f"FAILURE: Processed wrong file: {called_path}")
        else:
            print(f"FAILURE: Pipeline called {mock_pipe_instance.call_count} times. (Expected 1)")

    # Cleanup
    shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    test_resume_logic()
