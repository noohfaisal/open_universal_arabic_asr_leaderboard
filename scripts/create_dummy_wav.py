
import wave
import struct
import math
import argparse

def create_dummy_wav(filename, duration=1.0, sample_rate=16000):
    num_samples = int(duration * sample_rate)
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        
        # Generate a simple sine wave
        for i in range(num_samples):
            value = int(32767.0 * math.sin(2 * math.pi * 440.0 * i / sample_rate))
            data = struct.pack('<h', value)
            wav_file.writeframes(data)
            
    print(f"Created {filename} with duration {duration}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    create_dummy_wav(args.filename)
