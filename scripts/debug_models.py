
try:
    from omnilingual_asr.models.inference import ASRInferencePipeline
    print("ASRInferencePipeline imported successfully.")
except ImportError:
    print("Could not import ASRInferencePipeline")

try:
    import fairseq2.models
    # Try to find a registry or list function
    # inspecting fairseq2.models
    print("fairseq2 dir:", dir(fairseq2.models))
except ImportError:
    print("Could not import fairseq2")
