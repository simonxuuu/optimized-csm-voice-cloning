import os
from pathlib import Path
import modal

# Define the Modal app
app = modal.App("csm-voice-cloning")

# Set up the image with required dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsox-dev", "libsox-fmt-all", "sox")  # Added sox libraries
    .pip_install(
        "torch==2.4.0",
        "torchaudio==2.4.0",
        "tokenizers==0.21.0",
        "transformers==4.49.0",
        "huggingface_hub==0.28.1",
        "moshi==0.2.2",
        "torchtune==0.4.0",
        "torchao==0.9.0",
        "silentcipher @ git+https://github.com/SesameAILabs/silentcipher@master",
        "numpy",
    ).add_local_file("/Users/zay/csm/generator.py", "/root/generator.py")
    .add_local_file("/Users/zay/csm/models.py", "/root/models.py")
    .add_local_file("/Users/zay/csm/watermarking.py", "/root/watermarking.py")
    .add_local_file("/Users/zay/csm/audio.mp3", "/root/audio.mp3")
)

# Create a volume to cache the model weights
cache_dir = "/cache"
model_cache = modal.Volume.from_name(
    "csm-model-cache", create_if_missing=True
)

# Create a volume to store voice outputs
voice_output_dir = "/voice_output"
voice_output_volume = modal.Volume.from_name(
    "csm-voice-output", create_if_missing=True
)

# Function to generate speech with a specific speaker ID
@app.function(gpu="l40s", image=image, volumes={cache_dir: model_cache, voice_output_dir: voice_output_volume})
def generate_speech_with_speaker(
    text="Hello, this is my voice speaking.",
    speaker_id=0,
    output_filename="output.wav"
):
    import os
    import torch
    import torchaudio
    from huggingface_hub import hf_hub_download
    
    # Set HF cache directory to our persistent volume
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_TOKEN"] = ""
    
    # Import the generator
    from generator import load_csm_1b, Segment
    
    # Download the model if not already cached
    model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
    
    # Load the model
    generator = load_csm_1b(model_path, "cuda")
    print("Model loaded successfully!")
    
    # Create output directory
    os.makedirs(voice_output_dir, exist_ok=True)
    output_path = os.path.join(voice_output_dir, output_filename)
    
    print(f"Generating speech with speaker ID {speaker_id}")
    print(f"Text: {text}")
    
    # Generate audio
    audio = generator.generate(
        text=text,
        speaker=speaker_id,
        context=[],
        max_audio_length_ms=30_000,  # Increased from 10_000 to allow for longer utterances
        temperature=0.7,  # Reduced temperature for more accurate pronunciation
        topk=30,  # Reduced from default 50 to make output more focused
        max_seq_len=2048 # Increase max seq len for longer audio but might cause issues so you might need to edit in the models.py as well 
    )
    
    # Save the audio
    torchaudio.save(output_path, audio.unsqueeze(0).cpu(), generator.sample_rate)
    
    # Also save to a temporary file for returning
    temp_output_path = "/tmp/output.wav"
    torchaudio.save(temp_output_path, audio.unsqueeze(0).cpu(), generator.sample_rate)
    
    # Read the file and return the bytes
    with open(temp_output_path, "rb") as f:
        audio_bytes = f.read()
    
    print(f"Speech generated successfully! Saved to {output_path}")
    return audio_bytes

# Function to generate speech with context from a previous audio sample
@app.function(gpu="l40s", image=image, volumes={cache_dir: model_cache, voice_output_dir: voice_output_volume})
def generate_speech_with_context(
    text="Hello, this is my voice speaking with context.",
    speaker_id=0,
    context_audio_path="/root/zay-1.mp3",
    context_text="This is a sample of my voice for context.",
    output_filename="output_with_context.wav"
):
    import os
    import torch
    import torchaudio
    from huggingface_hub import hf_hub_download
    import numpy as np
    
    # Set HF cache directory to our persistent volume
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_TOKEN"] = "" # Need to set token here
    
    # Import the generator
    from generator import load_csm_1b, Segment
    
    # Download the model if not already cached
    model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
    
    # Load the model
    generator = load_csm_1b(model_path, "cuda")
    print("Model loaded successfully!")
    
    # Create output directory
    os.makedirs(voice_output_dir, exist_ok=True)
    output_path = os.path.join(voice_output_dir, output_filename)
    
    print(f"Generating speech with context")
    print(f"Context audio: {context_audio_path}")
    print(f"Text: {text}")
    
    # Load context audio
    context_audio, sr = torchaudio.load(context_audio_path)
    context_audio = context_audio.mean(dim=0)  # Convert to mono
    
    # Resample if needed
    if sr != generator.sample_rate:
        context_audio = torchaudio.functional.resample(
            context_audio, orig_freq=sr, new_freq=generator.sample_rate
        )
    
    # Normalize audio volume for better consistency
    context_audio = context_audio / (torch.max(torch.abs(context_audio)) + 1e-8)
    
    # Improved silence removal that handles internal silences
    def remove_silence(audio, threshold=0.01, min_silence_duration=0.2, sample_rate=24000):
        # Convert to numpy for easier processing
        audio_np = audio.cpu().numpy()
        
        # Calculate energy
        energy = np.abs(audio_np)
        
        # Find regions above threshold (speech)
        is_speech = energy > threshold
        
        # Convert min_silence_duration to samples
        min_silence_samples = int(min_silence_duration * sample_rate)
        
        # Find speech segments
        speech_segments = []
        in_speech = False
        speech_start = 0
        
        for i in range(len(is_speech)):
            if is_speech[i] and not in_speech:
                # Start of speech segment
                in_speech = True
                speech_start = i
            elif not is_speech[i] and in_speech:
                # Potential end of speech segment
                # Only end if silence is long enough
                silence_count = 0
                for j in range(i, min(len(is_speech), i + min_silence_samples)):
                    if not is_speech[j]:
                        silence_count += 1
                    else:
                        break
                
                if silence_count >= min_silence_samples:
                    # End of speech segment
                    in_speech = False
                    speech_segments.append((speech_start, i))
        
        # Handle case where audio ends during speech
        if in_speech:
            speech_segments.append((speech_start, len(is_speech)))
        
        # Concatenate speech segments
        if not speech_segments:
            return audio  # Return original if no speech found
        
        # Add small buffer around segments
        buffer_samples = int(0.05 * sample_rate)  # 50ms buffer
        processed_segments = []
        
        for start, end in speech_segments:
            buffered_start = max(0, start - buffer_samples)
            buffered_end = min(len(audio_np), end + buffer_samples)
            processed_segments.append(audio_np[buffered_start:buffered_end])
        
        # Concatenate all segments
        processed_audio = np.concatenate(processed_segments)
        
        return torch.tensor(processed_audio, device=audio.device)
    
    # Apply improved silence removal with slightly more aggressive settings for longer files
    audio_duration_sec = len(context_audio) / generator.sample_rate
    print(f"Original audio duration: {audio_duration_sec:.2f} seconds")
    
    # Adjust threshold based on audio length
    silence_threshold = 0.015
    if audio_duration_sec > 10:
        # For longer files, be more aggressive with silence removal
        silence_threshold = 0.02
    
    context_audio = remove_silence(context_audio, threshold=silence_threshold, 
                                  min_silence_duration=0.15, 
                                  sample_rate=generator.sample_rate)
    
    processed_duration_sec = len(context_audio) / generator.sample_rate
    print(f"Processed audio duration: {processed_duration_sec:.2f} seconds")
    
    # Use the entire audio file for better voice cloning
    print(f"Using the entire processed audio file ({processed_duration_sec:.2f} seconds) for voice cloning")
    
    # Create context segment
    context_segment = Segment(
        text=context_text,
        speaker=speaker_id,
        audio=context_audio
    )
    
    # Preprocess text for better pronunciation
    # Add punctuation if missing to help with phrasing
    if not any(p in text for p in ['.', ',', '!', '?']):
        text = text + '.'
    
    # Generate audio with context
    audio = generator.generate(
        text=text,
        speaker=speaker_id,
        context=[context_segment],
        max_audio_length_ms=15_000,  # Adjusted based on your feedback
        temperature=0.6,  # Lower temperature for more accurate pronunciation
        topk=20,  # More focused sampling for clearer speech
    )
    
    # Save the audio
    torchaudio.save(output_path, audio.unsqueeze(0).cpu(), generator.sample_rate)
    
    # Also save to a temporary file for returning
    temp_output_path = "/tmp/output_with_context.wav"
    torchaudio.save(temp_output_path, audio.unsqueeze(0).cpu(), generator.sample_rate)
    
    # Read the file and return the bytes
    with open(temp_output_path, "rb") as f:
        audio_bytes = f.read()
    
    print(f"Speech with context generated successfully! Saved to {output_path}")
    return audio_bytes

# Main entrypoint for the app
@app.local_entrypoint()
def main():
    import sys
    import os
    
    # Check if a custom audio file is provided as an argument
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        context_audio_path = sys.argv[1]
        print(f"Using custom audio file: {context_audio_path}")
    else:
        context_audio_path = "/root/audio.mp3"
        print(f"Using default audio file: {context_audio_path}")
    
    # Define the context text - try to match what's in your sample
    context_text = "" # Use whisper to transcribe the audio
    
    # Generate speech using the approach that worked best
    print("Generating speech with the successful approach...")
    audio_bytes = generate_speech_with_context.remote(
        text="My name is Zay. I am speaking clearly. This is my voice. Bro what",
        speaker_id=999,
        context_audio_path=context_audio_path,
        context_text=context_text,
        output_filename="cloned_voice.wav"
    )
    
    # Save the output locally
    output_file = "cloned_voice.wav"
    with open(output_file, "wb") as f:
        f.write(audio_bytes)
    
    print(f"Cloned voice audio saved to {output_file}")