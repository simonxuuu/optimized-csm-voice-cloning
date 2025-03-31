import os
from pathlib import Path
import torch
import torchaudio
from huggingface_hub import hf_hub_download
import numpy as np
import sys
import io
import time

os.environ["HF_TOKEN"] = "" # Need to set token here

def generate_speech_with_context(
    text="Hello, this is my voice speaking with context.",
    speaker_id=999,
    context_audio_path="audio.mp3",
    context_text="This is a sample of my voice for context.",
    output_filename="output_with_context.wav",
    optimize_speed=True,  # New parameter for speed optimization
    max_audio_length_ms=15_000,
    batch_size=16
):
    # Import the generator
    from generator import load_csm_1b, Segment
    
    # Start timing
    start_time = time.time()
    
    # Download the model if not already cached
    model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
    
    # Load the model
    print("Loading model...")
    generator = load_csm_1b(model_path, "cuda")
    print(f"Model loaded successfully in {time.time() - start_time:.2f} seconds!")
    
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
    
    # Optimize performance if requested
    if optimize_speed:
        # Skip silence removal for faster processing
        processed_audio = context_audio
        
        # Limit context length if it's too long (for faster processing)
        max_context_length = int(5.0 * generator.sample_rate)  # Use at most 5 seconds of context
        if len(processed_audio) > max_context_length:
            processed_audio = processed_audio[:max_context_length]
        
        print(f"Using optimized context audio ({len(processed_audio)/generator.sample_rate:.2f} seconds) for fast generation")
    else:
        # Use the original silence removal for better quality (but slower)
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
        
        processed_audio = remove_silence(context_audio, threshold=silence_threshold, 
                                       min_silence_duration=0.15, 
                                       sample_rate=generator.sample_rate)
        
        processed_duration_sec = len(processed_audio) / generator.sample_rate
        print(f"Processed audio duration: {processed_duration_sec:.2f} seconds")
    
    # Create context segment
    context_segment = Segment(
        text=context_text,
        speaker=speaker_id,
        audio=processed_audio
    )
    
    # Preprocess text for better pronunciation
    # Add punctuation if missing to help with phrasing
    if not any(p in text for p in ['.', ',', '!', '?']):
        text = text + '.'
    
    # Set optimization parameters
    optimize_memory = True
    apply_watermark = False  # Skip watermarking for speed
    
    # Precompute cache and optimize CUDA operations
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    # Generate audio with context
    generation_start_time = time.time()
    audio = generator.generate(
        text=text,
        speaker=speaker_id,
        context=[context_segment],
        max_audio_length_ms=max_audio_length_ms,
        temperature=0.6,  # Lower temperature for more accurate pronunciation
        topk=20,  # More focused sampling for clearer speech
        max_seq_len=2048,  # Maximum sequence length
        batch_size=batch_size,  # Process frames in batches for speed
        apply_watermark=not optimize_speed,  # Skip watermarking if optimizing for speed
        optimize_memory=optimize_memory  # Use memory optimizations
    )
    
    generation_time = time.time() - generation_start_time
    print(f"Voice generation completed in {generation_time:.2f} seconds")
    
    # Save the audio to the output directory
    torchaudio.save(output_filename, audio.unsqueeze(0).cpu(), generator.sample_rate)
    
    total_time = time.time() - start_time
    print(f"Speech with context generated successfully in {total_time:.2f} seconds! Saved to {output_filename}")
    return generation_time

def main():
    context_audio_path = "audio.mp3"
    context_text = "" # Use whisper to transcribe the audio
    
    # Generate speech using the optimized approach
    print("Generating speech with optimized settings for maximum speed...")
    generation_time = generate_speech_with_context(
        text="My name is Zay. I am speaking clearly. This is my voice.",
        speaker_id=999,
        context_audio_path=context_audio_path,
        context_text=context_text,
        output_filename="cloned_voice_fast.wav",
        optimize_speed=True,
        max_audio_length_ms=15_000,
        batch_size=32  # Use larger batch size for T4 GPU
    )
    
    print(f"Generation completed in {generation_time:.2f} seconds")

if __name__ == "__main__":
    main()