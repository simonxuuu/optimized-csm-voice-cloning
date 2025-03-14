# Voice Cloning with CSM-1B

This repository contains tools to clone your voice using the Sesame CSM-1B model. It provides two methods for voice cloning:
1. Local execution on your own GPU
2. Cloud execution using Modal

> **Note:** While this solution does capture some voice characteristics and provides a recognizable clone, it's not the best voice cloning solution available. The results are decent but not perfect. If you have ideas on how to improve the cloning quality, feel free to contribute!

## Prerequisites

- Python 3.10+
- CUDA-compatible GPU (for local execution)
- Hugging Face account with access to the CSM-1B model
- Hugging Face API token

## Installation

1. Clone this repository:
```bash
git clone https://github.com/isaiahbjork/csm-voice-cloning.git
cd csm-voice-cloning
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Setting Up Your Hugging Face Token

You need to set your Hugging Face token to download the model. You can do this in two ways:

1. Set it as an environment variable:
```bash
export HF_TOKEN="your_hugging_face_token"
```

2. Or directly in the `voice_clone.py` file:
```python
os.environ["HF_TOKEN"] = "your_hugging_face_token"
```

## Accepting the Model on Hugging Face

Before using the model, you need to accept the terms on Hugging Face:

1. Visit the [Sesame CSM-1B model page](https://huggingface.co/sesame/csm-1b)
2. Click on "Access repository" and accept the terms
3. Make sure you're logged in with the same account that your HF_TOKEN belongs to

## Preparing Your Voice Sample

1. Record a clear audio sample of your voice (2-3 minutes is recommended)
2. Save it as an MP3 or WAV file
3. Transcribe the audio using Whisper or another transcription tool to get the exact text

## Running Voice Cloning Locally

1. Edit the `voice_clone.py` file to set your parameters directly in the code:

```python
# Set the path to your voice sample
context_audio_path = "path/to/your/voice/sample.mp3"

# Set the transcription of your voice sample
# You need to use Whisper or another tool to transcribe your audio
context_text = "The exact transcription of your voice sample..."

# Set the text you want to synthesize
text = "Text you want to synthesize with your voice."

# Set the output filename
output_filename = "output.wav"
```

2. Run the script:
```bash
python voice_clone.py
```

## Running Voice Cloning on Modal

Modal provides cloud GPU resources for faster processing:

1. Install Modal:
```bash
pip install modal
```

2. Set up Modal authentication:
```bash
modal token new
```

3. Edit the `modal_voice_cloning.py` file to set your parameters directly in the code:

```python
# Set the path to your voice sample
context_audio_path = "path/to/your/voice/sample.mp3"

# Set the transcription of your voice sample
# You need to use Whisper or another tool to transcribe your audio
context_text = "The exact transcription of your voice sample..."

# Set the text you want to synthesize
text = "Text you want to synthesize with your voice."

# Set the output filename
output_filename = "output.wav"
```

4. Run the Modal script:
```bash
modal run modal_voice_cloning.py
```

## Important Note on Model Sequence Length

If you encounter tensor dimension errors, you may need to adjust the model's maximum sequence length in `models.py`. The default sequence length is 2048, which works for most cases, but if you're using longer audio samples, you might need to increase this value.

Look for the `max_seq_len` parameter in the `llama3_2_1B()` and `llama3_2_100M()` functions in `models.py` and ensure they have the same value:

```python
def llama3_2_1B():
    return llama3_2.llama3_2(
        # other parameters...
        max_seq_len=2048,  # Increase this value if needed
        # other parameters...
    )
```

## Example

Using a 2 minute and 50 second audio sample works fine with the default settings. For longer samples, you may need to adjust the sequence length as mentioned above.

## Troubleshooting

- **Tensor dimension errors**: Adjust the model sequence length as described above
- **CUDA out of memory**: Try reducing the audio sample length or use a GPU with more memory
- **Model download issues**: Ensure you've accepted the model terms on Hugging Face and your token is correct

## License

This project uses the Sesame CSM-1B model, which is subject to its own license terms. Please refer to the [model page](https://huggingface.co/sesame/csm-1b) for details. 