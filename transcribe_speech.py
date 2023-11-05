# Import the necessary libraries
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# Load the pre-trained model and tokenizer
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")

# Define a function to transcribe speech to text
def transcribe_speech(audio_file_path):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_file_path)
    
    # Preprocess the audio waveform
    input_values = tokenizer(waveform, return_tensors="pt").input_values
    
    # Perform speech recognition
    with torch.no_grad():
        logits = model(input_values).logits
    
    # Decode the predicted tokens
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    
    return transcription

# Example usage
audio_file_path = "path/to/audio.wav"
transcription = transcribe_speech(audio_file_path)
print(transcription)
