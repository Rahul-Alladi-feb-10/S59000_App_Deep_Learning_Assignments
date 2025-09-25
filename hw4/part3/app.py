import gradio as gr
from transformers import pipeline

# Initialize the Whisper pipeline for speech-to-text
# Using the small model for good balance of speed and accuracy
whisper_pipeline = pipeline(
    task="automatic-speech-recognition", 
    model="openai/whisper-small"
)

def transcribe_audio(audio_file):
    """
    Convert audio to text using Whisper model
    
    Args:
        audio_file: Audio file uploaded by user
    
    Returns:
        str: Transcribed text
    """
    if audio_file is None:
        return "Please upload an audio file."
    
    try:
        # Process the audio file
        result = whisper_pipeline(audio_file)
        transcription = result["text"]
        return transcription
    except Exception as e:
        return f"Error processing audio: {str(e)}"

# Create the Gradio interface
gradio_app = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(
        label="Upload Audio File", 
        type="filepath",
        sources=["upload", "microphone"]
    ),
    outputs=gr.Textbox(
        label="Transcription", 
        lines=5,
        placeholder="Your transcribed text will appear here..."
    ),
    title="🎤 Speech-to-Text with Whisper",
    description="Upload an audio file or record directly to convert speech to text using OpenAI's Whisper model.",
    examples=[
        # You can add example audio files here if you have any
    ],
    theme=gr.themes.Soft()
)

# Launch the app
if __name__ == "__main__":
    gradio_app.launch()