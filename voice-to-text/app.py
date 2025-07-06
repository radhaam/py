import whisper

model = whisper.load_model("base")  # or "small", "medium", "large"

# Path to your audio file
result = model.transcribe("inputaudio.wav")
print("Transcription:", result["text"])
