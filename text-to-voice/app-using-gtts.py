from gtts import gTTS
from playsound import playsound
import os
#pip install gtts

my_text = "This is a demonstration of text to voice conversion using gTTS."
language = 'en'
my_audio = gTTS(text=my_text, lang=language, slow=False)
my_audio.save("output1.mp3")
#pip install playsound
# Specify the path to your audio file
# If using an online environment, ensure the file is accessible in that environment

audio_file_path = "output1.mp3" #or .wav
playsound(audio_file_path)
print("Playing sound using playsound")
