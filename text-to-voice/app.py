import pyttsx3

engine = pyttsx3.init()
text = "Hello! This is an AI speaking."
engine.say(text)
engine.save_to_file(text, 'output.wav')
engine.runAndWait()

