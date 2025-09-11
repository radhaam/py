import os
import requests
import pyttsx3
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from transformers import pipeline
from twilio.rest import Client
from dotenv import load_dotenv, dotenv_values

config = dotenv_values(".env")

# -------- CONFIG --------
NEWSAPI_KEY = config["NEWSAPI_KEY"]   # get from https://newsapi.org/
NEWS_COUNTRY = "us"

TWILIO_ACCOUNT_SID = config["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = config["TWILIO_AUTH_TOKEN"]
TWILIO_WHATSAPP_FROM = "whatsapp:+14155238886"  # Twilio Sandbox default
RECIPIENT_WHATSAPP_TO = config["RECIPIENT_WHATSAPP_TO"] # e.g. "whatsapp:+91xxxxxxxxxx"

AUDIO_FILE = "daily_news.wav"

# -------- Init --------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
tts_engine = pyttsx3.init()

# -------- Helpers --------
def fetch_news():
    url = f"https://newsapi.org/v2/top-headlines?country={NEWS_COUNTRY}&category=business&apiKey={NEWSAPI_KEY}"
    resp = requests.get(url)
    data = resp.json()
    return [a["title"] for a in data.get("articles", [])[:5]]

def summarize_news(headlines):
    text = " ".join(headlines)
    result = summarizer(text, do_sample=False)
    return result[0]["summary_text"]

def text_to_speech(text, filename=AUDIO_FILE):
    tts_engine.save_to_file(text, filename)
    tts_engine.runAndWait()
    return filename

def send_whatsapp_voice(file_path, text_preview):
    message = twilio_client.messages.create(
        from_=TWILIO_WHATSAPP_FROM,
        to=RECIPIENT_WHATSAPP_TO,
        body=f"Here is your daily news update:\n{text_preview}\n\n(Voice message attached is not directly supported by Twilio. Use audio file URL instead.)"
    )
    print("Sent WhatsApp SID:", message.sid)

# -------- Job --------
def job():
    print("Fetching news...")
    headlines = fetch_news()
    if not headlines:
        print("No news today.")
        return

    summary = summarize_news(headlines)
    print("Summary:", summary)

    #audio_file = text_to_speech(summary)

    # Twilio doesn't allow direct push of .wav as voice note on WhatsApp
    # Instead: upload audio to your server (or ngrok/localtunnel), then pass its URL as media_url
    message = twilio_client.messages.create(
        from_=TWILIO_WHATSAPP_FROM,
        to=RECIPIENT_WHATSAPP_TO,
        body="Here is your daily news update: "+summary
    )
    print("WhatsApp sent:", message.sid)

# -------- Scheduler --------
if __name__ == "__main__":
    print("summary")
    job()
    #scheduler = BlockingScheduler()
    # Run every day at 8 AM
    #scheduler.add_job(job, "cron", hour=8, minute=0)
    #print("Scheduler started. Waiting for job...")
    #scheduler.start()
