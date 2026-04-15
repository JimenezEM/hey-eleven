from dotenv import load_dotenv
import os
import json

from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

from vosk import Model as VoskModel, KaldiRecognizer
import sounddevice as sd
import numpy as np

# ─────────────────────────────────────────────
# ENV
# ─────────────────────────────────────────────
load_dotenv()

ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
AGENT_ID       = os.getenv("AGENT_ID")

if not ELEVEN_API_KEY or not AGENT_ID:
    raise ValueError("Missing API key or Agent ID.")

client = ElevenLabs(api_key=ELEVEN_API_KEY)

# ─────────────────────────────────────────────
# ElevenLabs conversation
# ─────────────────────────────────────────────
def start_conversation_with_agent():
    print("\n🎤 Starting conversation...\n")

    conversation = Conversation(
        client=client,
        agent_id=AGENT_ID,
        requires_auth=True,
        audio_interface=DefaultAudioInterface(),
        callback_agent_response=lambda r: print(f"Agent: {r}"),
        callback_user_transcript=lambda t: print(f"You:   {t}"),
    )

    try:
        conversation.start_session()
    except KeyboardInterrupt:
        print("\n⛔ Conversation interrupted.")

    print("\n🔁 Back to listening...\n")


# ─────────────────────────────────────────────
# Wake word via speech recognition
# ─────────────────────────────────────────────
def listen_for_wake_word():
    print("🔊 Loading speech model...")

    model = VoskModel("vosk-model-small-es-0.42")
    recognizer = KaldiRecognizer(model, 16000)

    print("👂 Listening for 'hey eleven'...")
    print("Press Ctrl+C to quit.\n")

    while True:
        wake_detected = False

        def audio_callback(indata, frames, time, status):
            nonlocal wake_detected

            audio = indata.tobytes()

            if recognizer.AcceptWaveform(audio):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")

                if text:
                    print(f"🧠 Heard: {text}")

                if "hola" in text or "museito" in text:
                    print("\n✨ Wake word detected!")
                    wake_detected = True

        # Start listening
        with sd.InputStream(
            samplerate=16000,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=audio_callback,
        ):
            try:
                while True:
                    if wake_detected:
                        print("🛑 Stopping listener...")
                        break
            except KeyboardInterrupt:
                print("\n👋 Exiting.")
                return

        # NOW we are OUTSIDE the mic stream → safe to start assistant
        print("🚀 Launching assistant...\n")
        start_conversation_with_agent()

        print("👂 Listening again...\n")

# ─────────────────────────────────────────────
# ENTRY
# ─────────────────────────────────────────────
if __name__ == "__main__":
    listen_for_wake_word()