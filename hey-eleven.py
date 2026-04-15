from dotenv import load_dotenv
import os
import json
import threading
import time

from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

from vosk import Model as VoskModel, KaldiRecognizer
import sounddevice as sd

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
    print("\nStarting conversation...\n")

    done_event = threading.Event()

    conversation = Conversation(
        client=client,
        agent_id=AGENT_ID,
        requires_auth=True,
        audio_interface=DefaultAudioInterface(),
        callback_agent_response=lambda r: print(f"Agent: {r}"),
        callback_user_transcript=lambda t: print(f"You:   {t}"),

        # callbacks fire from inside ElevenLabs' thread
        # when the session actually ends — not when start_session() returns
        callback_agent_response_correction=None,
    )

    # Monkey-patch the end-of-session signal onto the audio interface
    # so we know when ElevenLabs truly closes the mic
    original_stop = conversation.audio_interface.stop
    def patched_stop():
        original_stop()
        print("Agent mic released.")
        done_event.set()
    conversation.audio_interface.stop = patched_stop

    try:
        conversation.start_session()
        # non-blocking, fires background thread
        print("Waiting for session to end...")
        done_event.wait()
    except KeyboardInterrupt:
        print("\nConversation interrupted.")
        try:
            conversation.end_session()
        except Exception:
            pass
        done_event.set()

    # Extra buffer after mic is released by ElevenLabs
    time.sleep(0.8)
    print("\nBack to listening...\n")


# ─────────────────────────────────────────────
# Wake word via Vosk
# ─────────────────────────────────────────────
def listen_for_wake_word():
    print("Loading speech model...")
    model = VoskModel("vosk-model-small-es-0.42")

    print("Listening for 'Hola Museito'...")
    print("Press Ctrl+C to quit.\n")

    while True:
        recognizer = KaldiRecognizer(model, 16000)
        wake_event = threading.Event()

        def audio_callback(indata, frames, time_info, status):
            if wake_event.is_set():
                return

            audio = indata.tobytes()

            if recognizer.AcceptWaveform(audio):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")

                if text:
                    print(f"🧠 Heard: {text}")

                if "hola" in text or "museito" in text:
                    print("\nWake word detected!")
                    wake_event.set()

        with sd.InputStream(
            samplerate=16000,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=audio_callback,
        ):
            try:
                wake_event.wait()
                print("Stopping listener...")
            except KeyboardInterrupt:
                print("\Exiting.")
                return

        time.sleep(0.5)
        print("Launching assistant...\n")
        start_conversation_with_agent()
        print("Listening again...\n")


if __name__ == "__main__":
    listen_for_wake_word()