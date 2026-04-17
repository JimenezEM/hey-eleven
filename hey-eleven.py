from dotenv import load_dotenv
import os
import json
import threading
import time
import sounddevice as sd
import numpy as np

from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface
from vosk import Model as VoskModel, KaldiRecognizer

# variables de entorno desde .env
load_dotenv()

# credenciales ElevenLabs
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
AGENT_ID = os.getenv("AGENT_ID")

# validación de credenciales
if not ELEVEN_API_KEY or not AGENT_ID:
    raise ValueError("Missing API key or Agent ID.")

# inición del cliente
client = ElevenLabs(api_key=ELEVEN_API_KEY)

# palabras despedida
FAREWELL_PHRASES = [
    "adiós", "adios", "hasta luego", "hasta pronto",
    "nos vemos", "que tengas", "bye", "goodbye", "chao", "chau"
]

# tiempo máximo de silencio permitido
SILENCE_TIMEOUT = 25

# tiempo mínimo (evitar interferencia)
SELF_HEAR_GUARD = 4.0

# --------------------------------------------------
# reinicio de audio
# --------------------------------------------------
def reset_portaudio():
    try:
        sd.stop()
    except Exception:
        pass

    try:
        sd._lib.Pa_Terminate()
    except Exception:
        pass

    try:
        sd._lib.Pa_Initialize()
    except Exception:
        pass

    time.sleep(0.5)


# --------------------------------------------------
# conversación con agente
# --------------------------------------------------
def start_conversation_with_agent():
    print("\nstarting conversation...\n")

    end_requested = threading.Event()
    silence_timer = [None]
    end_reason = [None]
    conversation = [None]

    # evitar que el agente detecte su propio audio
    agent_speaking_until = [0]

    def mark_agent_speaking(seconds=SELF_HEAR_GUARD):
        agent_speaking_until[0] = time.time() + seconds

    def agent_is_speaking():
        return time.time() < agent_speaking_until[0]

    def cancel_timer():
        if silence_timer[0]:
            silence_timer[0].cancel()
            silence_timer[0] = None

    def request_end(reason):
        if end_requested.is_set():
            return

        end_reason[0] = reason
        cancel_timer()

        print(f"ending session ({reason})...")
        end_requested.set()

    # cierre seguro sin bloquear el hilo principal
    def safe_end_session():
        try:
            if conversation[0]:
                conversation[0].end_session()
        except Exception as e:
            print(f"force end error: {e}")

    # fuerza el cierre inmediato de la sesión
    def force_end_now(reason):
        if end_requested.is_set():
            return

        request_end(reason)

        shutdown_thread = threading.Thread(
            target=safe_end_session,
            daemon=True
        )
        shutdown_thread.start()
        shutdown_thread.join(timeout=3)

    # temporizador de silencio
    def reset_silence_timer(delay=SILENCE_TIMEOUT):
        if end_requested.is_set():
            return

        cancel_timer()

        timer = threading.Timer(
            delay,
            lambda: force_end_now("silence timeout")
        )
        timer.daemon = True
        timer.start()

        silence_timer[0] = timer

    # respuesta del agente
    def on_agent_response(response):
        if end_requested.is_set():
            return

        # cantidad de tiempo ignorar el micrófono
        estimated_seconds = max(
            SELF_HEAR_GUARD,
            len(response.split()) * 0.35
        )

        mark_agent_speaking(estimated_seconds)

        print(f"Agent: {response}")

        # con despedida, termina sesión automáticamente
        if any(p in response.lower() for p in FAREWELL_PHRASES):
            print("farewell detected — ending in 5 seconds...")

            cancel_timer()

            timer = threading.Timer(
                5.0,
                lambda: force_end_now("farewell")
            )
            timer.daemon = True
            timer.start()

            silence_timer[0] = timer
            return

        reset_silence_timer()

    # se ejecuta cuando el usuario habla
    def on_user_transcript(text):
        if end_requested.is_set():
            return

        # ignora audio capturado del mismo agente
        if agent_is_speaking():
            return

        clean = text.strip()

        if not clean or clean == "...":
            return

        print(f"You: {clean}")

        # si el usuario se despide, finaliza
        if any(p in clean.lower() for p in FAREWELL_PHRASES):
            force_end_now("user farewell")
            return

        reset_silence_timer()

    # conversación con ElevenLabs
    conversation[0] = Conversation(
        client=client,
        agent_id=AGENT_ID,
        requires_auth=True,
        audio_interface=DefaultAudioInterface(),
        callback_agent_response=on_agent_response,
        callback_user_transcript=on_user_transcript,
    )

    try:
        conversation[0].start_session()

        print("waiting for session to end...\n")

        reset_silence_timer()

        end_requested.wait()

        try:
            waiter = threading.Thread(
                target=conversation[0].wait_for_session_end,
                daemon=True
            )
            waiter.start()
            waiter.join(timeout=3)
        except Exception:
            pass

    except KeyboardInterrupt:
        print("\ninterrupted.")
        force_end_now("interrupted")

    finally:
        print("resetting audio...")
        reset_portaudio()

        print("\nreturning to wake word listener...\n")


# --------------------------------------------------
# Escucha de palabra clave
# --------------------------------------------------
def listen_for_wake_word():
    print("loading speech model...")
    model = VoskModel("vosk-model-small-es-0.42")

    print("listening for 'Hola Museito'...")
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
                text = result.get("text", "").lower().strip()

                if text:
                    print(f"heard: {text}")

                # palabra de activación
                if "hola" in text or "museito" in text:
                    print("\wake word detected")
                    wake_event.set()

        try:
            with sd.InputStream(
                samplerate=16000,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=audio_callback,
            ):
                wake_event.wait()

        except sd.PortAudioError as e:
            print(f"mic error: {e} — retrying in 2s...")
            time.sleep(2)
            continue

        print("stopping listener...")
        time.sleep(0.5)

        print("launching assistant...\n")
        start_conversation_with_agent()

        time.sleep(1.0)


# main
if __name__ == "__main__":
    listen_for_wake_word()