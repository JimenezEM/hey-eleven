from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

from dotenv import load_dotenv
import os

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
AGENT_ID = os.getenv("AGENT_ID")

# ─────────────────────────────────────────────
# ElevenLabs client
# ─────────────────────────────────────────────
client = ElevenLabs(api_key=ELEVEN_API_KEY)

def start_conversation_with_agent() -> None:
    print("\nConnecting to ElevenLabs agent...\n")

    conversation = Conversation(
        client=client,
        agent_id=AGENT_ID,
        requires_auth=True,
        audio_interface=DefaultAudioInterface(),
        callback_agent_response=lambda response: print(f"Agent: {response}"),
        callback_user_transcript=lambda transcript: print(f"You:   {transcript}"),
    )

    try:
        conversation.start_session()
    except KeyboardInterrupt:
        print("\nStopping conversation...")

    print("\nConversation ended.\n")

def main():
    print("=" * 50)
    print("   Voice Assistant (Press ENTER to talk)")
    print("   Type 'q' then ENTER to quit")
    print("=" * 50)

    while True:
        user_input = input("\nPress ENTER to start talking... ")

        if user_input.lower() == "q":
            print("Goodbye!")
            break

        start_conversation_with_agent()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()