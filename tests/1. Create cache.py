# These tests are not yet complete
# --------------------------------

# This test file creates a cache for testing
# ------------------------------------------

# Specify settings
import os
os.environ['CACHED_OPENAI_VERBOSE'] = 'True'
os.environ['CACHED_OPENAI_DEV_MODE'] = 'True'

# Load the package
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../cached_openai')))
import src as cached_openai

# Create a client
client = cached_openai.OpenAI()

# Run some chat completions
chats = []
chats.append(client.chat.completions.create(model    = 'gpt-4o',
                                            messages = [{'role':'user', 'content':'give me a random color'}]))
for seed in range(3):
    chats.append(client.chat.completions.create(model    = 'gpt-4o',
                                                messages = [{'role':'user', 'content':'give me a random color'}],
                                                seed     = seed))

chats = [i.id for i in chats]

print(chats)


response_sound = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Today is a wonderful day to build something people love!",
)
response_sound.stream_to_file('file.mp3')