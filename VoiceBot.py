# Import required libraries
import asyncio
from dotenv import load_dotenv
import shutil
import subprocess
import requests
import time
import os

# Import LangChain components for LLM integration
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

# Import Deepgram components for speech processing
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

# Load environment variables from .env file
load_dotenv()

class LanguageModelProcessor:
    """
    Handles interaction with language models (Groq or OpenAI) and manages conversation state.
    """
    def __init__(self):
        # Initialize language model with Groq's Mixtral (can switch to OpenAI models)
        self.llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY"))
        # Alternative OpenAI model configurations (commented out)
        # self.llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview", openai_api_key=os.getenv("OPENAI_API_KEY"))
        # self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125", openai_api_key=os.getenv("OPENAI_API_KEY"))

        # Initialize conversation memory to maintain context
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Load system prompt from file to define AI assistant's behavior
        with open('sys_command.txt', 'r') as file:
            system_prompt = file.read().strip()
        
        # Set up chat prompt template with system message, chat history, and user input
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])
        
        # Create LLM chain combining model, prompt, and memory
        self.conversation = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )

    def process(self, text):
        """
        Process user input through the language model and return response.
        Measures and logs processing time.
        """
        # Add user message to conversation history
        self.memory.chat_memory.add_user_message(text)

        # Time the LLM response
        start_time = time.time()
        response = self.conversation.invoke({"text": text})
        end_time = time.time()

        # Add AI response to conversation history
        self.memory.chat_memory.add_ai_message(response['text'])

        # Calculate and log processing time
        elapsed_time = int((end_time - start_time) * 1000)
        print(f"LLM ({elapsed_time}ms): {response['text']}")
        return response['text']

class TextToSpeech:
    """
    Handles text-to-speech conversion using Deepgram's API and audio playback.
    """
    MODEL_NAME = "aura-helios-en"

    @staticmethod
    def is_installed(lib_name: str) -> bool:
        """Check if required system library is installed."""
        lib = shutil.which(lib_name)
        return lib is not None

    def speak(self, text):
        """
        Convert text to speech and play audio.
        Measures and logs time to first byte (TTFB).
        """
        # Deepgram API configuration
        DEEPGRAM_API_KEY = "770ef91e91e48b0bea67bf18acbcb7f4903fc2be"
        DEEPGRAM_URL = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
        headers = {
             "Authorization": f"Token {DEEPGRAM_API_KEY}",
             "Content-Type": "application/json"
        }
        payload = {"text": text}

        # Set up audio player process using ffplay
        player_command = ["ffplay", "-autoexit", "-", "-nodisp"]
        player_process = subprocess.Popen(
            player_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Track timing for performance monitoring
        start_time = time.time()
        first_byte_time = None

        # Stream audio data from Deepgram API to ffplay
        with requests.post(DEEPGRAM_URL, stream=True, headers=headers, json=payload) as r:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    if first_byte_time is None:
                        first_byte_time = time.time()
                        ttfb = int((first_byte_time - start_time)*1000)
                        print(f"TTS Time to First Byte (TTFB): {ttfb}ms\n")
                    player_process.stdin.write(chunk)
                    player_process.stdin.flush()

        # Clean up audio player process
        if player_process.stdin:
            player_process.stdin.close()
        player_process.wait()

class TranscriptCollector:
    """
    Manages collection and assembly of speech transcript parts.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Clear transcript buffer."""
        self.transcript_parts = []

    def add_part(self, part):
        """Add new transcript segment."""
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        """Combine all transcript parts into complete text."""
        return ' '.join(self.transcript_parts)

# Global transcript collector instance
transcript_collector = TranscriptCollector()

async def get_transcript(callback):
    """
    Asynchronously capture and transcribe speech using Deepgram.
    
    Args:
        callback: Function to call with completed transcript
    """
    transcription_complete = asyncio.Event()

    try:
        # Initialize Deepgram client with keepalive
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram: DeepgramClient = DeepgramClient("", config)

        # Set up live transcription connection
        dg_connection = deepgram.listen.asynclive.v("1")
        print("Listening...")

        async def on_message(self, result, **kwargs):
            """Handle incoming transcription results."""
            sentence = result.channel.alternatives[0].transcript
            
            if not result.speech_final:
                # Accumulate partial transcripts
                transcript_collector.add_part(sentence)
            else:
                # Process complete sentence
                transcript_collector.add_part(sentence)
                full_sentence = transcript_collector.get_full_transcript()
                if len(full_sentence.strip()) > 0:
                    full_sentence = full_sentence.strip()
                    print(f"Human: {full_sentence}")
                    callback(full_sentence)
                    transcript_collector.reset()
                    transcription_complete.set()

        # Configure transcription parameters
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=300,
            smart_format=True,
        )

        # Start transcription
        await dg_connection.start(options)
        microphone = Microphone(dg_connection.send)
        microphone.start()

        # Wait for transcription to complete
        await transcription_complete.wait()

        # Clean up resources
        microphone.finish()
        await dg_connection.finish()

    except Exception as e:
        print(f"Could not open socket: {e}")
        return

class ConversationManager:
    """
    Orchestrates the complete conversation flow between speech input,
    language model processing, and speech output.
    """
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()

    async def main(self):
        """Main conversation loop."""
        def handle_full_sentence(full_sentence):
            """Callback for completed transcriptions."""
            self.transcription_response = full_sentence

        # Main conversation loop
        while True:
            # Get speech input
            await get_transcript(handle_full_sentence)
            
            # Check for exit command
            if "goodbye" in self.transcription_response.lower():
                break
            
            # Process through LLM and generate speech response
            llm_response = self.llm.process(self.transcription_response)
            tts = TextToSpeech()
            tts.speak(llm_response)

            # Reset for next iteration
            self.transcription_response = ""

# Entry point
if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())
