# Voice Chatbot

Hey! 👋 This is a cool voice chatbot that lets you have natural conversations with an AI. Just speak, and it'll respond with a voice - simple as that!

## Quick Setup

1. Make sure you've got Python 3.10+ installed
2. Run `pip install -r requirements.txt`
3. Set up your `.env` file with your API keys:
```
GROQ_API_KEY=your_groq_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
```

## Running the Bot

1. Just run:
```bash
python VoiceBot.py
```
2. Wait for "Listening..."
3. Start talking!
4. Say "goodbye" when you're done

## What's Inside?

The bot uses three main pieces:
- Deepgram for converting your speech to text (and back to speech)
- Groq's Mixtral model for smart responses
- FFplay for playing the audio

Everything's in the `Components` folder if you want to peek under the hood:
- `asr.py`: Handles speech recognition
- `llm.py`: Processes conversation
- `tts.py`: Converts text back to speech

## Need Help?

If things aren't working:
- Check your mic is plugged in and working
- Make sure FFplay is installed
- Double-check your API keys in the `.env` file

## That's it!

Give it a try and have fun chatting! If you run into any issues or have ideas for making it better, let me know. 😊
