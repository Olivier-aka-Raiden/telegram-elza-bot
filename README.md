# Personal AI Assistant Bot

A sophisticated AI assistant hosted on Google Cloud Run, using FastAPI backend and integrated with Telegram. This bot leverages OpenAI's GPT-4 model to process and respond to text messages, images, and voice inputs.

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Deployment](#deployment)
6. [Usage](#usage)
7. [Project Structure](#project-structure)
8. [Troubleshooting](#troubleshooting)
9. [Contributing](#contributing)
10. [License](#license)

## Features

- Text message processing and responses
- Image analysis and description
- Voice message transcription and voice response
- Text-to-speech for voice responses
- Hosted on Google Cloud Run for 24/7 availability
- FastAPI backend for efficient API handling
- Telegram bot integration for user interaction and easy access through telegram


Certainly! I'll add a new category to the README to explain the additional features you've implemented for your personal needs. Here's how we can incorporate this information:

## Personal Features

This bot includes some specialized features tailored for personal use:

### News Retrieval

The bot can fetch and summarize recent news from specific regions using GPT function calls. This feature allows users to stay updated on current events in France or Alsace.

#### How it works:

1. The bot uses a custom tool defined in the GPT model's function calling capability.
2. Users can request news by asking about current events in France or Alsace.
3. The bot processes the request and calls the appropriate function to fetch relevant news.

#### Function Definition:

```python
self.tools = [
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "récupère les dernières actualités, en France ou spécifiquement en Alsace",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "enum": ["France", "Alsace"],
                        "description": "Le périmètre de recherche pour récupérer les actualités",
                    },
                },
                "required": ["location"],
            },
        },
    }
]
```

#### Usage:

To get news, you can ask the bot questions like:
- "What's the latest news from France?"
- "Tell me about recent events in Alsace."
- "Any important headlines from France today?"

The bot will then use the appropriate function to fetch and summarize relevant news articles based on the specified location.

Note: This feature requires appropriate backend implementation to actually fetch and process the news data. Ensure that the necessary APIs or web scraping mechanisms are in place to support this functionality.

## Persistence and Context Memory

This bot implements a sophisticated persistence mechanism to maintain context across conversations and improve response relevance over time.

### Embedding-based Message Storage

The bot stores all messages along with their embeddings in a Google Cloud Storage bucket. This allows for efficient retrieval of relevant past conversations.

#### How it works:

1. For each message, the bot generates an embedding using OpenAI's text-embedding-ada-002 model.
2. The message and its embedding are stored together in the Google Cloud Storage bucket.
3. When a new message comes in, the bot computes its embedding and finds the most similar past messages.
4. These relevant past messages are included in the system prompt to provide context for the AI's response.

#### Implementation Details:

```python
# Generate embedding for the message
response = client.embeddings.create(
    model="text-embedding-ada-002",
    input=message_obj["content"]
)

# Prepare the message-embedding pair for storage
emb_mess_pair = {
    "embedding": json.dumps(response.data[0].embedding),
    "message": json.dumps(message_obj)
}

# Store the pair in Google Cloud Storage
# (Implementation details for storage not shown here)
```

### Context Retrieval

When processing a new message:

1. The bot computes the embedding for the incoming message.
2. It then calculates the cosine similarity between this embedding and all stored embeddings.
3. The most similar past messages are retrieved and incorporated into the system prompt.

This approach allows the bot to:
- Maintain long-term memory of conversations
- Provide more contextually relevant responses
- Improve continuity in multi-turn conversations

### Benefits

- **Improved Consistency**: The bot can refer to past conversations, maintaining consistency in its responses over time.
- **Personalization**: By remembering user-specific information, the bot can provide more tailored responses.
- **Enhanced Context Understanding**: The AI can draw upon a broader context when formulating responses, leading to more informed and relevant answers.

### Technical Considerations

- Ensure proper security measures are in place to protect stored messages and embeddings.
- Regularly monitor and manage the storage usage in your Google Cloud Storage bucket.
- Consider implementing a retention policy or periodic cleanup to manage the growth of stored data.

### Usage

This feature works automatically in the background. Users will experience more contextually aware and personalized responses without needing to explicitly reference past conversations.

Note: The exact implementation of similarity calculation and the method of incorporating past messages into the prompt may vary based on specific requirements and optimizations.
## Prerequisites

- Python 3.10 or higher
- Google Cloud Platform account
- OpenAI API account
- Telegram Bot Token

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/personal-ai-assistant-bot.git
   cd personal-ai-assistant-bot
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Configuration

1. Create a `.env` file in the project root and add the following environment variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   TELEGRAM_TOKEN=your_telegram_bot_token
   ```

2. Obtain API keys:
   - OpenAI API:
     1. Go to [OpenAI](https://www.openai.com/)
     2. Register and create a new project
     3. Generate an API key for your project
   - Telegram Bot Token: Create a new bot using [BotFather](https://core.telegram.org/bots#6-botfather) on Telegram

## Deployment

1. Install and set up the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)

2. Authenticate with Google Cloud:
   ```
   gcloud auth login
   ```

3. Set your project ID:
   ```
   gcloud config set project YOUR_PROJECT_ID
   ```

4. Build and deploy the container to Google Cloud Run:
   ```
   gcloud run deploy personal-ai-assistant --source . --platform managed --region your-preferred-region
   ```

5. Follow the prompts to complete the deployment

## Usage

1. Start a chat with your Telegram bot

2. Send text messages, images, or voice messages to interact with the AI assistant

3. The bot will process your input and respond accordingly, using text or voice as appropriate

## Project Structure

```
telegram-elza-bot/
├── main.py
├── requirements.txt
├── Dockerfile
├── .gcloudignore
├── .env.example
├── .gitignore
├── README.md
├── LICENSE
└── start.bat
```

## Troubleshooting

- If you encounter issues with API rate limits, check your usage and consider upgrading your plan
- For deployment problems, review the Google Cloud Run logs for detailed error messages
- Ensure all required environment variables are correctly set in your `.env` file and in your Google Cloud Run configuration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
