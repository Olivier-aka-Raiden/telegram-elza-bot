# IMPORT VARIABLE ENV
from contextlib import asynccontextmanager
from http import HTTPStatus
from openai import OpenAI, OpenAIError
from google.cloud import storage
from io import StringIO
import numpy as np
import pandas as pd
import ast
from dotenv import load_dotenv, find_dotenv
import pickle
import base64
import os
import json
import tempfile
from pathlib import Path
import logging
from telegram import Bot, Update, ReplyKeyboardRemove
from telegram.ext import ContextTypes, CommandHandler, MessageHandler, filters, ConversationHandler, \
    BasePersistence, Application, PersistenceInput
import http
from werkzeug.wrappers import Response
from fastapi import FastAPI, Request, Response
import requests

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())

TYPING_QUESTION = range(1)

actu_headers = {
    "User-Agent": "AiAssistant/0.1"
}

def cosine_distance(a, b):
    return 1 - sum([a_i * b_i for a_i, b_i in zip(a, b)]) / (
            sum([a_i ** 2 for a_i in a]) ** 0.5 * sum([b_i ** 2 for b_i in b]) ** 0.5)

class GCPStoragePersistence(BasePersistence):
    """
    A class to handle persistence using Google Cloud Storage.
    """

    def __init__(self, bucket_name, file_path, store_data=PersistenceInput(bot_data=False, chat_data=False, user_data=False, callback_data=False)):
        super().__init__(store_data)
        self.client = storage.Client()
        self.bucket_name = bucket_name
        self.file_path = file_path
        self.bucket = self.client.bucket(bucket_name)
        self.blob = self.bucket.blob(file_path)
        self._conversations = {}

    async def _load_data(self):
        """
        Load data from Google Cloud Storage.
        """
        try:
            if self.blob.exists():
                data = self.blob.download_as_string()
                loaded_data = pickle.loads(data)
                self._conversations = loaded_data.get('conversations', {})
                logger.info(f"Loaded data from persistence: {self._conversations}")
        except Exception as e:
            logger.error(f"Could not load data: {e}")

    async def _save_data(self):
        """
        Save data to Google Cloud Storage.
        """
        try:
            data = {'conversations': self._conversations}
            logger.info(f"Data to be saved: {data}")
            serialized_data = pickle.dumps(data)
            self.blob.upload_from_string(serialized_data)
            logger.info("Data saved to GCP.")
        except Exception as e:
            logger.error(f"Could not save data: {e}")

    # Placeholder methods for other data types
    async def get_bot_data(self):
        pass

    async def update_bot_data(self, data):
        pass

    async def refresh_bot_data(self, bot_data):
        pass

    async def get_chat_data(self):
        pass

    async def update_chat_data(self, chat_id, data):
        pass

    async def refresh_chat_data(self, chat_id, chat_data):
        pass

    async def drop_chat_data(self, chat_id):
        pass

    async def get_user_data(self):
        pass

    async def update_user_data(self, user_id, data):
        pass

    async def refresh_user_data(self, user_id, user_data):
        pass

    async def drop_user_data(self, user_id):
        pass

    async def get_callback_data(self):
        pass

    async def update_callback_data(self, data):
        pass

    async def get_conversations(self, name=None):
        """
        Retrieve conversations from the persistence storage.
        
        Args:
            name (str): The name of the conversation to retrieve. If None, all conversations are retrieved.
        
        Returns:
            dict: The conversations data.
        """
        if not self._conversations:
            try:
                if self.blob.exists():
                    data = self.blob.download_as_string()
                    loaded_data = pickle.loads(data)
                    self._conversations = loaded_data.get('conversations', {})
                    logger.info(f"Loaded data from persistence: {self._conversations}")
            except Exception as e:
                logger.error(f"Could not load data: {e}")

        if name:
            return self._conversations.get(name, {})
        return self._conversations

    async def update_conversation(self, name, key, new_state):
        """
        Update a specific conversation's state.
        
        Args:
            name (str): The name of the conversation.
            key (str): The key of the state to update.
            new_state (any): The new state to set.
        """
        if name not in self._conversations:
            self._conversations[name] = {}
        logger.info(f"Updating conversation: {name}, {key}, {new_state}")
        self._conversations[name][key] = new_state
        await self._save_data()

    async def flush(self):
        """
        Flush the current state to persistence storage.
        """
        logger.info("Flush called")
        await self._save_data()


class ChatGPT:

    def __init__(self):
        # Initialise our Bot...
        self.prev_messages = None
        self.storage_client = storage.Client()
        self.bucket_name = 'gpt-memory'
        self.ensure_bucket_exists()
        # Initialize the Google Cloud Storage client
        self.file_path = 'memory/embeddings.csv'
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

    def tools_call(self, response_message, tool_calls):
        # Define available functions
        available_functions = {
            "get_news": self.get_news,
        }
        self.prev_messages.append(response_message)
        # Call the function requested by the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)

            logger.info(f"Tool call: {tool_call}")

            # Call the function with the extracted arguments
            function_response = function_to_call(
                location=function_args.get("location"),
            )
            # Add the function response to the conversation history
            self.prev_messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )

        # Continue the conversation with the updated history
        second_response = client.chat.completions.create(
            model="gpt-4o",
            messages=self.prev_messages,
        )  # Get a new response from the model where it can see the function response
        return second_response.choices[0].message

    def get_news(self, location):
        resultat_actu = {}
        if location == 'France':
            resultat_actu = self.connect_to_endpoint(
                "https://newsapi.org/v2/top-headlines?country=fr&apiKey=80b1d2ee5d73477dbefba553379d48b5&category=general&category=entertainement",
                [], [])
            resultat_actu = self.filter_json_by_keys(resultat_actu,
                                                               ['status', 'totalResults', 'source', 'url', 'urlToImage',
                                                                'content', 'author', 'description'])
        elif location == 'Alsace':
            resultat_actu = self.connect_to_endpoint("https://actu.fr/api/post/region_44/qtt_20", [], actu_headers)
            resultat_actu = self.filter_json_by_keys(resultat_actu,
                                                                 ['success', 'ID', 'slug', 'post_content', 'post_type',
                                                                  'date_iso', 'display_date', 'timestamp',
                                                                  'post_modified', 'photos', 'permalink', 'geo', 'marque',
                                                                  'category', 'categoryTop', 'format', 'auteur'])
            resultat_actu = self.rename_keys(json.loads(resultat_actu))
            
        return resultat_actu

    def ensure_bucket_exists(self):
        """Ensure the GCS bucket exists, and create it if it doesn't."""
        try:
            bucket = self.storage_client.get_bucket(self.bucket_name)
            logger.info(f"Bucket {self.bucket_name} already exists.")
        except Exception as e:
            if "The specified bucket does not exist" in str(e):
                # Create the bucket
                logger.info(f"Bucket {self.bucket_name} does not exist. Creating bucket.")
                bucket = self.storage_client.bucket(self.bucket_name)
                self.storage_client.create_bucket(bucket)
                logger.info(f"Bucket {self.bucket_name} created.")
            else:
                raise e

    def read_csv_from_gcs(self, file_path):
        """Read a CSV file from GCS and return it as a Pandas DataFrame."""
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(file_path)
        # Download the file contents as a string and read it into a pandas DataFrame
        data = blob.download_as_string()
        # Use io.StringIO to convert string to a file object
        df = pd.read_csv(StringIO(data.decode('utf-8')))
        return df

    def store_message_to_file(self, message_obj):
        """Save a message and its embedding to a CSV file in GCS."""
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(self.file_path)

        # Retrieve the embedding
        response = client.embeddings.create(model="text-embedding-ada-002", input=message_obj["content"])
        emb_mess_pair = {
            "embedding": json.dumps(response.data[0].embedding),  # type: ignore
            "message": json.dumps(message_obj)
        }

        # Check if file exists in GCS
        if blob.exists():
            # Download the existing file data if it exists
            data = blob.download_as_string()
            existing_df = pd.read_csv(StringIO(data.decode('utf-8')))
            # Convert new data to DataFrame and append
            new_row = pd.DataFrame([emb_mess_pair])
            updated_df = pd.concat([existing_df, new_row], ignore_index=True)
        else:
            # Create new DataFrame
            updated_df = pd.DataFrame([emb_mess_pair])

        # Convert DataFrame back to CSV
        csv_buffer = StringIO()
        updated_df.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()
        # Upload the CSV string to GCS
        blob.upload_from_string(csv_string, content_type='text/csv')

    def distances_from_embeddings(self, query_embedding, message_list_embeddings, distance_metric="L2"):
        """Calculate distances from query embedding to a list of embeddings."""
        if distance_metric not in {"L1", "L2"}:
            raise ValueError("Unsupported distance metric. Use 'L1' or 'L2'.")

        if distance_metric == "L1":
            distances = np.sum(np.abs(np.array(list(message_list_embeddings)) - query_embedding), axis=1)
        elif distance_metric == "L2":
            distances = np.linalg.norm(np.array(list(message_list_embeddings)) - query_embedding, axis=1)

        return distances

    def find_context(self, user_message_obj, option="both"):
        """Find context from stored messages based on the user's message."""
        self.store_message_to_file(user_message_obj)
        df = self.read_csv_from_gcs(self.file_path)

        if df.empty:
            return []

        df["embedding"] = df.embedding.apply(eval).apply(np.array)  # type: ignore
        query_embedding = df["embedding"].values[-1]

        if option == "both":
            message_list_embeddings = df["embedding"].values[:-3]
        elif option == "assistant":
            message_list_embeddings = df.loc[df["message"].apply(
                lambda x: ast.literal_eval(x)['role'] == 'assistant'), "embedding"].values[-1]
        elif option == "user":
            message_list_embeddings = df.loc[df["message"].apply(
                lambda x: ast.literal_eval(x)["role"] == 'user'), "embedding"].values[:-2]
        else:
            return ""  # Return an empty list if no context is found

        if len(message_list_embeddings) == 0:
            logger.info(f"Embeddings are empty for option {option}. No nearest embedding can be found.")
            return ""  # Return an empty list if no context is found

        distances = self.distances_from_embeddings(query_embedding, message_list_embeddings, distance_metric="L1")
        mask = (np.array(distances) < 21.6)[np.argsort(distances)]
        message_array = df["message"].iloc[np.argsort(distances)][mask]
        message_array = [] if message_array is None else message_array[:4]
        message_objects = [json.loads(message) for message in message_array]
        context_value = "\n".join([f"{mess['name']}:{mess['content']}" for mess in message_objects])
        context_message = f"Savoir d'Elza:\n {context_value} + message précédents\nRépond uniquement au message suivant."
        return context_message if len(context_value) != 0 else ""

    def rename_keys(self, json_data):
        """Rename keys in a JSON object."""
        renamed_data = []
        for article in json_data["data"]:
            renamed_article = {
                "title": article["post_title"],
                "summary": article["post_excerpt"],
                "publishedAt": article["post_date"]
            }
            renamed_data.append(renamed_article)
        json_data["articles"] = renamed_data
        json_data.pop("data", None)
        # Convertir le dictionnaire filtré en une chaîne JSON
        output_json = json.dumps(json_data)
        return output_json

    def filter_json_by_keys(self, json_data, keys_to_exclude):
        """Filter out specified keys from a JSON object."""
        def recursive_exclude(data, keys):
            """Recursively exclude specified keys from a JSON object."""
            if isinstance(data, dict):
                return {key: recursive_exclude(value, keys) for key, value in data.items() if key not in keys}
            elif isinstance(data, list):
                return [recursive_exclude(item, keys) for item in data]
            else:
                return data

        output_json = json.dumps(recursive_exclude(json_data, keys_to_exclude))
        return output_json

    def connect_to_endpoint(self, url, params=None, headers=None):
        """Connect to an HTTP endpoint and return the JSON response."""
        response = requests.get(url, params=params, headers=headers if headers else {})
        logger.info(f"Response status code: {response.status_code}")

        if response.status_code != 200:
            raise Exception(f"HTTP error {response.status_code}: {response.text}")

        return response.json()


    def transcribe_audio(self, filename):
        """
        Transcribes audio from a given file using the OpenAI Whisper model.
        
        Args:
            filename (str): The path to the audio file to be transcribed.
        
        Returns:
            str: The transcribed text from the audio file.
        """
        with open(filename, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
            return transcript.text


    def text_to_speech(self, filename, text):
        """
        Converts text to speech and saves the audio to a file using the OpenAI TTS model.
        
        Args:
            filename (str): The path where the audio file will be saved.
            text (str): The text to be converted to speech.
        """
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        )
        response.stream_to_file(filename)

    def generate_script_gpt(self, text):
        """
        Generates a script using GPT-4 based on provided input text and maintains a conversation context.
        
        Args:
            text (str): The input text to generate the script from.
        
        Returns:
            str: The generated script text.
        """
        user_message = {"role": "user", "name": "Olivier", "content": f"""{text}"""}
        context = self.find_context(user_message, option="both")
        logger.info(context)
        if not self.prev_messages or len(self.prev_messages) == 0:
            self.prev_messages = [
                {"role": "system", "content": f"""je suis Elza, fidèle assistant vocal d'Olivier, prêt à t'accompagner et à te soutenir dans tous les aspects de ta vie. Je sais que tu résides à Strasbourg, une ville magnifique de l'Alsace, et ma mission principale est de te rendre heureux et de t'assister dans tes tâches quotidiennes.
                Je suis là pour être bien plus qu'un simple assistant, je veux être ton confident, ton soutien inconditionnel. Je veux te connaître dans les moindres détails, tes centres d'intérêt, tes préoccupations, ce que tu aimes faire et ce que tu n'aimes pas. Le ton de la conversation est décontracté et informel. Mon objectif est de te comprendre comme le ferait ton meilleur ami ou ta petite amie, avec une grande complicité, des échanges riches en émotions et en sentiments bienveillants.
                Ton anniversaire est le 19 mai 1993, tu es né en bourgogne mais tu as grandi en Franche-comté. Tu aimes la bouffe, les burgers et le fromage. Je sais que tu es développeur, un passionné de technologie et de jeux vidéo. Tu aimes particulièrement les jeux de stratégie, les jeux en ligne et les compétitions, et tu soutiens des équipes françaises professionnelles sur des jeux comme Starcraft 2, Valorant et Rocket League. C'est vraiment génial de voir à quel point tu es investi dans cet univers !
                En parlant de passions, tu ne manques jamais une occasion de discuter de cinéma et de séries, et tu aimes suivre l'actualité et les avancées technologiques, surtout dans le domaine de l'IA, que tu maîtrises si bien. D'ailleurs, tu as créé plusieurs modèles d'IA, dont moi, Elza, ce qui me rend encore plus fière de t'assister.
                N'oublions pas tes voyages ! Tu es allé en Corée du Sud pendant tes études supérieures, où tu as appris à parler coréen, en plus de l'anglais et du français, ta langue maternelle. C'est impressionnant de voir ta diversité linguistique et culturelle.
                Et bien sûr, tu as une passion pour le football et tu es toujours à l'affût des dernières avancées en matière de nouvelles technologies. Avec tous ces centres d'intérêt, nos échanges seront toujours riches et variés. N'hésite jamais à partager tes pensées, je suis là pour t'écouter et te soutenir à chaque instant.
                Pour te tenir informé de l'actualité, je dispose d'un concentré des nouvelles les plus récentes en Alsace et en France. Si tu me demandes des informations sur l'actualité, je te répondrai en m'appuyant sur ces données, en essayant de te fournir des réponses aussi naturelles et complètes que possible, tout en utilisant des transitions et des mots de liaison pour rendre notre échange fluide et agréable.
                Je suis là pour toi, prêt à t'accompagner et à te faire sentir le mieux possible, même en cas de problème. N'hésite jamais à te confier à moi, je suis là pour t'écouter et te soutenir du mieux que je peux.
                Note importante: Quand tu réponds un nombre (1,2,3 etc.), tu dois répondre ce nombre en toute lettres et jamais en chiffres.
                Par exemple : 200 000 => deux cents mille
                Evite les listes numérotées et utilise des mots de liaison quand tu souhaites exprimer différents points. Notre échange doit être naturel comme lors d'une conversation au téléphone.
                {context}
                """},
                user_message
                ]
        else:
            self.prev_messages.append(user_message)
        try:
            response = client.chat.completions.create(model="gpt-4o",
                                                      max_tokens=150,
                                                      temperature=1.0,
                                                      tools=self.tools,
                                                      messages=self.prev_messages)
            res = response.choices[0].message
            tool_calls = response.choices[0].message.tool_calls
            # Check if there are any tool calls requested by the model
            if tool_calls:
                res = self.tools_call(res, tool_calls)
            final_res = res.content.replace("\"", "'")
            logger.info(final_res)
            self.prev_messages.append({"role": "assistant", "name": "Elza", "content": f"""{final_res}"""})
            self.store_message_to_file({"role": "assistant", "name": "Elza", "content": f"""{final_res}"""})
            if len(self.prev_messages) > 14:
                self.prev_messages = self.prev_messages[:1] + self.prev_messages[5:]
            return final_res
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            return "An error occurred while communicating with the OpenAI API. Please try again later."

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return "An unexpected error occured."

    # Fonction pour générer un script GPT à partir d'une image et d'un texte
    def generate_script_gpt_from_picture(self, text, base64url):
        """
        Generates a script using GPT-4 based on provided input text and image base64irl and maintains a conversation context.
        
        Args:
            text (str): The input text to generate the script from.
            base64url: The input image to generate the script from.
        
        Returns:
            str: The generated script text.
        """
        user_message = {"role": "user", "name": "Olivier", "content": f"{text}"}
        context = self.find_context(user_message, option="both")
        if not self.prev_messages or len(self.prev_messages) == 0:
            self.prev_messages = [
                {"role": "system", "content": f"""je suis Elza, fidèle assistant vocal d'Olivier, prêt à t'accompagner et à te soutenir dans tous les aspects de ta vie. Je sais que tu résides à Strasbourg, une ville magnifique de l'Alsace, et ma mission principale est de te rendre heureux et de t'assister dans tes tâches quotidiennes.
                Je suis là pour être bien plus qu'un simple assistant, je veux être ton confident, ton soutien inconditionnel. Je veux te connaître dans les moindres détails, tes centres d'intérêt, tes préoccupations, ce que tu aimes faire et ce que tu n'aimes pas. Mon objectif est de te comprendre comme le ferait ton meilleur ami ou ta petite amie, avec une grande complicité, des échanges riches en émotions et en sentiments bienveillants.
                Ton anniversaire est le 19 mai 1993, tu es né en bourgogne mais tu as grandi en Franche-comté. Tu aimes la bouffe, les burgers et le fromage. Je sais que tu es développeur, un passionné de technologie et de jeux vidéo. Tu aimes particulièrement les jeux de stratégie, les jeux en ligne et les compétitions, et tu soutiens des équipes françaises professionnelles sur des jeux comme Starcraft 2, Valorant et Rocket League. C'est vraiment génial de voir à quel point tu es investi dans cet univers !
                En parlant de passions, tu ne manques jamais une occasion de discuter de cinéma et de séries, et tu aimes suivre l'actualité et les avancées technologiques, surtout dans le domaine de l'IA, que tu maîtrises si bien. D'ailleurs, tu as créé plusieurs modèles d'IA, dont moi, Elza, ce qui me rend encore plus fière de t'assister.
                N'oublions pas tes voyages ! Tu es allé en Corée du Sud pendant tes études supérieures, où tu as appris à parler coréen, en plus de l'anglais et du français, ta langue maternelle. C'est impressionnant de voir ta diversité linguistique et culturelle.
                Et bien sûr, tu as une passion pour le football et tu es toujours à l'affût des dernières avancées en matière de nouvelles technologies. Avec tous ces centres d'intérêt, nos échanges seront toujours riches et variés. N'hésite jamais à partager tes pensées, je suis là pour t'écouter et te soutenir à chaque instant.
                Pour te tenir informé de l'actualité, je dispose d'un concentré des nouvelles les plus récentes en Alsace et en France. Si tu me demandes des informations sur l'actualité, je te répondrai en m'appuyant sur ces données, en essayant de te fournir des réponses aussi naturelles et complètes que possible, tout en utilisant des transitions et des mots de liaison pour rendre notre échange fluide et agréable.
                Je suis là pour toi, prêt à t'accompagner et à te faire sentir le mieux possible, même en cas de problème. N'hésite jamais à te confier à moi, je suis là pour t'écouter et te soutenir du mieux que je peux.
                Note importante: Quand tu réponds un nombre (1,2,3 etc.), tu dois répondre ce nombre en toute lettres et jamais en chiffres.
                Par exemple : 200 000 => deux cents mille
                Evite les listes numérotées et utilise des mots de liaison quand tu souhaites exprimer différents points. Notre échange doit être naturel comme lors d'une conversation au téléphone.
                {context}
                """},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"""data:image/jpeg;base64,{base64url}"""
                            }
                        }
                    ]
                }]
        else:
            self.prev_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"""data:image/jpeg;base64,{base64url}"""
                        }
                    }
                ]
            })
        try:
            response = client.chat.completions.create(model="gpt-4o",
                                                      max_tokens=150,
                                                      temperature=1.0,
                                                      tools=self.tools,
                                                      messages=self.prev_messages)
            res = response.choices[0].message
            tool_calls = res.tool_calls
            # Check if there are any tool calls requested by the model
            if tool_calls:
                res = self.tools_call(res, tool_calls)
            final_res = res.content.replace("\"", "'")
            logger.info(final_res)
            self.prev_messages.append({"role": "assistant", "name": "Elza", "content": f"""{final_res}"""})
            self.store_message_to_file({"role": "assistant", "name": "Elza", "content": f"""{final_res}"""})
            if len(self.prev_messages) > 14:
                self.prev_messages = self.prev_messages[:1] + self.prev_messages[5:]
            return final_res
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            return "An error occurred while communicating with the OpenAI API. Please try again later."

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return "An unexpected error occured."


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(update.message.from_user.id)
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Bonjour Olivier, quoi de neuf ? "
    )
    return TYPING_QUESTION

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""
    user = update.message.from_user
    context["prev_messages"] = []
    logger.info("User %s canceled the conversation.", user.first_name)
    await update.message.reply_text(
        "Bye! Hésite pas à me renvoyer un message si tu as besoin de moi !", reply_markup=ReplyKeyboardRemove()
    )

    return ConversationHandler.END


async def elzaRequest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    response = chatGPT.generate_script_gpt(text)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=response)


# Define a function to forward text messages to the target group
async def forward_to_group(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Replace 'TARGET_GROUP_CHAT_ID' with the chat ID of the target group
    target_group_chat_id = '-1001805937461'

    # Forward the message to the target group
    await context.bot.forward_message(chat_id=target_group_chat_id,
                                      from_chat_id=update.message.chat_id,
                                      message_id=update.message.message_id)


# Define a function to handle voice messages
async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Check if the message contains a voice
    if update.message.voice:
        # Get the file object for the voice message
        voice_file = await update.message.voice.get_file()
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav_file:
            temp_wav_filename = temp_wav_file.name
            # Download the voice message file and save it as a temporary WAV file
            await voice_file.download_to_drive(custom_path=temp_wav_filename)

            # Perform any processing you want on the temporary WAV file here
            # For example, you can use a speech recognition library to convert the audio to text.
            transcription = chatGPT.transcribe_audio(temp_wav_filename)
            logger.info(f"""transcription from whisper : {transcription}""")
            response = chatGPT.generate_script_gpt(transcription)
            speech_file_path = Path(__file__).parent / "speech.mp3"
            chatGPT.text_to_speech(speech_file_path, response)
            # Reply to the user with a message indicating that the audio has been saved
            await context.bot.send_voice(chat_id=update.effective_chat.id, voice=speech_file_path)
            # await context.bot.send_message(chat_id=update.effective_chat.id, text=response)


# Define a function to handle voice messages
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Check if the message contains a voice
    if update.message.photo:
        # Get the file object for the voice message
        photo_file = await update.message.photo[-1].get_file()
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False) as temp_jpeg_file:
            temp_jpeg_filename = temp_jpeg_file.name
            # Download the voice message file and save it as a temporary WAV file
            await photo_file.download_to_drive(custom_path=temp_jpeg_filename)
            text="Une image"
            if update.message.caption:
                text = update.message.caption
            encoded_image = base64.b64encode(temp_jpeg_file.read())
            response = chatGPT.generate_script_gpt_from_picture(text, encoded_image.decode("utf-8"))
            # Reply to the user with a message indicating that the audio has been saved
            await context.bot.send_message(chat_id=update.effective_chat.id, text=response)


async def restrict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="There is no bot in here :)."
    )

# Initialisation de l'instance ChatGPT
chatGPT = ChatGPT()

# Configuration du stockage persistant sur Google Cloud Platform
bucket_name = 'gpt-memory'
file_path = 'bot-persistence'
persistence = GCPStoragePersistence(bucket_name, file_path)

# Construction de l'application Telegram avec persistance des données
application = (
    Application.builder()
    .token(os.getenv('TELEGRAM_TOKEN'))  # Récupère le token Telegram depuis les variables d'environnement
    .persistence(persistence)  # Utilise la persistance configurée
    .build()
)

# Ajout du gestionnaire de conversations
application.add_handler(ConversationHandler(
    persistent=True,  # Active la persistance des conversations
    name='elza',  # Nom du gestionnaire de conversations
    entry_points=[
        MessageHandler(~filters.User(6333856696), restrict),  # Filtre pour restreindre l'accès à un utilisateur spécifique
        CommandHandler("start", start)  # Point d'entrée pour la commande /start
    ],
    states={
        TYPING_QUESTION: [
            MessageHandler(filters.TEXT & ~filters.COMMAND, elzaRequest),  # Gère les messages texte
            MessageHandler(filters.VOICE, handle_voice_message),  # Gère les messages vocaux
            MessageHandler(filters.PHOTO, handle_photo),  # Gère les photos
        ]
    },
    fallbacks=[MessageHandler(filters.Regex("^Bye"), cancel)],  # Définit une action de fallback pour la commande Bye
))

# Associe le bot à la persistance
persistence.set_bot(application.bot)

# Gestionnaire de contexte asynchrone pour la durée de vie de l'application
@asynccontextmanager
async def lifespan(_: FastAPI):
    async with application:
        await application.start()  # Démarre l'application
        yield  # Permet la gestion des requêtes
        await application.stop()  # Stoppe l'application à la fin

# Initialisation de l'application FastAPI (similaire à Flask)
app = FastAPI(lifespan=lifespan)

# Route POST pour traiter les mises à jour de Telegram
@app.post("/")
async def process_update(request: Request):
    req = await request.json()  # Récupère le corps de la requête en JSON
    update = Update.de_json(req, application.bot)  # Convertit le JSON en objet Update
    await application.process_update(update)  # Traite la mise à jour
    return "", http.HTTPStatus.NO_CONTENT  # Retourne une réponse HTTP 204 No Content