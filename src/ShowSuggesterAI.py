# src/SuggesterAI.py

import subprocess
import time
import numpy as np
from openai import OpenAI
import os
import openai
import pandas as pd
import pickle
import logging
from rapidfuzz import process
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_embeddings(file_path): 
    """
    Generates embeddings for each TV show in the CSV file using the specified OpenAI model.
    Saves a dictionary of {title: embedding_vector} to a pickle file.
    """
    api_key=os.getenv('OPENAI_API_KEY')
    if not api_key:
        logging.error("OpenAI API key not set in environment variables.")
        raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")
    openai.api_key = api_key
    client = OpenAI()
    
    # Load TV show data
    shows_df = pd.read_csv(file_path)

    # Create embeddings dictionary
    embeddings = {}
    for index, row in shows_df.iterrows():
        title = row['Title']
        description = row['Description']
        response = client.embeddings.create(
            input=description,
            model="text-embedding-3-small"
        )
        embeddings[title] = response.data[0].embedding
        logging.info(f"Generated embedding for: {title}")

    # Save embeddings dict to a pickle
    with open("embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    logging.info("Embeddings generated and saved to embeddings.pkl")
    return embeddings


def load_embeddings(file_path):
    """
    Loads and returns a dictionary of embeddings from a pickle file.
    """
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)
    logging.info(f"Loaded embeddings from {file_path}")
    return embeddings


def validate_user_input(input_shows, available_shows):
    """
    Uses fuzzy matching to map user-typed show names to known show titles.
    Returns a list of corrected shows or None if no valid matches were found.
    """
    corrected_shows = []
    for show in input_shows:
        match = process.extractOne(show, available_shows)
        if match and match[1] > 70:
            corrected_shows.append(match[0])
            logging.info(f"Input '{show}' matched to '{match[0]}' with confidence {match[1]}%.")
        else:
            logging.warning(f"Input '{show}' did not match any available shows.")
    if not corrected_shows:
        return None  # Return None if no matches were found
    return corrected_shows


def calculate_user_vector(selected_shows, embeddings):
    """
    Given a list of confirmed show names and their embeddings,
    returns the average embedding vector for these shows.
    """
    vectors = []
    for show in selected_shows:
        if show in embeddings:
            vectors.append(embeddings[show])
        else:
            logging.warning(f"Show '{show}' not found in embeddings dictionary.")
    if not vectors:
        logging.error("No valid shows found in embeddings for the user selection.")
        return None
    logging.info(f"Calculating user vector based on selected shows: {selected_shows}")
    return np.mean(vectors, axis=0)


def generate_recommendations(user_vector, show_vectors, show_titles, excluded_titles, top_n=5):
    """
    Given a user vector, list of show vectors, and corresponding show titles,
    returns a list of (show_title, percentage_score) of the top N recommendations,
    excluding any titles in 'excluded_titles'.
    """
    if user_vector is None:
        logging.error("User vector is None. Cannot generate recommendations.")
        return []
    
    # Compute cosine similarity for each show
    similarities = cosine_similarity([user_vector], show_vectors)[0]

    # Scale the similarities so that min -> 0%, max -> 100%
    min_sim = float(np.min(similarities))
    max_sim = float(np.max(similarities))

    # Avoid divide-by-zero if all similarities are the same
    if max_sim > min_sim:
        scaled_sims = [(sim - min_sim) / (max_sim - min_sim) for sim in similarities]
    else:
        # If all similarities are identical, just set them all to 1.0 or 0.5, etc.
        scaled_sims = [1.0 for _ in similarities]

    # Convert similarity to a percentage
    show_scores = [(title, scaled_sims[i] * 100) for i, title in enumerate(show_titles)]

    # Exclude the user’s input shows
    show_scores = [item for item in show_scores if item[0] not in excluded_titles]

    # Sort in descending order of the scaled similarity
    show_scores.sort(key=lambda x: x[1], reverse=True)

    return show_scores[:top_n]
    

def create_fictional_show_name_and_description(basis):
    """
    Creates a fictional TV show name and description based on the provided 'basis' string.
    """
    default_show = (f"{basis[0]} Universe", f"A thrilling series loosely inspired by '{basis}'.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OpenAI API key not set. Returning fallback show info.")
        return default_show

    openai.api_key = api_key
    client = OpenAI()

    # Build a prompt instructing the model to return valid JSON
    system_message = (
        "You are a creative TV show writer. "
        "I want you to create a new TV show concept in JSON format."
    )
    user_message = (
        f"Please create a fictional TV show inspired by '{basis}'. "
        "Return your response in **valid JSON** with the following structure:\n\n"
        "{\n"
        '  "title": "<the show title>",\n'
        '  "description": "<a short, punchy description>"\n'
        "}\n\n"
        "Do not include any extra keys or text outside the JSON."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
        )

        # Extract the assistant's message content
        response_text = response.choices[0].message.content.strip()

        # Attempt to parse the JSON
        try:
            data = json.loads(response_text)
            show_name = data.get("title", "").strip()
            show_description = data.get("description", "").strip()

            if not show_name:
                show_name = f"{basis[0]} Universe"
            if not show_description:
                show_description = f"An adventurous saga inspired by '{basis}'."

            return (show_name, show_description)

        except json.JSONDecodeError as e:
            logging.error("Failed to parse JSON from LLM response.")
            logging.error(f"Raw response was: {response_text}")
            return default_show

    except Exception as ex:
        logging.error(f"Error calling OpenAI ChatCompletion: {ex}")
        return default_show


def generate_lightx_image(show_name, show_description):
    """
    Uses the LightX image generation API to create poster images for two fictional shows.
    """
    logging.info("Generating LightX images for two fictional shows...")

    # Retrieve API key from environment variable
    api_key = os.getenv('LIGHTX_API_KEY')
    if not api_key:
        logging.error("LightX API key is not set in environment variables")

    # LightX API endpoint
    text2image_url = 'https://api.lightxeditor.com/external/api/v1/text2image'
    # Order status endpoint
    order_status_url = "https://api.lightxeditor.com/external/api/v1/order-status"

    headers = {
        'Content-Type': 'application/json',
        'x-api-key': api_key
    }

    # Build the prompts for each show
    prompt_show = (
        f"Design a bold, cinematic poster for the new TV show '{show_name}'. "
        f"Feature the title prominently and depict the key characters in a dramatic pose, "
        f"reflecting the storyline: '{show_description}'. "
        "Use striking visuals, high-contrast lighting, and include a short, catchy tagline "
        "that teases the main theme."
    )

    # helper function to send a request to LightX API
    def send_lightx_request(prompt):
        """
        Calls the LightX API with the given prompt and returns the image URL.
        """
        data = {"textPrompt": prompt}

        try:
            logging.info(f"Sending request to text2image with prompt: {prompt}")
            response = requests.post(text2image_url, headers=headers, json=data)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("statusCode") == 2000:
                    body = data.get("body", {})
                    order_id = body.get("orderId")
                if not order_id:
                    logging.error("No 'orderId' in text2image response.")
                    return ""
                logging.info(f"Order created successfully. orderId={order_id}")
                return order_id
            else:
                logging.error(f"text2image request failed: {response.status_code} {response.text}")
                return ""
        except Exception as e:
            logging.error(f"Exception during create_order: {e}")
            return ""
    
    # helper function to retrieve the image URL
    def retrieve_image_url(order_id):
        """
        Retrieves the image URL for a given order ID.
        """
        payload = {"orderId": order_id}

        max_retries = 5
        for attempt in range(1, max_retries + 1):
            logging.info(f"Attempt {attempt} of {max_retries}: Checking order status for orderId={order_id}")
            try:
                response = requests.post(order_status_url, headers=headers, data=json.dumps(payload))

                if response.status_code == 200:
                    data = response.json()

                    # Check if the request to LightX was successful
                    if data.get("statusCode") == 2000:
                        body = data.get("body", {})
                        current_status = body.get("status")
                        logging.info(f"Current status: {current_status}")

                        # If status is active or failed, return immediately
                        if current_status in ["active", "failed"]:
                            output_url = body.get("output")
                            if not output_url:
                                logging.error(f"Failed to retrieve output URL for order ID: {order_id}")
                                return None
                            return output_url
                        else:
                            # sleep 3s, then retry
                            time.sleep(3)
            except Exception as e:
                logging.error(f"Exception during retrieve_image_url: {e}")
                return None

    # Send request
    show_order_id = send_lightx_request(prompt_show)

    # Retrieve image URLs
    show_image_url = retrieve_image_url(show_order_id)

    return show_image_url


def download_and_open_imageURL(image_url, filename):
    """
    Downloads an image from a URL and opens it using the default image viewer.
    """
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        subprocess.run(["open", filename], check=True)
    else:
        logging.error(f"Failed to download image from URL: {image_url}")