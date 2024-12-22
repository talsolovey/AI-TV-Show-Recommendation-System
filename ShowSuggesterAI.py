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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

    # Convert similarity to a percentage and then we sort them (descending).
    show_scores = [(title, similarities[i] * 100) for i, title in enumerate(show_titles)]

    # Exclude user’s input shows + sort in descending order of similarity
    show_scores = [item for item in show_scores if item[0] not in excluded_titles]
    show_scores.sort(key=lambda x: x[1], reverse=True)

    # Take the top_n
    return show_scores[:top_n]
    

def create_fictional_show_name_and_description(basis):
    """
    Creates a fictional TV show name and description based on the provided 'basis' string.
    """
    show_name = f"{basis} Universe"
    show_description = (
        f"A thrilling new series inspired by '{basis}'. "
         "Get ready for suspense, drama, and adventure!"
    )
    return show_name, show_description

def generate_lightx_images(show1name, show1description, show2name, show2description):
    """
    Uses the LightX image generation API to create poster images for two fictional shows.
    """
    logging.info("Generating LightX images for two fictional shows...")

    # Retrieve API key from environment variable
    api_key = os.getenv('X_API_KEY')
    if not api_key:
        logging.error("LightX API key is not set in environment variables")

    # LightX API endpoint
    url = 'https://api.lightxeditor.com/external/api/v1/text2image'

    headers = {
        'Content-Type': 'application/json',
        'x-api-key': api_key
    }

    # Build the prompts for each show
    prompt_show1 = (
        f"Create an eye-catching poster or ad for a TV show titled '{show1name}' "
        f"that is about '{show1description}'. "
        "Use a cinematic style with bold, striking visuals."
    )

    prompt_show2 = (
        f"Create a dynamic, cinematic poster or ad for a TV show named '{show2name}' "
        f"that is about '{show2description}'. "
        "Include neon lighting effects for a modern, futuristic vibe."
    )

    headers = {
        'Content-Type': 'application/json',
        'x-api-key': api_key
    }

    # 1) Generate image for Show #1
    logging.info(f"Sending Show #1 prompt to LightX: {prompt_show1}")
    response1 = requests.post(url, headers=headers, json={"textPrompt": prompt_show1})
    if response1.status_code == 200:
        try:
            data1 = response1.json()
            show1_image_path = data1.get("imageUrl", "show1_ad_no_url.jpg")
            logging.info(f"LightX success for Show #1. Image URL: {show1_image_path}")
        except ValueError as e:
            logging.error("Failed to parse JSON from LightX response for Show #1.")
            logging.error(str(e))
    else:
        logging.error(f"LightX generation for Show #1 failed: {response1.status_code}")
        logging.error(response1.text)
    
    # 2) Generate image for Show #2
    logging.info(f"Sending Show #2 prompt to LightX: {prompt_show2}")
    response2 = requests.post(url, headers=headers, json={"textPrompt": prompt_show2})
    if response2.status_code == 200:
        try:
            data2 = response2.json()
            show2_image_path = data2.get("imageUrl", "show2_ad_no_url.jpg")
            logging.info(f"LightX success for Show #2. Image URL: {show2_image_path}")
        except ValueError as e:
            logging.error("Failed to parse JSON from LightX response for Show #2.")
            logging.error(str(e))
    else:
        logging.error(f"LightX generation for Show #2 failed: {response2.status_code}")
        logging.error(response2.text)

    # Return paths/URLs
    return show1_image_path, show2_image_path



def main():
    """
    Main function implementing the full user flow:
    1) Ask for user input (loved shows)
    2) Fuzzy match and confirm
    3) Generate recommendations
    4) Print recommended shows
    5) Print two newly created shows + show ads
    """
    pickle_file_path = "embeddings.pkl"
    csv_file_path = "imdb_tvshows.csv"

    # STEP: Attempt to load embeddings; if not found, generate them
    try:
        embeddings = load_embeddings(pickle_file_path)
    except FileNotFoundError:
        logging.warning("Embeddings file not found. Generating embeddings from CSV...")
        embeddings = generate_embeddings(csv_file_path)

    show_titles = list(embeddings.keys())


if __name__ == "__main__":
    main()