import numpy as np
from openai import OpenAI
import os
import openai
import pandas as pd
import pickle
import logging
from rapidfuzz import process
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_embeddings(file_path): 
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

    # Save embeddings to a pickle file
    with open("embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    logging.info("Embeddings generated and saved to embeddings.pkl")
    return embeddings

def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)
    logging.info(f"Loaded embeddings from {file_path}")
    return embeddings

def validate_user_input(input_shows, available_shows):
    corrected_shows = []
    for show in input_shows:
        match = process.extractOne(show, available_shows)
        if match and match[1] > 70:  # Lowered threshold to 70% to be more lenient
            corrected_shows.append(match[0])
            logging.info(f"Input '{show}' matched to '{match[0]}' with confidence {match[1]}%.")
        else:
            logging.warning(f"Input '{show}' did not match any available shows.")
    if not corrected_shows:
        return None  # Return None if no matches were found
    return corrected_shows

def calculate_user_vector(selected_shows, embeddings):
    vectors = [embeddings[show] for show in selected_shows if show in embeddings]
    if not vectors:
        logging.error("No valid shows found in embeddings.")
        return None
    logging.info(f"Calculating user vector based on selected shows: {selected_shows}")
    return np.mean(vectors, axis=0)

def generate_recommendations(user_vector, show_vectors, show_titles, top_n=5):
    if user_vector is None:
        logging.error("User vector is None. Cannot generate recommendations.")
        return []
    similarities = cosine_similarity([user_vector], show_vectors)[0]
    sorted_indices = np.argsort(similarities)[::-1][:top_n]
    recommendations = [(show_titles[i], round(similarities[i] * 100, 2)) for i in sorted_indices]
    logging.info(f"Generated top {top_n} recommendations.")
    return recommendations


def main():
    # Load embeddings or generate if not present
    try:
        embeddings = load_embeddings("embeddings.pkl")
    except FileNotFoundError:
        logging.warning("Embeddings file not found. Generating embeddings...")
        embeddings = generate_embeddings("imdb_tvshows.csv")

    show_list = list(embeddings.keys())

if __name__ == "__main__":
    main()