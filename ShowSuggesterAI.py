from openai import OpenAI
import os
import openai
import pandas as pd
import pickle
import logging

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

def validate_user_input(user_shows, available_shows):
    raise NotImplementedError("This method is not yet implemented")

def calculate_user_vector(selected_shows, embeddings):
    raise NotImplementedError("This method is not yet implemented")

def generate_recommendations(user_vector, show_vectors, show_titles, top_n=5):
    raise NotImplementedError("This method is not yet implemented")


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