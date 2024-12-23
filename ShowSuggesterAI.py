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
    prompt_show1 = (
        f"Create an eye-catching poster or ad for a TV show titled '{show1name}' "
        f"that is about '{show1description}'. "
        "Use a cinematic style with bold, striking visuals."
    )

    prompt_show2 = (
        f"Design a captivating poster or ad for a TV show called '{show2name}' "
        f"that is inspired by '{show2description}'. "
        "The show features a mix of mystery, intrigue, and action."
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

    # Send requests for both shows
    show1_order_id = send_lightx_request(prompt_show1)
    show2_order_id = send_lightx_request(prompt_show2)

    # Retrieve image URLs
    show1_image_url = retrieve_image_url(show1_order_id)
    show2_image_url = retrieve_image_url(show2_order_id)

    return show1_image_url, show2_image_url


def main():
    """
    Main function implementing the full user flow:
    1) Ask for user input (loved shows)
    2) Fuzzy match and confirm
    3) Generate recommendations
    4) Print recommended shows
    5) Print two newly created shows + show ads
    """

    # Paths
    pickle_file_path = "embeddings.pkl"
    csv_file_path = "imdb_tvshows.csv"

    # Attempt to load embeddings; if not found, generate them
    try:
        embeddings = load_embeddings(pickle_file_path)
    except FileNotFoundError:
        logging.warning("Embeddings file not found. Generating embeddings from CSV...")
        embeddings = generate_embeddings(csv_file_path)

    show_titles = list(embeddings.keys())

    # Repeat until user confirms
    while True:
        user_input = input("Which TV shows did you really like watching? Separate them by a comma."
                           "Make sure to enter more than 1 show:\n")
        user_shows_raw = [s.strip() for s in user_input.split(",")]

        if len(user_shows_raw) < 2:
            print("Please enter more than one show. Let's try again.\n")
            continue

         # Fuzzy match
        corrected_shows = validate_user_input(user_shows_raw, show_titles)
        if corrected_shows:
            # Confirm
            print(f"\nMaking sure, do you mean {', '.join(corrected_shows)}? (y/n)")
            confirmation = input().lower().strip()
            if confirmation == 'y':
                user_shows = corrected_shows
                break
        print("\nSorry about that. Let's try again, please make sure to write the names of the TV shows correctly.\n")

    print("\nGreat! Generating recommendations now...\n")

    # Calculate user vector
    user_vector = calculate_user_vector(user_shows, embeddings)
    if user_vector is None:
        print("No valid user shows found. Exiting.")
        return
    
    # Prepare data for recommendations
    show_vectors = list(embeddings.values())
    show_titles_list = list(embeddings.keys())

    # Generate top 5 recommendations
    recommendations = generate_recommendations(
        user_vector=user_vector,
        show_vectors=show_vectors,
        show_titles=show_titles_list,
        excluded_titles=user_shows, # Exclude user's input shows
        top_n=5
    )

    # Step #4: Print the recommended shows in the desired format
    print("Here are the TV shows that I think you would love:")
    for show_title, score in recommendations:
        print(f"{show_title} ({round(score, 2)}%)")

    # Step #5: Create two fictional shows and generate LightX image ads
    # Show #1: based on user input
    show1_basis = user_shows[0]
    show1name, show1description = create_fictional_show_name_and_description(user_shows[0])

    # Show #2: based on the first recommended show
    show2_basis = recommendations[0][0]
    show2name, show2description = create_fictional_show_name_and_description(recommendations)

    show1_ad_path = generate_lightx_images(
        show1name, show1description
    )
    show2_ad_path = generate_lightx_images(
        show2name, show2description
    )

    print("\nI have also created just for you two shows which I think you would love.")
    print("Show #1 is based on the fact that you loved the input shows that you gave me.")
    print(f"Its name is {show1name} and it is about {show1description}.")
    print("Show #2 is based on the shows that I recommended for you.")
    print(f"Its name is {show2name} and it is about {show2description}.")

    print("Here are also the 2 TV show ads. Hope you like them!")
    print(f" - Ad for Show #1 is saved at: {show1_ad_path}")
    print(f" - Ad for Show #2 is saved at: {show2_ad_path}")

if __name__ == "__main__":
    main()