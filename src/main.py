#src/main.py

import logging
from ShowSuggesterAI import(
    generate_embeddings,
    load_embeddings,
    validate_user_input,
    calculate_user_vector,
    generate_recommendations,
    create_fictional_show_name_and_description,
    generate_lightx_image,
    download_and_open_imageURL
)

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
    show1_basis = ", ".join(user_shows)
    show1name, show1description = create_fictional_show_name_and_description(show1_basis)

    # Show #2: based on the first recommended show
    show2_basis = ", ".join(user_shows)
    show2name, show2description = create_fictional_show_name_and_description(show2_basis)

    #show1_ad_path = generate_lightx_image(
    #    show1name, show1description
    #)
    #show2_ad_path = generate_lightx_image(
    #    show2name, show2description
    #)

    print("\nI have also created just for you two shows which I think you would love.")
    print("Show #1 is based on the fact that you loved the input shows that you gave me.")
    print(f"Its name is {show1name} and it is about {show1description}.")
    print("Show #2 is based on the shows that I recommended for you.")
    print(f"Its name is {show2name} and it is about {show2description}.")

    print("Here are also the 2 TV show ads. Hope you like them!")
    #print(f" - Ad for Show #1 is saved at: {show1_ad_path}")
    #print(f" - Ad for Show #2 is saved at: {show2_ad_path}")
    download_and_open_imageURL("https://d3aa3s3yhl0emm.cloudfront.net/text2image/692ec19d-bad2-4150-a913-8e44736bc086.jpg", "show1_ad.jpg")

if __name__ == "__main__":
    main()
