import pytest
import numpy as np
import pickle
from unittest.mock import patch, MagicMock
from ShowSuggesterAI import (
    load_embeddings,
    validate_user_input,
    calculate_user_vector,
    generate_recommendations,
    create_fictional_show_name_and_description,
    generate_lightx_images
)

@pytest.fixture
def sample_embeddings():
    """
    Provide a simple embeddings dictionary for testing.
    """
    return {
        "Game of Thrones": np.array([0.1, 0.2, 0.3]),
        "Sacred Games": np.array([0.4, 0.5, 0.6]),
        "Breaking Bad": np.array([0.7, 0.8, 0.9]),
    }

# -----------------------------------------------------------------------------
# Load Embeddings Tests
# -----------------------------------------------------------------------------
def test_load_embeddings_returns_dict(tmp_path):
    # Create a temporary pickle file with dummy embeddings
    embeddings = {"Game of Thrones": [0.1, 0.2, 0.3]}
    file_path = tmp_path/ "embeddings.pkl"
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    # Check that load_embeddings successfully returns a dictionary
    loaded_embeddings = load_embeddings(str(file_path))
    assert isinstance(loaded_embeddings, dict)


def test_load_embeddings_contains_expected_key(tmp_path):
    # Create a temporary pickle file with dummy embeddings
    embeddings = {"Game of Thrones": [0.1, 0.2, 0.3]}
    file_path = tmp_path / "embeddings.pkl"
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)

   # Check that the loaded dictionary has the specific key
    loaded_embeddings = load_embeddings(str(file_path))
    assert "Game of Thrones" in loaded_embeddings

# -----------------------------------------------------------------------------
# Validate User Input Tests
# -----------------------------------------------------------------------------
def test_validate_user_input_matches_correct_show():
    shows = ["Game of Thrones", "Sacred Games", "Breaking Bad"]
    corrected_shows = validate_user_input(["game of thron"], shows)
    # Expecting fuzzy match to "Game of Thrones"
    assert corrected_shows == ["Game of Thrones"]

def test_validate_user_input_multiple_matches():
    shows = ["Game of Thrones", "Sacred Games", "Breaking Bad"]
    corrected_shows = validate_user_input(["game of thron", "sacred gmes"], shows)
    # Expecting multiple fuzzy matches
    assert corrected_shows == ["Game of Thrones", "Sacred Games"]

def test_validate_user_input_no_match():
    shows = ["Game of Thrones", "Sacred Games", "Breaking Bad"]
    corrected_shows = validate_user_input(["unknown show"], shows)
    # No match -> returns None
    assert corrected_shows is None

# -----------------------------------------------------------------------------
# Calculate User Vector Tests
# -----------------------------------------------------------------------------
def test_calculate_user_vector_correct_average(sample_embeddings):
    selected_shows = ["Game of Thrones", "Sacred Games"]
    user_vector = calculate_user_vector(selected_shows, sample_embeddings)
    expected_vector = np.mean([
        sample_embeddings["Game of Thrones"],
        sample_embeddings["Sacred Games"]
        ], axis=0)
    assert np.allclose(user_vector, expected_vector)

def test_calculate_user_vector_ignores_unknown(sample_embeddings):
    selected_shows = ["Game of Thrones", "Unknown Show"]
    user_vector = calculate_user_vector(selected_shows, sample_embeddings)
    expected_vector = sample_embeddings["Game of Thrones"]
    # Only "Game of Thrones" is valid, so user_vector should match that embedding
    assert np.allclose(user_vector, expected_vector)

def test_calculate_user_vector_none_when_no_valid_shows(sample_embeddings):
    selected_shows = ["Unknown1", "Unknown2"]
    user_vec = calculate_user_vector(selected_shows, sample_embeddings)
    assert user_vec is None

# -----------------------------------------------------------------------------
# Generate Recommendations Tests
# -----------------------------------------------------------------------------
def test_generate_recommendations_length(sample_embeddings):
    user_vector = np.array([0.2, 0.3, 0.4])
    show_vectors = list(sample_embeddings.values())
    show_titles = list(sample_embeddings.keys())
    # Request top 2 recommendations
    recommendations = generate_recommendations(
        user_vector, show_vectors, show_titles, excluded_titles=[],  top_n=2
    )
    assert len(recommendations) == 2

def test_generate_recommendations_excludes_shows(sample_embeddings):
    user_vector = np.array([0.2, 0.3, 0.4])
    show_vectors = list(sample_embeddings.values())
    show_titles = list(sample_embeddings.keys())
    # Exclude a specific show from recommendations
    recommendations = generate_recommendations(
        user_vector, show_vectors, show_titles, excluded_titles=["Game of Thrones"], top_n=5
    )
    # Check "Game of Thrones" isn't in results
    assert all(r[0] != "Game of Thrones" for r in recommendations)

def test_generate_recommendations_sorts_order(sample_embeddings):
    user_vector = np.array([0.2, 0.3, 0.4])
    show_vectors = list(sample_embeddings.values())
    show_titles = list(sample_embeddings.keys())
    recommendations = generate_recommendations(
        user_vector, show_vectors, show_titles, excluded_titles=[], top_n=3
    )
    # Extract the similarity (2nd item in each tuple) and ensure it is descending
    similarities = [rec[1] for rec in recommendations]
    assert similarities == sorted(similarities, reverse=True)


# -----------------------------------------------------------------------------
# create_fictional_show_name_and_description Tests
# -----------------------------------------------------------------------------
def test_create_fictional_show_name_and_description_returns_strings():
    """
    Test that the function returns the show name as a string value.
    """
    show_name, _ = create_fictional_show_name_and_description("Breaking Bad")
    assert isinstance(show_name, str)

def test_create_fictional_show_name_and_description_returns_strings():
    """
    Test that the function returns the show description as a string value.
    """
    _, show_description = create_fictional_show_name_and_description("Breaking Bad")
    assert isinstance(show_description, str)

def test_create_fictional_show_name_and_description_includes_basis():
    """
    Test that the returned description includes the basis text (if provided).
    """
    basis_text = "Breaking Bad"
    _, show_description = create_fictional_show_name_and_description(basis_text)
    assert basis_text in show_description


# -----------------------------------------------------------------------------
# generate_lightx_images Tests
# -----------------------------------------------------------------------------
@patch("ShowSuggesterAI.requests.post")
def test_generate_lightx_images_show1_path(mock_post):
    """
    Ensures generate_lightx_images returns the correct path for Show #1.
    """
    # Mock both calls. The first call => Show #1, second => Show #2
    mock_resp1 = MagicMock(status_code=200)
    mock_resp1.json.return_value = {"imageUrl": "fake_path_show1.jpg"}
    mock_resp2 = MagicMock(status_code=200)
    mock_resp2.json.return_value = {"imageUrl": "fake_path_show2.jpg"}
    mock_post.side_effect = [mock_resp1, mock_resp2]

    show1_ad_path, _ = generate_lightx_images("Show1", "Desc1", "Show2", "Desc2")
    assert show1_ad_path == "fake_path_show1.jpg"


@patch("ShowSuggesterAI.requests.post")
def test_generate_lightx_images_calls_lightx_twice(mock_post):
    """
    Ensures generate_lightx_images calls requests.post exactly two times.
    """
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"imageUrl": "fake_path_showN.jpg"}
    mock_post.return_value = mock_resp

    generate_lightx_images("Show1", "Desc1", "Show2", "Desc2")
    assert mock_post.call_count == 2


@patch("ShowSuggesterAI.requests.post")
def test_generate_lightx_images_prompt_first_call(mock_post):
    """
    Ensures the JSON payload for the first call contains the correct prompt for Show #1.
    """
    mock_resp1 = MagicMock(status_code=200)
    mock_resp1.json.return_value = {"imageUrl": "fake_path_show1.jpg"}
    mock_resp2 = MagicMock(status_code=200)
    mock_resp2.json.return_value = {"imageUrl": "fake_path_show2.jpg"}
    mock_post.side_effect = [mock_resp1, mock_resp2]

    generate_lightx_images("Show1", "Desc1", "Show2", "Desc2")
    first_call_args, first_call_kwargs = mock_post.call_args_list[0]
    
    # Make sure the JSON body for the first call contains the "textPrompt" we expect for Show #1.
    assert "Show1" in first_call_kwargs["json"]["textPrompt"]




