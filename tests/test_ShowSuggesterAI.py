# tests/test_ShowsuggesterAI.py

import pytest
import numpy as np
import pickle
import json
from unittest.mock import patch, MagicMock
from src.ShowSuggesterAI import (
    load_embeddings,
    validate_user_input,
    calculate_user_vector,
    generate_recommendations,
    create_fictional_show_name_and_description,
    generate_lightx_image
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
@patch("src.ShowSuggesterAI.openai.ChatCompletion.create")
def test_create_fictional_show_name_and_description_includes_basis(mock_openai):
    """
    Test that the returned description includes the basis text (if provided).
    """
    # Mock a successful JSON response from OpenAI
    basis_text = "Breaking Bad, Game of Thrones"
    fake_llm_response = {
        "id": "some_id",
        "object": "chat.completion",
        "created": 1234567890,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "title": "A Breaking Bad Universe",
                        "description": f"An intriguing show heavily inspired by {basis_text}."
                    })
                },
                "finish_reason": "stop",
                "index": 0
            }
        ]
    }
    mock_openai.return_value = MagicMock(**fake_llm_response)

    # Call the function
    _, show_description = create_fictional_show_name_and_description(
        basis=basis_text,
    )
    assert basis_text in show_description


# -----------------------------------------------------------------------------
# generate_lightx_images Tests
# -----------------------------------------------------------------------------
@patch("src.ShowSuggesterAI.requests.post")
def test_generate_lightx_images_show_path(mock_post):
    """
    Ensures generate_lightx_images returns the correct path (URL)
    under a successful response from the LightX text2image/order-status flow.
    """
    # Mock the first call => text2image
    mock_resp_text2image = MagicMock(status_code=200)
    # Suppose it returns {"orderId": "order_for_show"}
    mock_resp_text2image.json.return_value = {
        "statusCode": 2000,
        "message": "SUCCESS",
        "body": {
            "orderId": "7906da5353b504162db5199d6",
            "maxRetriesAllowed": 5,
            "avgResponseTimeInSec": 15,
            "status": "init"
        }
    }


    # Next calls => order-status
    mock_resp_orderstatus = MagicMock(status_code=200)
    mock_resp_orderstatus.json.return_value = {
        "statusCode": 2000,
        "message": "SUCCESS",
        "body": {
            "orderId": "7906da5353b504162db5199d6",
            "status": "active",
            "output": "fake_path_show.jpg"
        }
    }

    # The function calls requests.post() to text2image and to order-status.
    # We set side_effect to a list of these mock responses
    mock_post.side_effect = [
        mock_resp_text2image,  # text2image
        mock_resp_orderstatus, # order-status
    ]

    show_ad_path = generate_lightx_image("show_name", "show_desc")
    assert show_ad_path == "fake_path_show.jpg"


@patch("src.ShowSuggesterAI.requests.post")
def test_generate_lightx_images_calls_lightx_twice(mock_post):
    """
    Ensures generate_lightx_images calls requests.post enough:
    (1) text2image -> (2) order-status
    """
    # Create two identical responses
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {
        "orderId": "some_orderId",
        "statusCode": 2000,
        "message": "SUCCESS",
        "body": {
            "orderId": "some_orderId",
            "status": "active",
            "output": "fake_path_showN.jpg"
        }
    }
    mock_post.side_effect = [mock_resp, mock_resp]

    generate_lightx_image("show_name", "show_desc")
    
    assert mock_post.call_count == 2





