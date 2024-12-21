import pytest
from ShowSuggesterAI import (
    load_embeddings,
    validate_user_input,
    calculate_user_vector,
    generate_recommendations,
)
import numpy as np
import pickle

@pytest.fixture
def sample_embeddings():
    return {
        "Game of Thrones": np.array([0.1, 0.2, 0.3]),
        "Sacred Games": np.array([0.4, 0.5, 0.6]),
        "Breaking Bad": np.array([0.7, 0.8, 0.9]),
    }

# Load Embeddings Tests
def test_load_embeddings_returns_dict(tmp_path):
    embeddings = {"Game of Thrones": [0.1, 0.2, 0.3]}
    file_path = tmp_path / "embeddings.pkl"
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)
    loaded_embeddings = load_embeddings(file_path)
    assert isinstance(loaded_embeddings, dict)

def test_load_embeddings_contains_expected_key(tmp_path):
    embeddings = {"Game of Thrones": [0.1, 0.2, 0.3]}
    file_path = tmp_path / "embeddings.pkl"
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)
    loaded_embeddings = load_embeddings(file_path)
    assert "Game of Thrones" in loaded_embeddings

# Validate User Input Tests
def test_validate_user_input_matches_correct_show():
    shows = ["Game of Thrones", "Sacred Games", "Breaking Bad"]
    corrected_shows = validate_user_input(["game of thron"], shows)
    assert corrected_shows == ["Game of Thrones"]

def test_validate_user_input_handles_multiple_matches():
    shows = ["Game of Thrones", "Sacred Games", "Breaking Bad"]
    corrected_shows = validate_user_input(["game of thron", "sacred gmes"], shows)
    assert corrected_shows == ["Game of Thrones", "Sacred Games"]

def test_validate_user_input_returns_none_for_invalid_input():
    shows = ["Game of Thrones", "Sacred Games", "Breaking Bad"]
    corrected_shows = validate_user_input(["unknown show"], shows)
    assert corrected_shows is None

# Calculate User Vector Tests
def test_calculate_user_vector_creates_correct_vector(sample_embeddings):
    selected_shows = ["Game of Thrones", "Sacred Games"]
    user_vector = calculate_user_vector(selected_shows, sample_embeddings)
    expected_vector = np.mean([sample_embeddings["Game of Thrones"], sample_embeddings["Sacred Games"]], axis=0)
    assert np.allclose(user_vector, expected_vector)

def test_calculate_user_vector_ignores_missing_shows(sample_embeddings):
    selected_shows = ["Game of Thrones", "Unknown Show"]
    user_vector = calculate_user_vector(selected_shows, sample_embeddings)
    expected_vector = sample_embeddings["Game of Thrones"]
    assert np.allclose(user_vector, expected_vector)

# Generate Recommendations Tests
def test_generate_recommendations_returns_correct_length(sample_embeddings):
    user_vector = np.array([0.2, 0.3, 0.4])
    show_vectors = list(sample_embeddings.values())
    show_titles = list(sample_embeddings.keys())
    recommendations = generate_recommendations(user_vector, show_vectors, show_titles, top_n=2)
    assert len(recommendations) == 2

def test_generate_recommendations_contains_expected_shows(sample_embeddings):
    user_vector = np.array([0.2, 0.3, 0.4])
    show_vectors = list(sample_embeddings.values())
    show_titles = list(sample_embeddings.keys())
    recommendations = generate_recommendations(user_vector, show_vectors, show_titles, top_n=2)
    recommended_titles = [rec[0] for rec in recommendations]
    assert recommended_titles[0] in sample_embeddings.keys()

def test_generate_recommendations_sorts_by_similarity(sample_embeddings):
    user_vector = np.array([0.2, 0.3, 0.4])
    show_vectors = list(sample_embeddings.values())
    show_titles = list(sample_embeddings.keys())
    recommendations = generate_recommendations(user_vector, show_vectors, show_titles, top_n=2)
    similarities = [rec[1] for rec in recommendations]
    assert similarities == sorted(similarities, reverse=True)
