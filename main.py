import os
from pydantic import BaseModel, Field
from langchain_google_vertexai import ChatVertexAI
from google.cloud import aiplatform
from typing import List
import json
import unittest

os.environ["GOOGLE_API_KEY"] = ''
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ''
aiplatform.init(project="", location="us-central1")

prompt = """
What features are present in this training using the rubric above?
List all feature codes and their reasons in JSON format. Example:
[
  {"code": "TAILORED_CONTENT", "point": 1, "description": "Reason..."},
  {"code": "VICTUM_FOCUSED", "point": 2, "description": "Reason..."}
]
"""

with open("example.txt", "r") as file:
    content = file.read()

class FeaturePresent(BaseModel):
    code: str = Field(description="The feature code that the training text has")
    point: int = Field(description="How many points this feature is worth (from 1-3)")
    description: str = Field(description="Reason for giving the training the points for this feature", default="")

class Outputs(BaseModel):
    outputs: List[FeaturePresent]

def chunk_text(text, max_length=2000):
    """Splits text into manageable chunks."""
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

content_chunks = chunk_text(content, max_length=2000)

# Function to extract features using the LLM model
def extract_features(content, rubric, prompt):
    generation_config = {
        "temperature": 0,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 512,
    }

    safety_settings = {
        0: 0,  # HarmCategory.HARM_CATEGORY_HATE_SPEECH
        1: 0,  # HarmCategory.HARM_CATEGORY_HARASSMENT
        2: 0,  # HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT
        3: 0,  # HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT
    }

    llm = ChatVertexAI(
        model="gemini-1.5-pro",
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    structured_model = llm.with_structured_output(Outputs)
    content_chunks = chunk_text(content, max_length=2000)

    all_responses = []
    for chunk in content_chunks:
        try:
            response = structured_model.invoke(rubric + prompt + "Content:\n" + chunk)
            all_responses.extend(response.outputs)
        except Exception as e:
            print(f"Error processing chunk: {e}")

    unique_features = {feature.code: feature for feature in all_responses}.values()
    return unique_features


# Test Cases
class TestFeatureExtraction(unittest.TestCase):

    def setUp(self):
        # Load the rubric and prompt
        with open("rubric.txt", "r") as f:
            self.rubric = f.read()

        self.prompt = """
        What features are present in this training using the rubric above?
        List all feature codes and their reasons in JSON format. Example:
        [
          {"code": "TAILORED_CONTENT", "point": 1, "description": "Reason..."},
          {"code": "VICTUM_FOCUSED", "point": 2, "description": "Reason..."}
        ]
        """

        # Path to the dataset
        self.dataset_path = "test_data"

    def test_feature_extraction(self):
        for file_name in os.listdir(self.dataset_path):
            if file_name.endswith(".json"):
                with open(os.path.join(self.dataset_path, file_name), "r") as f:
                    data = json.load(f)

                content = data["content"]
                expected_features = {feature["code"]: feature for feature in data["features"]}
                expected_total_points = data["total_points"]

                extracted_features = extract_features(content, self.rubric, self.prompt)
                extracted_features_dict = {feature.code: feature for feature in extracted_features}
                extracted_total_points = sum(feature.point for feature in extracted_features)

                # Compare each feature in a separate subTest
                for code, expected_feature in expected_features.items():
                    with self.subTest(feature_code=code):
                        self.assertIn(code, extracted_features_dict, f"Feature code {code} is missing.")
                        self.assertEqual(
                            extracted_features_dict[code].point,
                            expected_feature["points"],
                            f"Feature points for {code} do not match."
                        )

                # Check total points in a separate subTest
                print(f'total points for this training: {extracted_total_points}')
                with self.subTest(file_name=file_name):
                    self.assertEqual(
                        extracted_total_points, 
                        expected_total_points, 
                        f"Total points do not match for file {file_name}. Expected {expected_total_points}, got {extracted_total_points}."
                    )


if __name__ == "__main__":
    unittest.main()

