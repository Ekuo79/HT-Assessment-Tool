import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
import openai
import re

# Load environment variables
load_dotenv()

# OpenAI API Key (ensure it's set in .env or environment)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the structure for outputs
class FeaturePresent(BaseModel):
    code: str = Field(description="The feature code that the training text has")
    point: int = Field(description="How many points this feature is worth (from 1-3)")
    description: str = Field(description="Reason for giving the training the points for this feature", default="")

class Outputs(BaseModel):
    outputs: List[FeaturePresent]

def chunk_text(text, max_length=2000):
    """Splits text into manageable chunks."""
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

def extract_features(content, rubric, prompt, model="gpt-4o-mini"):
    content_chunks = chunk_text(content, max_length=2000)
    all_responses = []

    for i, chunk in enumerate(content_chunks):
        try:
            messages = [
                {"role": "system", "content": "You are an AI trained to analyze human-trafficking awarness training materials using a rubric."},
                {"role": "user", "content": f"{rubric}\n\n{prompt}\n\nChunk of Content:\n{chunk}"}
            ]
            
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=512
            )

            print(response)

            if response and response.choices:
                parsed_output = response.choices[0].message.content.strip()
                parsed_output = re.sub(r"^```json\n|\n```$", "", parsed_output)
                parsed_output = json.loads(parsed_output)
                if isinstance(parsed_output, list):
                    all_responses.extend(parsed_output)
                else:
                    print(f"⚠️ Unexpected response format for chunk {i+1}: {parsed_output}")

        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")

    # Remove duplicates based on feature code
    unique_features = {feature["code"]: FeaturePresent(**feature) for feature in all_responses}.values()

    return unique_features

# Load rubric and training content
with open("rubric.txt", "r") as f:
    rubric = f.read()

# with open("FRLA_training.txt", "r") as file:
#     content = file.read()
with open("PACT_training.txt", "r") as file:
    content = file.read()

# Define prompt
prompt = """
What features are present in this chunk of training using the rubric above?
List the feature codes, their point values, and their reasons in JSON format. Example:
[
  {"code": "TAILORED_CONTENT", "point": 1, "description": "Reason..."},
  {"code": "VICTIM_FOCUSED", "point": 2, "description": "Reason..."}
]
"""

# Extract features using OpenAI GPT
features = extract_features(content, rubric, prompt)
total_points = sum(feature.point for feature in features)

# Output total

print("********************************************")
print(features)
print("***************************************************************")
print(f"Total Point Value: {total_points}")
