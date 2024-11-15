import os
from pydantic import BaseModel, Field
from langchain_google_vertexai import ChatVertexAI
from google.cloud import aiplatform
from typing import List

os.environ["GOOGLE_API_KEY"] = ''
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ''
aiplatform.init(project="", location="us-central1")

rubric = """
I have a rubric for assessing Anti-Human trafficking training for hotels. The features codes listed in capital letters have different associated point values. The bullet points with the "+" under the feature codes explain what the feature is and examples of what the trainings could contain to fulfill that feature (does not necessarily mean this is the only way to get the feature point and do not give points for "+" bullet points). This is the rubric below:
28 total points available

Features that get 1 point (5 total):
AUDIOVISUAL_CONTENT:
+ Audiovisual content such as videos

SURVIVOR_ACCOUNTS:
+ First-hand survivor accounts and cases

TAILORED_CONTENT:
+ Tailored content for different hotel employee positions

STATISTICS:
+ Including data like statistics and facts on trafficking

SEARCH_HOTEL_ADS:
+ Searching for ads with trafficking on hotel addresses
+ THORN, STOPLIGHT - LEO tech that lets you track sex ads, run phone numbers listed in sex ads against hotel records


Features that get 2 points (4 total):
VICTUM_FOCUSED:
+ Overall focus on victims of trafficking over attackers

RISK_REDUCTION:
+ Dont want employees to engage, call authorities

INTERACTIVITY:
+ Role-playing Human Trafficking situations
+ 5-10 question Quiz

PARTERNSHIPS:
+ Partnerships with anti-Human Trafficking organizations
+ Shelter NGOs may use hotels when housing is full
+ An NGO representative will accompany and pay for a victims stay at a hotel.
+ Could list which NGOs are participating
+ Be wary some propagate shallow advice and misinformation


Features that get 3 points (5 total):
OVERALL_DEFINTION:
+ Definitions and examples of Human Trafficking for informative reference
+ Real statistics
+ Real case studies
+ Generality for different workers
+ Show crime scene photos and real examples

HUMAN_TRAFFICKING_SIGNS:
+ Signs of Human Trafficking and what to look for (Victims are not always visibly obvious)
+ Rarely seeing a person on their own
+ Person does not answer questions
+ Dominant guest is controlling
+ Renting large blocks of rooms or floors
+ Scared/upset women hurriedly in/out
+ Any other example of Human Trafficking

LAW_ENFORCEMENT:
+ Collaboration with law enforcement agencies
+ If HT is suspect, call law enforcement (DNI)
+ Describe people involved 
+ Report Room number, Times and dates, Relevant locations, other important information to Law enforcement

LEGAL_REGULATIONS:
+ Information on what is illegal and why
+ If employee wronly suspect someone
+ What to do if suspicion complicit employee

SKILLS:
+ Skills to confirm HT through indicators and circumstance:
+ Be aware of anomalies/ attention to detail
+ Multiple guests in one room
+ Incriminating trash (condoms for example)
+ Denying cleaning services
+ General suspiciousness
"""

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
    code: str = Field(description="The feature code that the training text is missing")
    point: int = Field(description="How many points this feature is worth (from 1-3)")
    description: str = Field(description="Reason for not giving the training the points for this feature", default="")

class Outputs(BaseModel):
    outputs: List[FeaturePresent]

def chunk_text(text, max_length=2000):
    """Splits text into manageable chunks."""
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

content_chunks = chunk_text(content, max_length=2000)

# Create the model
generation_config = {
  "temperature": 0,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 512,
#   "response_mime_type": "text/plain",
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

all_responses = []
for chunk in content_chunks:
    try:
        response = structured_model.invoke(rubric + prompt + "Content:\n" + chunk)
        all_responses.extend(response.outputs)
    except Exception as e:
        print(f"Error processing chunk: {e}")


unique_features = {feature.code: feature for feature in all_responses}.values()
print("Features present in the training:")
total_points = 0
for feature in unique_features:
    try:
        total_points += feature.point
    except:
        pass
    print(f"Code: {feature.code}, Points: {feature.point}, Description: {feature.description}")

print(f'Total points: {total_points}')
