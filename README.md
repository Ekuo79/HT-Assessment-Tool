# Human Trafficking Training Assessment Tool

## How to Create Dataset
Create a JSON in the same format as EK1.json or EK2.json with the same keys.
Name the json file your initials and the training number to help us avoid conflicting file names.
- Put the content of the training in "content"
- Put the feature codes present in the Traning and its associated point value in "features"
- Put the total amount of points the Training earned in "total_points"

Take advantage of ChatGPT. Here's how I created EK2.json:
For example, ask ChatGPT (or any other LLM):

```Create an anti-human trafficking Training with this rubric below with all the features listed except for TAILORED_CONTENT and LEGAL_REGULATIONS: <insert rubric found in main.py>```

Then, paste the answer chatgpt gave into content , adjust the features to what is actually present, and adjust the total points.

Try to make a variety of different trainings with various point totals. For example, make trainings with around 1000 words, 2000 words and ones with around 3000-5000 words.

## Setup Instructions

### 1. Clone the Repository
Use the following command to clone the repository to your local machine:

```git clone <repository_url>```

Replace `<repository_url>` with the actual URL of the repository.

### 2. Navigate to the Project Directory
Change to the project's directory:
```cd <project_name>```

Replace `<project_name>` with the name of the repository.

### 3. Create a Virtual Environment
Create a virtual environment named `env`:
```python -m venv env```

### 4. Activate the Virtual Environment
**On Windows:**
```env\Scripts\activate```

### 5. Install Dependencies
With the virtual environment activated, install the required dependencies:
```pip install -r requirements.txt```

### 6. Google API Key
Replace empty string in main with your own API Key


