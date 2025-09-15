**Spotify Mood Predictor**
A data science project that predicts the mood of songs based on Spotify audio features using machine learning.

Project Overview
This project uses machine learning to predict the mood of songs based on Spotify audio features such as tempo, danceability, energy, valence, and more. The model categorizes songs into four moods:

Happy: Songs with high valence (> 0.6)
Sad: Songs with low valence (< 0.4)
Energetic: Songs with high energy (> 0.7) and medium valence (0.4-0.6)
Neutral: Songs that don't fit in the above categories


**project structure **
spotify_mood_predictor/
├── app/                  # Streamlit web application
│   └── app.py            # Streamlit app for predicting song moods
├── data/                 # Data directory for storing datasets
├── models/               # Directory for storing trained models
├── src/                  # Source code
│   ├── data_loader.py    # Functions for loading data
│   ├── data_preprocessor.py # Data preprocessing functions
│   ├── model_trainer.py  # Model training and evaluation functions
│   └── main.py           # Main script to run the full pipeline
└── README.md             # Project documentation


**Features**
Data preprocessing pipeline with null value handling and feature scaling
Mood classification based on Spotify audio features
Two machine learning models: K-Nearest Neighbors and Decision Tree Classifier
Model evaluation with cross-validation and performance metrics
Hyperparameter optimization for both models
Interactive Streamlit web app for testing predictions
Visualization of confusion matrices and feature importance

**Installation**
Clone this repository:
git clone https://github.com/yourusername/spotify-mood-predictor.git
cd spotify-mood-predictor
Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install required packages:
pip install -r requirements.txt
Usage
Running the Pipeline
To run the full data pipeline (data loading, preprocessing, model training, and evaluation):

python src/main.py
Command-line options:

--data_path: Path to your own Spotify dataset file (CSV or Excel)
--model_type: Type of model to train (knn, decision_tree, or both)
--optimize: Flag to perform hyperparameter optimization
--no_save: Flag to not save the trained models
If no data path is provided, the script will generate a sample dataset.

**Running the Web App**
To launch the Streamlit web app:

cd app
streamlit run app.py
The app allows you to:

Adjust audio features using sliders
Predict the mood based on the selected features
Visualize the features and their impact on the mood
Switch between different trained models
Data
This project can work with any Spotify dataset that contains the following audio features:

valence: Musical positiveness (0.0 to 1.0)
energy: Energy level (0.0 to 1.0)
danceability: How suitable for dancing (0.0 to 1.0)
tempo: Speed in beats per minute (BPM)
loudness: Overall loudness in decibels (dB)
acousticness: Acoustic level (0.0 to 1.0)
instrumentalness: Instrumental level (0.0 to 1.0)
liveness: Presence of audience (0.0 to 1.0)
speechiness: Presence of spoken words (0.0 to 1.0)
The project includes a function to generate a sample dataset if you don't have one.

**Dependencies**
Python 3.7+
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
joblib
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

**License**
This project is licensed under the MIT License - see the LICENSE file for details.

**Acknowledgments**
Spotify for their audio features API
Streamlit for the web app framework
The scikit-learn team for the machine learning tools
