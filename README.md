
 🎶 Spotify Mood Predictor  

A data science project that predicts the **mood of songs** based on Spotify audio features using machine learning.  

---

## 📌 Project Overview  
This project uses machine learning to analyze Spotify audio features (tempo, danceability, energy, valence, etc.) and predict the mood of songs. The model categorizes songs into **four moods**:  

- 😀 **Happy**: Songs with **high valence** (> 0.6)  
- 😢 **Sad**: Songs with **low valence** (< 0.4)  
- ⚡ **Energetic**: Songs with **high energy** (> 0.7) and medium valence (0.4–0.6)  
- 😐 **Neutral**: Songs that don’t fit into the above categories  

---

## 📂 Project Structure  
```
spotify_mood_predictor/
├── app/                  # Streamlit web application
│   └── app.py            # Streamlit app for predicting song moods
├── data/                 # Data directory for storing datasets
├── models/               # Directory for storing trained models
├── src/                  # Source code
│   ├── data_loader.py        # Functions for loading data
│   ├── data_preprocessor.py  # Data preprocessing functions
│   ├── model_trainer.py      # Model training and evaluation functions
│   └── main.py               # Main script to run the full pipeline
└── README.md             # Project documentation
```

---

## ✨ Features  
- Data preprocessing pipeline (null handling, scaling).  
- Mood classification based on Spotify audio features.  
- Machine learning models: **K-Nearest Neighbors** and **Decision Tree Classifier**.  
- Model evaluation with cross-validation & metrics.  
- Hyperparameter optimization for both models.  
- Interactive **Streamlit web app** for testing predictions.  
- Visualizations: confusion matrices & feature importance.  

---

## ⚙️ Installation  

Clone the repository:  
```bash
git clone https://github.com/yourusername/spotify-mood-predictor.git
cd spotify-mood-predictor
```

Create a virtual environment (recommended):  
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

Install dependencies:  
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage  

### Running the Pipeline  
Run the full pipeline (data loading → preprocessing → model training → evaluation):  
```bash
python src/main.py
```

Options:  
- `--data_path`: Path to your Spotify dataset (CSV/Excel).  
- `--model_type`: Model to train (`knn`, `decision_tree`, or `both`).  
- `--optimize`: Enable hyperparameter optimization.  
- `--no_save`: Do not save trained models.  

If no dataset is provided, a sample dataset will be generated.  

---

### Running the Web App  
Launch the Streamlit app:  
```bash
cd app
streamlit run app.py
```

The app allows you to:  
- Adjust audio features with sliders.  
- Predict mood instantly.  
- Visualize features & their impact.  
- Switch between trained models.  

---

## 🎵 Data  
The project requires Spotify audio features:  
- `valence` (positiveness)  
- `energy`  
- `danceability`  
- `tempo` (BPM)  
- `loudness` (dB)  
- `acousticness`  
- `instrumentalness`  
- `liveness`  
- `speechiness`  

If no dataset is available, the project generates a sample one.  

---

## 📦 Dependencies  
- Python 3.7+  
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  
- streamlit  
- joblib  

---

## 🤝 Contributing  
Contributions are welcome! Please open an issue or submit a pull request.  

---

## 📜 License  
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.  

---

## 🙏 Acknowledgments  
- **Spotify** for audio features API  
- **Streamlit** for the web app framework  
- **scikit-learn team** for ML tools  
