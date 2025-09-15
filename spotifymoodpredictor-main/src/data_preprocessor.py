import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Get the absolute path to the project directory
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def drop_nulls(data):
    """
    Drop rows with null values from the dataset
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the Spotify data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with null values removed
    """
    # Check for null values
    null_counts = data.isnull().sum()
    print(f"Null values before dropping:\n{null_counts[null_counts > 0]}")
    
    # Drop rows with null values
    data_clean = data.dropna()
    
    print(f"Shape before dropping nulls: {data.shape}")
    print(f"Shape after dropping nulls: {data_clean.shape}")
    
    return data_clean

def scale_features(data, features_to_scale=None):
    """
    Scale numerical features using StandardScaler
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the Spotify data
    features_to_scale : list, optional
        List of features to scale. If None, all numerical features will be scaled.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with scaled features
    """
    # If no features specified, use all numeric columns
    if features_to_scale is None:
        # Get numerical columns
        features_to_scale = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Check if all specified features exist in the dataframe
    for feature in features_to_scale:
        if feature not in data.columns:
            raise ValueError(f"Feature '{feature}' not found in the dataset.")
    
    # Create a copy of the dataframe to avoid modifying the original
    scaled_data = data.copy()
    
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Scale the features
    scaled_data[features_to_scale] = scaler.fit_transform(data[features_to_scale])
    
    print(f"Scaled features: {', '.join(features_to_scale)}")
    
    return scaled_data, scaler

def create_mood_labels(data):
    """
    Create a 'mood' target column based on valence and energy values
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the Spotify data with 'valence' and 'energy' columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with new 'mood' column
    """
    # Check if required columns exist
    required_columns = ['valence', 'energy']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found in the dataset.")
    
    # Create a copy of the dataframe
    labeled_data = data.copy()
    
    # Create mood labels based on rules:
    # - Happy if valence > 0.6
    # - Sad if valence < 0.4
    # - Energetic if energy > 0.7 and valence between 0.4 and 0.6
    # - Neutral for everything else
    conditions = [
        (labeled_data['valence'] > 0.6),
        (labeled_data['valence'] < 0.4),
        (labeled_data['energy'] > 0.7) & (labeled_data['valence'] >= 0.4) & (labeled_data['valence'] <= 0.6)
    ]
    
    choices = ['Happy', 'Sad', 'Energetic']
    
    # Apply the conditions
    labeled_data['mood'] = np.select(conditions, choices, default='Neutral')
    
    # Display mood distribution
    mood_counts = labeled_data['mood'].value_counts()
    print(f"Mood distribution:\n{mood_counts}")
    
    return labeled_data

def prepare_features_and_target(data, features=None):
    """
    Prepare feature and target variables for model training
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the Spotify data with 'mood' column
    features : list, optional
        List of features to use. If None, uses default audio features.
        
    Returns:
    --------
    tuple
        (X, y) where X is the feature matrix and y is the target vector
    """
    # Check if 'mood' column exists
    if 'mood' not in data.columns:
        raise ValueError("'mood' column not found. Please run create_mood_labels first.")
    
    # Default audio features if none specified
    if features is None:
        features = ['danceability', 'energy', 'loudness', 'tempo', 'valence', 
                   'acousticness', 'instrumentalness', 'liveness', 'speechiness']
    
    # Check if all specified features exist in the dataframe
    for feature in features:
        if feature not in data.columns:
            raise ValueError(f"Feature '{feature}' not found in the dataset.")
    
    # Extract features and target
    X = data[features]
    y = data['mood']
    
    print(f"Features used: {', '.join(features)}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    return X, y

def clean_kaggle_dataset(data):
    """
    Clean and prepare the Kaggle Spotify dataset
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the Kaggle Spotify dataset
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned DataFrame
    """
    # Drop duplicates if needed
    if 'track_id' in data.columns:
        print(f"Shape before dropping duplicates: {data.shape}")
        data = data.drop_duplicates(subset=['track_id'])
        print(f"Shape after dropping duplicates: {data.shape}")
    
    # Rename columns if they have different names in the Kaggle dataset
    column_mapping = {
        'track_name': 'track_name',
        'track_artist': 'track_artist',
        'artist_name': 'track_artist'  # Some Kaggle datasets use artist_name instead of track_artist
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in data.columns and new_col not in data.columns:
            data = data.rename(columns={old_col: new_col})
    
    # Ensure all required features are numeric
    numeric_features = ['danceability', 'energy', 'loudness', 'tempo', 'valence', 
                       'acousticness', 'instrumentalness', 'liveness', 'speechiness']
    
    for feature in numeric_features:
        if feature in data.columns:
            data[feature] = pd.to_numeric(data[feature], errors='coerce')
    
    return data

def full_preprocessing_pipeline(data, features_to_use=None, scale=True):
    """
    Run the complete preprocessing pipeline
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the Spotify data
    features_to_use : list, optional
        List of features to use for modeling. If None, uses default audio features.
    scale : bool
        Whether to scale the features
        
    Returns:
    --------
    tuple
        (X, y, scaler) where X is the feature matrix, y is the target vector,
        and scaler is the fitted StandardScaler if scale=True, otherwise None
    """
    # Clean the dataset if it's from Kaggle
    data = clean_kaggle_dataset(data)
    
    # Drop null values
    clean_data = drop_nulls(data)
    
    # Create mood labels
    labeled_data = create_mood_labels(clean_data)
    
    # Scale features if requested
    scaler = None
    if scale:
        if features_to_use is None:
            # Default audio features
            features_to_use = ['danceability', 'energy', 'loudness', 'tempo', 'valence', 
                              'acousticness', 'instrumentalness', 'liveness', 'speechiness']
        
        scaled_data, scaler = scale_features(labeled_data, features_to_use)
    else:
        scaled_data = labeled_data
    
    # Prepare features and target
    X, y = prepare_features_and_target(scaled_data, features_to_use)
    
    return X, y, scaler

if __name__ == "__main__":
    # This code runs when the script is executed directly
    # It demonstrates how to use the functions in this module
    
    from data_loader import load_spotify_data, download_sample_data
    
    # Check if sample data exists, if not download it
    sample_data_path = os.path.join(PROJECT_DIR, 'data', 'spotify_data.csv')
    if not os.path.exists(sample_data_path):
        sample_data_path = download_sample_data(sample_data_path)
    
    # Load the sample data
    data = load_spotify_data(sample_data_path)
    
    # Run the preprocessing pipeline
    X, y, scaler = full_preprocessing_pipeline(data)
    
    # Show a sample of the processed data
    print("\nSample of processed features:")
    print(X.head())
    
    print("\nSample of target labels:")
    print(y.head()) 