import pandas as pd
import os
import kagglehub

# Get the absolute path to the project directory
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_spotify_data(file_path=None, use_kaggle=False):
    """
    Load Spotify data from a specified file path or download it from Kaggle
    
    Parameters:
    -----------
    file_path : str or None
        The file path to load the data from. If None, look in the default directory
    use_kaggle : bool
        Whether to try downloading the dataset from Kaggle
        
    Returns:
    --------
    pandas.DataFrame or None
        DataFrame containing the Spotify data, or None if not available
    """
    print(f"DEBUG: load_spotify_data called with file_path={file_path}, use_kaggle={use_kaggle}")
    
    try:
        # If Kaggle download is requested, try to download the dataset
        if use_kaggle:
            print("DEBUG: Attempting to download dataset from Kaggle...")
            try:
                # Direct Kaggle import
                import kagglehub
                print("DEBUG: Successfully imported kagglehub")
                
                # Try to download sample from Kaggle
                from kagglehub import KaggleDatasetAdapter
                print("DEBUG: Creating KaggleDatasetAdapter")
                
                try:
                    print("DEBUG: Loading dataset from Kaggle...")
                    kaggle_df = kagglehub.load_dataset(
                        KaggleDatasetAdapter.PANDAS,
                        "joebeachcapital/30000-spotify-songs",
                        ""  # Empty string loads the first file
                    )
                    print(f"DEBUG: Successfully loaded Kaggle dataset with shape: {kaggle_df.shape}")
                    
                    # Sample a reasonable number of rows for performance
                    sample_size = 5000  # Increased from 1000 to 5000
                    kaggle_sample = kaggle_df.sample(min(len(kaggle_df), sample_size), random_state=42)
                    
                    # Save the sample for future use
                    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    sample_path = os.path.join(PROJECT_DIR, 'data', 'spotify_sample.csv')
                    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
                    kaggle_sample.to_csv(sample_path, index=False)
                    print(f"DEBUG: Saved Kaggle sample to {sample_path}")
                    
                    return kaggle_sample
                except Exception as kaggle_error:
                    print(f"DEBUG: Error downloading Kaggle dataset: {type(kaggle_error).__name__}: {kaggle_error}")
                    import traceback
                    print(traceback.format_exc())
                    raise kaggle_error
                
            except Exception as e:
                print(f"DEBUG: Error during Kaggle download: {type(e).__name__}: {e}")
                
        # If we're here, either Kaggle was not requested or it failed
        # Try to load the dataset from file
        if file_path is None:
            # Try to find a dataset file in the default directory
            PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            DATA_DIR = os.path.join(PROJECT_DIR, 'data')
            
            # Check if directory exists
            if not os.path.exists(DATA_DIR):
                os.makedirs(DATA_DIR, exist_ok=True)
                print(f"DEBUG: Created data directory at {DATA_DIR}")
            
            # Look for CSV files
            dataset_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
            print(f"DEBUG: Found dataset files: {dataset_files}")
            
            if not dataset_files:
                # No dataset files found, create a sample
                print("DEBUG: No dataset files found, creating sample...")
                return download_sample_data(os.path.join(DATA_DIR, 'spotify_sample.csv'))
            
            # Prefer the Kaggle sample if it exists and use_kaggle is True
            if use_kaggle and 'spotify_sample.csv' in dataset_files:
                file_path = os.path.join(DATA_DIR, 'spotify_sample.csv')
                print(f"DEBUG: Using Kaggle sample file: {file_path}")
            else:
                # Otherwise, use the first available dataset
                file_path = os.path.join(DATA_DIR, dataset_files[0])
                print(f"DEBUG: Using first available dataset: {file_path}")
        
        print(f"DEBUG: Loading dataset from file: {file_path}")
        
        # Determine file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Load based on file extension
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
            print(f"DEBUG: Loaded CSV file with shape: {df.shape}")
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            print(f"DEBUG: Loaded Excel file with shape: {df.shape}")
        elif file_ext == '.json':
            df = pd.read_json(file_path)
            print(f"DEBUG: Loaded JSON file with shape: {df.shape}")
        else:
            print(f"DEBUG: Unsupported file format: {file_ext}")
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        return df
    
    except Exception as e:
        print(f"DEBUG: Error in load_spotify_data: {type(e).__name__}: {e}")
        import traceback
        print(traceback.format_exc())
        # Create a sample dataset if loading fails
        if file_path:
            print(f"DEBUG: Creating sample dataset at {file_path}")
            return download_sample_data(file_path)
        else:
            print("DEBUG: Creating sample dataset at default location")
            PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            DATA_DIR = os.path.join(PROJECT_DIR, 'data')
            os.makedirs(DATA_DIR, exist_ok=True)
            return download_sample_data(os.path.join(DATA_DIR, 'spotify_sample.csv'))

def download_sample_data(output_path=None):
    """
    Download a sample Spotify dataset if the user doesn't have one
    
    Parameters:
    -----------
    output_path : str, optional
        Path where the downloaded dataset will be saved
        
    Returns:
    --------
    str
        Path to the downloaded dataset
    """
    # Set default output path if none provided
    if output_path is None:
        output_path = os.path.join(PROJECT_DIR, 'data', 'spotify_data.csv')
    
    # Using a publicly available Spotify dataset from Kaggle
    # This is a sample function - in a real scenario, you might use Kaggle API
    # or requests to download a dataset
    
    # For this example, we'll create a small sample dataset
    import numpy as np
    
    # Create a sample dataset with the required features
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'track_id': [f'track_{i}' for i in range(n_samples)],
        'track_name': [f'Song {i}' for i in range(n_samples)],
        'track_artist': [f'Artist {i%50}' for i in range(n_samples)],
        'tempo': np.random.uniform(60, 200, n_samples),
        'danceability': np.random.uniform(0, 1, n_samples),
        'energy': np.random.uniform(0, 1, n_samples),
        'loudness': np.random.uniform(-20, 0, n_samples),
        'valence': np.random.uniform(0, 1, n_samples),
        'acousticness': np.random.uniform(0, 1, n_samples),
        'instrumentalness': np.random.uniform(0, 1, n_samples),
        'liveness': np.random.uniform(0, 1, n_samples),
        'speechiness': np.random.uniform(0, 1, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save the sample dataset
    df.to_csv(output_path, index=False)
    print(f"Sample dataset created and saved to {output_path}")
    
    return output_path

if __name__ == "__main__":
    # This code runs when the script is executed directly
    # It demonstrates how to use the functions in this module
    
    # Download and use the Kaggle dataset
    try:
        data = load_spotify_data(use_kaggle=True)
        print("\nFirst 5 rows of the Kaggle dataset:")
        print(data.head())
    except (ImportError, Exception) as e:
        print(f"Error loading Kaggle dataset: {e}")
        print("Falling back to sample data...")
        # Check if sample data exists, if not download it
        sample_data_path = os.path.join(PROJECT_DIR, 'data', 'spotify_data.csv')
        if not os.path.exists(sample_data_path):
            sample_data_path = download_sample_data(sample_data_path)
        
        # Load the sample data
        data = load_spotify_data(sample_data_path)
        
        # Display the first few rows
        print("\nFirst 5 rows of the sample dataset:")
        print(data.head()) 