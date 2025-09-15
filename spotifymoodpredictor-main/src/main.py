import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Get the absolute path to the project directory
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add parent directory to path to ensure imports work correctly
sys.path.append(PROJECT_DIR)

from src.data_loader import load_spotify_data, download_sample_data
from src.data_preprocessor import full_preprocessing_pipeline
from src.model_trainer import (split_data, train_knn_model, train_decision_tree_model,
                              evaluate_model, optimize_knn, optimize_decision_tree,
                              save_model, load_model)

def run_pipeline(data_path=None, use_kaggle=False, model_type='both', optimize=False, save_models=True):
    """
    Run the complete Spotify Mood Prediction pipeline
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the Spotify dataset file. If None, use sample data.
    use_kaggle : bool, optional
        Whether to download and use the Kaggle dataset
    model_type : str, optional
        Type of model to train ('knn', 'decision_tree', or 'both')
    optimize : bool, optional
        Whether to perform hyperparameter optimization
    save_models : bool, optional
        Whether to save the trained models
        
    Returns:
    --------
    dict
        Dictionary containing the trained models and evaluation results
    """
    # Step 1: Load data
    print("\n==== STEP 1: LOADING DATA ====")
    try:
        if use_kaggle:
            print("Using Kaggle dataset...")
            data = load_spotify_data(use_kaggle=True)
        elif data_path is not None and os.path.exists(data_path):
            print(f"Loading data from {data_path}...")
            data = load_spotify_data(data_path)
        else:
            print("No valid data path provided. Using sample data...")
            data_path = os.path.join(PROJECT_DIR, 'data', 'spotify_data.csv')
            if not os.path.exists(data_path):
                data_path = download_sample_data(data_path)
            data = load_spotify_data(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Falling back to sample data...")
        data_path = os.path.join(PROJECT_DIR, 'data', 'spotify_data.csv')
        if not os.path.exists(data_path):
            data_path = download_sample_data(data_path)
        data = load_spotify_data(data_path)
    
    # Step 2: Preprocess data
    print("\n==== STEP 2: PREPROCESSING DATA ====")
    X, y, scaler = full_preprocessing_pipeline(data)
    
    # Step 3: Split data
    print("\n==== STEP 3: SPLITTING DATA ====")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 4: Train models
    print("\n==== STEP 4: TRAINING MODELS ====")
    models = {}
    
    if model_type.lower() in ['knn', 'both']:
        print("\nTraining KNN model:")
        knn_model = train_knn_model(X_train, y_train)
        models['knn'] = knn_model
        
        # Optimize KNN if requested
        if optimize:
            print("\nOptimizing KNN model:")
            best_knn = optimize_knn(X_train, y_train, X_test, y_test)
            models['knn_optimized'] = best_knn
    
    if model_type.lower() in ['decision_tree', 'both']:
        print("\nTraining Decision Tree model:")
        dt_model = train_decision_tree_model(X_train, y_train)
        models['decision_tree'] = dt_model
        
        # Optimize Decision Tree if requested
        if optimize:
            print("\nOptimizing Decision Tree model:")
            best_dt = optimize_decision_tree(X_train, y_train, X_test, y_test)
            models['decision_tree_optimized'] = best_dt
    
    # Step 5: Evaluate models
    print("\n==== STEP 5: EVALUATING MODELS ====")
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name} model:")
        model_results = evaluate_model(model, X_test, y_test, 
                                     feature_names=X.columns, 
                                     save_path=os.path.join(PROJECT_DIR, 'models', name))
        results[name] = model_results
    
    # Step 6: Save models if requested
    if save_models:
        print("\n==== STEP 6: SAVING MODELS ====")
        models_dir = os.path.join(PROJECT_DIR, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        for name, model in models.items():
            save_model(model, os.path.join(models_dir, f'{name}_model.pkl'))
        
        # Also save the scaler for future use
        if scaler is not None:
            from joblib import dump
            dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
            print(f"Scaler saved to {os.path.join(models_dir, 'scaler.pkl')}")
    
    return {'models': models, 'results': results, 'scaler': scaler}

def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
    --------
    argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Spotify Mood Predictor')
    
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to the Spotify dataset file')
    parser.add_argument('--use_kaggle', action='store_true',
                        help='Download and use the Kaggle dataset')
    parser.add_argument('--model_type', type=str, default='both',
                        choices=['knn', 'decision_tree', 'both'],
                        help='Type of model to train')
    parser.add_argument('--optimize', action='store_true',
                        help='Perform hyperparameter optimization')
    parser.add_argument('--no_save', action='store_true',
                        help='Do not save the trained models')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Run the pipeline
    results = run_pipeline(
        data_path=args.data_path,
        use_kaggle=args.use_kaggle,
        model_type=args.model_type,
        optimize=args.optimize,
        save_models=not args.no_save
    )
    
    print("\nPipeline completed successfully!") 