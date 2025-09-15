import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import plot_tree

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target vector
    test_size : float, optional
        Proportion of the dataset to include in the test split
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    print(f"Training set label distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"Testing set label distribution:\n{y_test.value_counts(normalize=True)}")
    
    return X_train, X_test, y_train, y_test

def train_knn_model(X_train, y_train, n_neighbors=5, cv=5):
    """
    Train a K-Nearest Neighbors classifier
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training feature matrix
    y_train : pandas.Series
        Training target vector
    n_neighbors : int, optional
        Number of neighbors to use for KNN
    cv : int, optional
        Number of cross-validation folds
        
    Returns:
    --------
    sklearn.neighbors.KNeighborsClassifier
        Trained KNN model
    """
    # Create KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Perform cross-validation
    cv_scores = cross_val_score(knn, X_train, y_train, cv=cv)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f}")
    
    # Train the model on the entire training set
    knn.fit(X_train, y_train)
    
    print(f"KNN model trained with n_neighbors={n_neighbors}")
    
    return knn

def train_decision_tree_model(X_train, y_train, max_depth=None, cv=5):
    """
    Train a Decision Tree classifier
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training feature matrix
    y_train : pandas.Series
        Training target vector
    max_depth : int, optional
        Maximum depth of the decision tree
    cv : int, optional
        Number of cross-validation folds
        
    Returns:
    --------
    sklearn.tree.DecisionTreeClassifier
        Trained Decision Tree model
    """
    # Create Decision Tree classifier
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    
    # Perform cross-validation
    cv_scores = cross_val_score(dt, X_train, y_train, cv=cv)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f}")
    
    # Train the model on the entire training set
    dt.fit(X_train, y_train)
    
    print(f"Decision Tree model trained with max_depth={max_depth}")
    
    return dt

def optimize_knn(X_train, y_train, X_test, y_test, cv=5):
    """
    Find the optimal number of neighbors for KNN using GridSearchCV
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training feature matrix
    y_train : pandas.Series
        Training target vector
    X_test : pandas.DataFrame
        Testing feature matrix
    y_test : pandas.Series
        Testing target vector
    cv : int, optional
        Number of cross-validation folds
        
    Returns:
    --------
    sklearn.neighbors.KNeighborsClassifier
        Optimized KNN model
    """
    # Define the parameter grid
    param_grid = {'n_neighbors': range(1, 31, 2)}
    
    # Create KNN classifier
    knn = KNeighborsClassifier()
    
    # Perform grid search
    grid_search = GridSearchCV(knn, param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and model
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {best_score:.4f}")
    
    # Train the model with the best parameters
    best_knn = KNeighborsClassifier(**best_params)
    best_knn.fit(X_train, y_train)
    
    # Evaluate on the test set
    test_score = best_knn.score(X_test, y_test)
    print(f"Test set accuracy: {test_score:.4f}")
    
    return best_knn

def optimize_decision_tree(X_train, y_train, X_test, y_test, cv=5):
    """
    Find the optimal parameters for Decision Tree using GridSearchCV
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training feature matrix
    y_train : pandas.Series
        Training target vector
    X_test : pandas.DataFrame
        Testing feature matrix
    y_test : pandas.Series
        Testing target vector
    cv : int, optional
        Number of cross-validation folds
        
    Returns:
    --------
    sklearn.tree.DecisionTreeClassifier
        Optimized Decision Tree model
    """
    # Define the parameter grid
    param_grid = {
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create Decision Tree classifier
    dt = DecisionTreeClassifier(random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(dt, param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and model
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {best_score:.4f}")
    
    # Train the model with the best parameters
    best_dt = DecisionTreeClassifier(random_state=42, **best_params)
    best_dt.fit(X_train, y_train)
    
    # Evaluate on the test set
    test_score = best_dt.score(X_test, y_test)
    print(f"Test set accuracy: {test_score:.4f}")
    
    return best_dt

def evaluate_model(model, X_test, y_test, feature_names=None, save_path=None):
    """
    Evaluate a trained model and visualize results
    
    Parameters:
    -----------
    model : trained model
        Trained classifier model
    X_test : pandas.DataFrame
        Testing feature matrix
    y_test : pandas.Series
        Testing target vector
    feature_names : list, optional
        Names of features (for decision tree visualization)
    save_path : str, optional
        Path to save visualizations
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Generate classification report
    class_report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(class_report)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_test), 
                yticklabels=np.unique(y_test))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}_confusion_matrix.png")
    
    plt.close()
    
    # Decision Tree visualization
    if isinstance(model, DecisionTreeClassifier) and feature_names is not None:
        plt.figure(figsize=(20, 15))
        plot_tree(model, filled=True, feature_names=feature_names, class_names=model.classes_, rounded=True)
        plt.title("Decision Tree Visualization")
        
        if save_path:
            plt.savefig(f"{save_path}_decision_tree.png")
        
        plt.close()
        
        # Feature importance for Decision Tree
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance)
            plt.title('Feature Importance')
            
            if save_path:
                plt.savefig(f"{save_path}_feature_importance.png")
            
            plt.close()
    
    # Return evaluation metrics
    return {
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': cm,
        'predictions': y_pred
    }

def save_model(model, filepath):
    """
    Save a trained model to a file
    
    Parameters:
    -----------
    model : trained model
        Trained classifier model
    filepath : str
        Path to save the model
        
    Returns:
    --------
    str
        Path where the model was saved
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save the model
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")
    
    return filepath

def load_model(filepath):
    """
    Load a trained model from a file
    
    Parameters:
    -----------
    filepath : str
        Path to the saved model
        
    Returns:
    --------
    trained model
        Loaded classifier model
    """
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file {filepath} not found.")
    
    # Load the model
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    
    return model

if __name__ == "__main__":
    # This code runs when the script is executed directly
    # It demonstrates how to use the functions in this module
    
    from data_loader import load_spotify_data, download_sample_data
    from data_preprocessor import full_preprocessing_pipeline
    
    # Check if sample data exists, if not download it
    import os
    sample_data_path = '../data/spotify_data.csv'
    if not os.path.exists(sample_data_path):
        sample_data_path = download_sample_data(sample_data_path)
    
    # Load the sample data
    data = load_spotify_data(sample_data_path)
    
    # Preprocess the data
    X, y, scaler = full_preprocessing_pipeline(data)
    
    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train a KNN model
    knn_model = train_knn_model(X_train, y_train)
    
    # Train a Decision Tree model
    dt_model = train_decision_tree_model(X_train, y_train)
    
    # Evaluate the models
    print("\nKNN Model Evaluation:")
    knn_eval = evaluate_model(knn_model, X_test, y_test, 
                             feature_names=X.columns, 
                             save_path='../models/knn')
    
    print("\nDecision Tree Model Evaluation:")
    dt_eval = evaluate_model(dt_model, X_test, y_test, 
                            feature_names=X.columns, 
                            save_path='../models/decision_tree')
    
    # Optimize models
    print("\nOptimizing KNN:")
    best_knn = optimize_knn(X_train, y_train, X_test, y_test)
    
    print("\nOptimizing Decision Tree:")
    best_dt = optimize_decision_tree(X_train, y_train, X_test, y_test)
    
    # Save the best models
    save_model(best_knn, '../models/best_knn_model.pkl')
    save_model(best_dt, '../models/best_decision_tree_model.pkl') 