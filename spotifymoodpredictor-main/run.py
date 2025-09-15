#!/usr/bin/env python3
"""
Spotify Mood Predictor - Main Runner Script

This script provides an easy way to run different components of the project.
"""

import os
import sys
import subprocess
import argparse
import shutil

# Get the absolute path to the project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_python_command():
    """Determine the correct Python command to use"""
    # Check for python3 first (common on macOS/Linux)
    if shutil.which('python3'):
        return 'python3'
    # Then check for python (common on Windows or if python refers to Python 3)
    elif shutil.which('python'):
        return 'python'
    # If neither is found, use sys.executable as fallback
    return sys.executable

def install_requirements():
    """Install required packages"""
    python_cmd = get_python_command()
    requirements_path = os.path.join(PROJECT_DIR, 'requirements.txt')
    print(f"Installing required packages from {requirements_path}...")
    subprocess.run([python_cmd, '-m', 'pip', 'install', '-r', requirements_path])

def run_pipeline():
    """Run the data processing and model training pipeline"""
    python_cmd = get_python_command()
    script_path = os.path.join(PROJECT_DIR, 'src', 'main.py')
    print(f"Running pipeline script at: {script_path}")
    subprocess.run([python_cmd, script_path])

def run_app(use_kaggle=False):
    """Run the Streamlit web app"""
    python_cmd = get_python_command()
    app_path = os.path.join(PROJECT_DIR, 'app', 'app.py')
    print(f"Running Streamlit app at: {app_path}")
    
    # Pass the use_kaggle flag to the app
    kaggle_flag = "--use_kaggle" if use_kaggle else ""
    
    # First try direct streamlit command
    streamlit_cmd = shutil.which('streamlit')
    if streamlit_cmd:
        subprocess.run([streamlit_cmd, "run", app_path, kaggle_flag] if kaggle_flag else [streamlit_cmd, "run", app_path])
    else:
        # Fall back to python -m streamlit
        cmd = [python_cmd, "-m", "streamlit", "run", app_path]
        if kaggle_flag:
            cmd.append(kaggle_flag)
        subprocess.run(cmd)

def setup_project():
    """Setup the project by creating necessary directories"""
    # Create directories if they don't exist
    for directory in ['data', 'models']:
        dir_path = os.path.join(PROJECT_DIR, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    print("Project setup complete.")

def main():
    """Parse arguments and run the specified component"""
    parser = argparse.ArgumentParser(description="Spotify Mood Predictor Runner")
    
    parser.add_argument('--run', choices=['pipeline', 'app', 'setup', 'all', 'install'],
                       default='all', help='Component to run')
    parser.add_argument('--use_kaggle', action='store_true',
                        help='Download and use the Kaggle dataset')
    
    args = parser.parse_args()
    
    # Always run setup first
    if args.run in ['setup', 'all', 'install']:
        setup_project()
    
    # Install requirements if requested
    if args.run in ['install', 'all']:
        install_requirements()
        
    if args.run in ['pipeline', 'all']:
        print("\nRunning data pipeline and model training...")
        run_pipeline()
        
    if args.run in ['app', 'all']:
        print("\nStarting Streamlit app...")
        run_app(args.use_kaggle)

if __name__ == "__main__":
    main() 