# Getting Started with Spotify Mood Predictor

This guide will help you quickly get the Spotify Mood Predictor up and running on your system.

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Quick Start

The easiest way to get started is to use the provided setup script:

```bash
# Navigate to the spotify_mood_predictor directory
cd /path/to/spotify_mood_predictor

# Make the setup script executable (if needed)
chmod +x setup.sh

# Run the setup script with sample data
./setup.sh

# Or run with Kaggle dataset (if you want to use real Spotify songs)
./setup.sh --use-kaggle
```

This script will:
1. Create a virtual environment
2. Install all required dependencies in the virtual environment
3. Run the data processing pipeline
4. Train the machine learning models
5. Launch the Streamlit web app

## Manual Setup

If you prefer to run commands manually:

### 1. Set Up a Virtual Environment

```bash
# Navigate to the project directory
cd /path/to/spotify_mood_predictor

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 2. Install Requirements

```bash
# First update pip and install setuptools (important!)
pip install --upgrade pip
pip install --upgrade setuptools wheel

# Then install dependencies
pip install -r requirements.txt
```

### 3. Run the Pipeline

```bash
# Run with sample data
python run.py --run pipeline

# Or run with Kaggle dataset
python run.py --run pipeline --use_kaggle
```

### 4. Launch the Web App

```bash
python run.py --run app
```

### 5. Deactivate the Environment When Done

```bash
deactivate
```

## Troubleshooting

### Common Issues

#### 1. Python Command Not Found

If you get an error like `python3: command not found`, try using `python` instead:

```bash
python -m venv venv
```

#### 2. Virtual Environment Creation Error

If you have issues creating a virtual environment with venv, try installing and using virtualenv:

```bash
pip3 install --user virtualenv
python3 -m virtualenv venv
```

#### 3. Permission Denied

If you get a permission error when running `./setup.sh` or `./run.py`, make the scripts executable:

```bash
chmod +x setup.sh
chmod +x run.py
```

#### 4. ModuleNotFoundError

If you see an error about missing modules:

```bash
# Make sure you've activated the virtual environment
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate  # On Windows

# Make sure setuptools is installed
pip install --upgrade setuptools wheel

# Then install requirements
pip install -r requirements.txt
```

#### 5. Installation Errors (BackendUnavailable)

If you see errors like "Cannot import 'setuptools.build_meta'" when installing packages:

```bash
# Make sure setuptools is installed first
pip install --upgrade setuptools wheel

# Then try installing again
pip install -r requirements.txt
```

#### 6. Kaggle API Issues

If you have issues with the Kaggle API when using `--use_kaggle`:

```bash
# Install kagglehub separately in your virtual environment
pip install kagglehub

# You might need to authenticate with Kaggle
# Follow the instructions at: https://github.com/kaggle/kaggle-api
```

## Python 3.13 Compatibility

If you're using Python 3.13 (which is very new), some packages may not yet be fully compatible. The setup script has been updated to handle this by removing specific version constraints. If you still encounter issues, consider using Python 3.9-3.11 which have better compatibility with many data science packages.

## macOS Specific Issues

If you're using macOS with Homebrew-installed Python (which is externally managed), you must use a virtual environment as described above. Trying to install packages globally will result in an "externally-managed-environment" error.

## Contact

If you encounter any issues or have questions, please open an issue on the GitHub repository. 