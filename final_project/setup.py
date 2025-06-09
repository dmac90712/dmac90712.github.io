#!/usr/bin/env python3
"""
Setup script for Data Science Final Project

This script helps you set up your environment and get started with the project.
"""

import os
import subprocess
import sys

def create_virtual_environment():
    """
    Create a virtual environment for the project
    """
    print("Creating virtual environment...")
    try:
        subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
        print("✓ Virtual environment created successfully")
        print("To activate it, run: source venv/bin/activate")
    except subprocess.CalledProcessError:
        print("✗ Failed to create virtual environment")
        return False
    return True

def install_requirements():
    """
    Install required packages
    """
    print("\nInstalling required packages...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
        print("✓ Requirements installed successfully")
    except subprocess.CalledProcessError:
        print("✗ Failed to install requirements")
        return False
    return True

def setup_kaggle_api():
    """
    Guide user through Kaggle API setup
    """
    print("\n📊 Setting up Kaggle API...")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Click 'Create API Token' to download kaggle.json")
    print("3. Move kaggle.json to ~/.kaggle/ (create folder if it doesn't exist)")
    print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
    print("5. Test with: kaggle datasets list")
    
    response = input("\nHave you completed the Kaggle API setup? (y/n): ")
    if response.lower() == 'y':
        print("✓ Great! You're ready to download datasets from Kaggle")
        return True
    else:
        print("⚠️  You can complete this later, but you'll need it to download datasets")
        return False

def create_sample_dataset_info():
    """
    Create a sample file with dataset suggestions
    """
    sample_datasets = """
# Sample Dataset Suggestions for Your Final Project

## Popular Beginner-Friendly Datasets:

### 1. Titanic Dataset
- **URL:** https://www.kaggle.com/c/titanic
- **Type:** Classification
- **Description:** Predict survival on the Titanic
- **Good for:** Binary classification, data cleaning practice

### 2. House Prices Dataset
- **URL:** https://www.kaggle.com/c/house-prices-advanced-regression-techniques
- **Type:** Regression
- **Description:** Predict house prices based on features
- **Good for:** Feature engineering, regression techniques

### 3. Iris Dataset
- **URL:** https://www.kaggle.com/uciml/iris
- **Type:** Classification
- **Description:** Classic flower classification dataset
- **Good for:** Multi-class classification, beginner ML

### 4. Wine Quality Dataset
- **URL:** https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009
- **Type:** Classification/Regression
- **Description:** Predict wine quality based on chemical features
- **Good for:** Feature analysis, quality prediction

### 5. Customer Segmentation Dataset
- **URL:** https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python
- **Type:** Clustering
- **Description:** Mall customer segmentation
- **Good for:** Unsupervised learning, clustering

## To download a dataset:
```bash
# Example: Download Titanic dataset
kaggle competitions download -c titanic
unzip titanic.zip -d data/
```

## Dataset Selection Tips:
1. Choose something that interests you personally
2. Ensure it has 1000+ rows for meaningful analysis
3. Look for datasets with mixed data types (numerical and categorical)
4. Check for some missing values to practice data cleaning
5. Make sure there's a clear target variable for supervised learning
"""
    
    with open('data/dataset_suggestions.md', 'w') as f:
        f.write(sample_datasets)
    
    print("✓ Created dataset suggestions file: data/dataset_suggestions.md")

def main():
    """
    Main setup function
    """
    print("🚀 Data Science Final Project Setup")
    print("====================================\n")
    
    # Check if we're in the right directory
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found. Make sure you're in the project directory.")
        return
    
    # Create virtual environment
    if not os.path.exists('venv'):
        if not create_virtual_environment():
            return
    else:
        print("✓ Virtual environment already exists")
    
    # Install requirements
    install_requirements()
    
    # Setup Kaggle API
    setup_kaggle_api()
    
    # Create sample dataset info
    create_sample_dataset_info()
    
    print("\n🎉 Setup complete! Next steps:")
    print("1. Activate virtual environment: source venv/bin/activate")
    print("2. Choose a dataset from data/dataset_suggestions.md")
    print("3. Download your dataset to the data/ folder")
    print("4. Open the Jupyter notebook: jupyter notebook notebooks/data_analysis.ipynb")
    print("5. Start your analysis!")
    
    print("\n📝 Don't forget to:")
    print("- Update the report template with your information")
    print("- Track your time spent on each phase")
    print("- Commit your progress to version control")

if __name__ == '__main__':
    main()

