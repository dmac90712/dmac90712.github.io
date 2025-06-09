#!/usr/bin/env python3
"""
Anime Recommendation Model Training Script
Author: Derek Smith
Date: June 9, 2025

This script trains machine learning models to predict high-rated anime
based on anime characteristics and user ratings.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """
    Load and preprocess the anime dataset for modeling
    """
    print("Loading data...")
    # Load datasets
    anime_df = pd.read_csv('../data/anime.csv')
    rating_df = pd.read_csv('../data/rating.csv')
    
    print("Preprocessing anime data...")
    # Clean anime dataset
    anime_clean = anime_df.copy()
    anime_clean['rating'] = anime_clean['rating'].fillna(anime_clean['rating'].median())
    
    # Convert episodes to numeric, replacing 'Unknown' with NaN
    anime_clean['episodes'] = pd.to_numeric(anime_clean['episodes'], errors='coerce')
    # Fill missing episodes with median by type
    anime_clean['episodes'] = anime_clean['episodes'].fillna(
        anime_clean.groupby('type')['episodes'].transform('median')
    )
    # If still NaN after groupby (e.g., entire type has NaN), fill with overall median
    anime_clean['episodes'] = anime_clean['episodes'].fillna(anime_clean['episodes'].median())
    anime_clean['genre_count'] = anime_clean['genre'].str.count(',') + 1
    anime_clean['genre_count'] = anime_clean['genre_count'].fillna(0)
    # Only drop rows where essential columns are still missing
    anime_clean = anime_clean.dropna(subset=['rating', 'episodes', 'members'])
    
    print("Preprocessing rating data...")
    # Clean rating dataset
    rating_clean = rating_df[rating_df['rating'] != -1]
    user_counts = rating_clean['user_id'].value_counts()
    active_users = user_counts[user_counts >= 10].index
    rating_clean = rating_clean[rating_clean['user_id'].isin(active_users)]
    rating_sample = rating_clean.sample(n=min(500000, len(rating_clean)), random_state=42)
    
    print("Merging datasets...")
    # Merge data
    merged_data = rating_sample.merge(anime_clean, on='anime_id', how='inner')
    
    # Feature engineering
    modeling_data = merged_data.copy()
    le_type = LabelEncoder()
    modeling_data['type_encoded'] = le_type.fit_transform(modeling_data['type'])
    modeling_data['high_rating'] = (modeling_data['rating_y'] >= 8).astype(int)
    modeling_data['log_episodes'] = np.log1p(modeling_data['episodes'])
    modeling_data['log_members'] = np.log1p(modeling_data['members'])
    
    # Prepare features
    feature_columns = ['type_encoded', 'log_episodes', 'log_members', 'genre_count', 'rating_x']
    X = modeling_data[feature_columns].fillna(modeling_data[feature_columns].median())
    y = modeling_data['high_rating']
    
    return X, y, feature_columns, le_type

def train_and_evaluate_models(X, y, feature_columns):
    """
    Train and evaluate machine learning models
    """
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training Random Forest...")
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    print("Training Gradient Boosting...")
    # Train Gradient Boosting
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_accuracy = accuracy_score(y_test, gb_pred)
    
    # Results
    print(f"\nModel Performance:")
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    print(f"Gradient Boosting Accuracy: {gb_accuracy:.4f}")
    
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, rf_pred))
    
    print("\nGradient Boosting Classification Report:")
    print(classification_report(y_test, gb_pred))
    
    # Feature importance
    print("\nFeature Importance (Random Forest):")
    importance_rf = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance_rf)
    
    return rf_model, gb_model, scaler, y_test, rf_pred, gb_pred

def save_models(rf_model, gb_model, scaler, le_type):
    """
    Save trained models and preprocessors
    """
    print("\nSaving models...")
    joblib.dump(rf_model, '../models/random_forest_model.pkl')
    joblib.dump(gb_model, '../models/gradient_boosting_model.pkl')
    joblib.dump(scaler, '../models/scaler.pkl')
    joblib.dump(le_type, '../models/label_encoder.pkl')
    print("Models saved successfully!")

def main():
    """
    Main function to run the complete pipeline
    """
    print("=" * 50)
    print("Anime Recommendation Model Training")
    print("=" * 50)
    
    # Load and preprocess data
    X, y, feature_columns, le_type = load_and_preprocess_data()
    print(f"\nDataset prepared: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: {y.value_counts(normalize=True).round(3).to_dict()}")
    
    # Train models
    rf_model, gb_model, scaler, y_test, rf_pred, gb_pred = train_and_evaluate_models(X, y, feature_columns)
    
    # Save models
    save_models(rf_model, gb_model, scaler, le_type)
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()

