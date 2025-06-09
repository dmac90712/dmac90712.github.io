#!/usr/bin/env python3
"""
Anime Dataset Visualization Generator
Author: Derek Smith
Date: June 9, 2025

This script generates all visualizations for the anime recommendation analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def load_and_preprocess_data():
    """
    Load and preprocess data for visualization
    """
    print("Loading data for visualization...")
    # Load datasets
    anime_df = pd.read_csv('../data/anime.csv')
    rating_df = pd.read_csv('../data/rating.csv')
    
    # Clean anime dataset
    anime_clean = anime_df.copy()
    anime_clean['rating'] = anime_clean['rating'].fillna(anime_clean['rating'].median())
    anime_clean['episodes'] = pd.to_numeric(anime_clean['episodes'], errors='coerce')
    anime_clean['episodes'] = anime_clean['episodes'].fillna(
        anime_clean.groupby('type')['episodes'].transform('median')
    )
    anime_clean['episodes'] = anime_clean['episodes'].fillna(anime_clean['episodes'].median())
    anime_clean['genre_count'] = anime_clean['genre'].str.count(',') + 1
    anime_clean['genre_count'] = anime_clean['genre_count'].fillna(0)
    anime_clean = anime_clean.dropna(subset=['rating', 'episodes', 'members'])
    
    # Clean rating dataset
    rating_clean = rating_df[rating_df['rating'] != -1]
    rating_sample = rating_clean.sample(n=min(100000, len(rating_clean)), random_state=42)
    
    return anime_clean, rating_sample

def generate_distribution_plots(anime_clean, rating_sample):
    """
    Generate distribution plots
    """
    print("Generating distribution plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Anime rating distribution
    axes[0,0].hist(anime_clean['rating'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Distribution of Anime Ratings')
    axes[0,0].set_xlabel('Rating')
    axes[0,0].set_ylabel('Frequency')
    
    # User rating distribution
    axes[0,1].hist(rating_sample['rating'], bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0,1].set_title('Distribution of User Ratings')
    axes[0,1].set_xlabel('Rating')
    axes[0,1].set_ylabel('Frequency')
    
    # Anime type distribution
    type_counts = anime_clean['type'].value_counts()
    axes[1,0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
    axes[1,0].set_title('Distribution of Anime Types')
    
    # Episodes vs Rating scatter
    scatter_data = anime_clean[anime_clean['episodes'] <= 100]  # Filter outliers
    axes[1,1].scatter(scatter_data['episodes'], scatter_data['rating'], alpha=0.6)
    axes[1,1].set_title('Episodes vs Rating')
    axes[1,1].set_xlabel('Number of Episodes')
    axes[1,1].set_ylabel('Rating')
    
    plt.tight_layout()
    plt.savefig('../visualizations/anime_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Distribution plots saved to visualizations/anime_distributions.png")

def generate_genre_analysis(anime_clean):
    """
    Generate genre analysis visualization
    """
    print("Generating genre analysis...")
    # Extract all genres
    all_genres = []
    for genres in anime_clean['genre'].dropna():
        genre_list = [g.strip() for g in str(genres).split(',')]
        all_genres.extend(genre_list)
    
    genre_counts = pd.Series(all_genres).value_counts().head(15)
    
    plt.figure(figsize=(12, 6))
    genre_counts.plot(kind='bar', color='steelblue')
    plt.title('Top 15 Most Common Anime Genres')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../visualizations/genre_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Genre analysis saved to visualizations/genre_distribution.png")
    
    return genre_counts

def generate_model_comparison_plots():
    """
    Generate model comparison visualizations
    """
    print("Generating model comparison plots...")
    
    # Load and prepare data for modeling
    anime_df = pd.read_csv('../data/anime.csv')
    rating_df = pd.read_csv('../data/rating.csv')
    
    # Quick preprocessing
    anime_clean = anime_df.copy()
    anime_clean['rating'] = anime_clean['rating'].fillna(anime_clean['rating'].median())
    anime_clean['episodes'] = pd.to_numeric(anime_clean['episodes'], errors='coerce')
    anime_clean['episodes'] = anime_clean['episodes'].fillna(anime_clean['episodes'].median())
    anime_clean['genre_count'] = anime_clean['genre'].str.count(',') + 1
    anime_clean['genre_count'] = anime_clean['genre_count'].fillna(0)
    anime_clean = anime_clean.dropna(subset=['rating', 'episodes', 'members'])
    
    rating_clean = rating_df[rating_df['rating'] != -1]
    rating_sample = rating_clean.sample(n=min(50000, len(rating_clean)), random_state=42)
    
    # Merge and prepare features
    merged_data = rating_sample.merge(anime_clean, on='anime_id', how='inner')
    modeling_data = merged_data.copy()
    le_type = LabelEncoder()
    modeling_data['type_encoded'] = le_type.fit_transform(modeling_data['type'])
    modeling_data['high_rating'] = (modeling_data['rating_y'] >= 8).astype(int)
    modeling_data['log_episodes'] = np.log1p(modeling_data['episodes'])
    modeling_data['log_members'] = np.log1p(modeling_data['members'])
    
    feature_columns = ['type_encoded', 'log_episodes', 'log_members', 'genre_count', 'rating_x']
    X = modeling_data[feature_columns].fillna(modeling_data[feature_columns].median())
    y = modeling_data['high_rating']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train models
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)  # Reduced complexity
    gb_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
    
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    
    rf_pred = rf_model.predict(X_test)
    gb_pred = gb_model.predict(X_test)
    
    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    cm_rf = confusion_matrix(y_test, rf_pred)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Random Forest Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    cm_gb = confusion_matrix(y_test, gb_pred)
    sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title('Gradient Boosting Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('../visualizations/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    importance_rf = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    importance_gb = pd.DataFrame({
        'feature': feature_columns,
        'importance': gb_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    axes[0].barh(importance_rf['feature'], importance_rf['importance'], color='skyblue')
    axes[0].set_title('Random Forest Feature Importance')
    axes[0].set_xlabel('Importance')
    
    axes[1].barh(importance_gb['feature'], importance_gb['importance'], color='lightcoral')
    axes[1].set_title('Gradient Boosting Feature Importance')
    axes[1].set_xlabel('Importance')
    
    plt.tight_layout()
    plt.savefig('../visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Model comparison plots saved to visualizations/")

def generate_top_anime_analysis(anime_clean):
    """
    Generate analysis of top-rated and most popular anime
    """
    print("Generating top anime analysis...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Top 10 highest rated anime
    top_rated = anime_clean.nlargest(10, 'rating')
    axes[0].barh(range(len(top_rated)), top_rated['rating'], color='gold')
    axes[0].set_yticks(range(len(top_rated)))
    axes[0].set_yticklabels([name[:30] + '...' if len(name) > 30 else name for name in top_rated['name']], fontsize=8)
    axes[0].set_xlabel('Rating')
    axes[0].set_title('Top 10 Highest Rated Anime')
    axes[0].invert_yaxis()
    
    # Top 10 most popular anime by members
    most_popular = anime_clean.nlargest(10, 'members')
    axes[1].barh(range(len(most_popular)), most_popular['members']/1000, color='lightblue')
    axes[1].set_yticks(range(len(most_popular)))
    axes[1].set_yticklabels([name[:30] + '...' if len(name) > 30 else name for name in most_popular['name']], fontsize=8)
    axes[1].set_xlabel('Members (thousands)')
    axes[1].set_title('Top 10 Most Popular Anime (by members)')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('../visualizations/top_anime_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Top anime analysis saved to visualizations/top_anime_analysis.png")

def main():
    """
    Main function to generate all visualizations
    """
    print("=" * 50)
    print("Generating Anime Dataset Visualizations")
    print("=" * 50)
    
    # Load data
    anime_clean, rating_sample = load_and_preprocess_data()
    
    # Generate visualizations
    generate_distribution_plots(anime_clean, rating_sample)
    genre_counts = generate_genre_analysis(anime_clean)
    generate_top_anime_analysis(anime_clean)
    generate_model_comparison_plots()
    
    print("\n" + "=" * 50)
    print("All visualizations generated successfully!")
    print("Check the visualizations/ directory for output files.")
    print("=" * 50)

if __name__ == "__main__":
    main()

