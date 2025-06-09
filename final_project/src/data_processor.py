"""
Data Processing Utilities for Final Project

This module contains helper functions for data cleaning,
preprocessing, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

class DataProcessor:
    """
    A class to handle common data processing tasks
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        self.scaler = StandardScaler()
        self.encoders = {}
    
    def basic_info(self):
        """
        Display basic information about the dataset
        """
        print(f"Dataset Shape: {self.df.shape}")
        print(f"\nColumn Info:")
        print(self.df.info())
        print(f"\nMissing Values:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_pct
        })
        print(missing_df[missing_df['Missing Count'] > 0])
        
        return missing_df
    
    def handle_missing_values(self, strategy='drop', threshold=0.5):
        """
        Handle missing values in the dataset
        
        Parameters:
        strategy: 'drop', 'mean', 'median', 'mode'
        threshold: for 'drop' strategy, drop columns with missing > threshold
        """
        if strategy == 'drop':
            # Drop columns with more than threshold missing values
            missing_pct = self.df.isnull().sum() / len(self.df)
            cols_to_drop = missing_pct[missing_pct > threshold].index
            self.df = self.df.drop(columns=cols_to_drop)
            print(f"Dropped columns: {list(cols_to_drop)}")
            
            # Drop rows with any remaining missing values
            self.df = self.df.dropna()
            
        elif strategy in ['mean', 'median']:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.df[col].isnull().any():
                    if strategy == 'mean':
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
                    else:
                        self.df[col].fillna(self.df[col].median(), inplace=True)
        
        elif strategy == 'mode':
            for col in self.df.columns:
                if self.df[col].isnull().any():
                    mode_val = self.df[col].mode()[0] if not self.df[col].mode().empty else 0
                    self.df[col].fillna(mode_val, inplace=True)
        
        print(f"Shape after handling missing values: {self.df.shape}")
        return self.df
    
    def detect_outliers(self, columns=None, method='iqr'):
        """
        Detect outliers using IQR or Z-score method
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        outliers = {}
        
        for col in columns:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers[col] = self.df[(self.df[col] < lower_bound) | 
                                       (self.df[col] > upper_bound)].index
            
            elif method == 'zscore':
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outliers[col] = self.df[z_scores > 3].index
        
        return outliers
    
    def encode_categorical(self, columns=None, method='label'):
        """
        Encode categorical variables
        
        Parameters:
        method: 'label' for LabelEncoder, 'onehot' for OneHotEncoder
        """
        if columns is None:
            columns = self.df.select_dtypes(include=['object']).columns
        
        for col in columns:
            if method == 'label':
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.encoders[col] = le
            
            elif method == 'onehot':
                dummies = pd.get_dummies(self.df[col], prefix=col)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df.drop(col, axis=1, inplace=True)
        
        return self.df
    
    def scale_features(self, columns=None, method='standard'):
        """
        Scale numerical features
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'standard':
            self.df[columns] = self.scaler.fit_transform(self.df[columns])
        
        return self.df
    
    def create_summary_stats(self):
        """
        Create summary statistics for the dataset
        """
        numeric_df = self.df.select_dtypes(include=[np.number])
        summary = numeric_df.describe()
        
        # Add additional statistics
        summary.loc['missing'] = numeric_df.isnull().sum()
        summary.loc['missing_pct'] = (numeric_df.isnull().sum() / len(numeric_df)) * 100
        
        return summary

def plot_correlation_matrix(df, figsize=(12, 8)):
    """
    Plot correlation matrix for numerical features
    """
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    return correlation_matrix

def plot_missing_values(df):
    """
    Visualize missing values in the dataset
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        print("No missing values found in the dataset!")
        return
    
    plt.figure(figsize=(10, 6))
    missing.plot(kind='bar')
    plt.title('Missing Values by Column')
    plt.xlabel('Columns')
    plt.ylabel('Number of Missing Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return missing

