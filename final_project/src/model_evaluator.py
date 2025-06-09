"""
Model Evaluation Utilities for Final Project

This module contains helper functions for model evaluation,
comparison, and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.model_selection import cross_val_score
import pickle
import os

class ModelEvaluator:
    """
    A class to evaluate and compare machine learning models
    """
    
    def __init__(self):
        self.results = {}
        self.models = {}
    
    def evaluate_classification_model(self, model, X_test, y_test, model_name):
        """
        Evaluate a classification model and store results
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # ROC AUC for binary classification
        roc_auc = None
        if y_pred_proba is not None and len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        self.results[model_name] = {
            'type': 'classification',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        self.models[model_name] = model
        
        # Print results
        print(f"\n{model_name} Classification Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        if roc_auc:
            print(f"ROC AUC: {roc_auc:.4f}")
        
        return self.results[model_name]
    
    def evaluate_regression_model(self, model, X_test, y_test, model_name):
        """
        Evaluate a regression model and store results
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        self.results[model_name] = {
            'type': 'regression',
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'y_true': y_test,
            'y_pred': y_pred
        }
        
        self.models[model_name] = model
        
        # Print results
        print(f"\n{model_name} Regression Results:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        return self.results[model_name]
    
    def cross_validate_model(self, model, X, y, cv=5, scoring='accuracy'):
        """
        Perform cross-validation on a model
        """
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        print(f"\nCross-Validation Results ({scoring}):")
        print(f"Mean: {scores.mean():.4f}")
        print(f"Std: {scores.std():.4f}")
        print(f"Scores: {scores}")
        
        return scores
    
    def plot_confusion_matrix(self, model_name, figsize=(8, 6)):
        """
        Plot confusion matrix for a classification model
        """
        if model_name not in self.results:
            print(f"Model {model_name} not found in results")
            return
        
        result = self.results[model_name]
        if result['type'] != 'classification':
            print(f"Model {model_name} is not a classification model")
            return
        
        cm = confusion_matrix(result['y_true'], result['y_pred'])
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        return cm
    
    def plot_residuals(self, model_name, figsize=(10, 6)):
        """
        Plot residuals for a regression model
        """
        if model_name not in self.results:
            print(f"Model {model_name} not found in results")
            return
        
        result = self.results[model_name]
        if result['type'] != 'regression':
            print(f"Model {model_name} is not a regression model")
            return
        
        residuals = result['y_true'] - result['y_pred']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Residuals vs Predicted
        ax1.scatter(result['y_pred'], residuals, alpha=0.6)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title(f'Residuals vs Predicted - {model_name}')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title(f'Q-Q Plot - {model_name}')
        
        plt.tight_layout()
        plt.show()
    
    def compare_models(self):
        """
        Compare all evaluated models
        """
        if not self.results:
            print("No models to compare")
            return
        
        # Separate classification and regression results
        classification_results = {}
        regression_results = {}
        
        for name, result in self.results.items():
            if result['type'] == 'classification':
                classification_results[name] = {
                    'Accuracy': result['accuracy'],
                    'Precision': result['precision'],
                    'Recall': result['recall'],
                    'F1 Score': result['f1_score']
                }
                if result['roc_auc']:
                    classification_results[name]['ROC AUC'] = result['roc_auc']
            
            elif result['type'] == 'regression':
                regression_results[name] = {
                    'MSE': result['mse'],
                    'RMSE': result['rmse'],
                    'MAE': result['mae'],
                    'R² Score': result['r2_score']
                }
        
        # Create comparison DataFrames
        if classification_results:
            print("\nClassification Models Comparison:")
            clf_df = pd.DataFrame(classification_results).T
            print(clf_df.round(4))
            
            # Plot comparison
            clf_df.plot(kind='bar', figsize=(12, 6))
            plt.title('Classification Models Comparison')
            plt.ylabel('Score')
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
        
        if regression_results:
            print("\nRegression Models Comparison:")
            reg_df = pd.DataFrame(regression_results).T
            print(reg_df.round(4))
            
            # Plot comparison
            reg_df.plot(kind='bar', figsize=(12, 6))
            plt.title('Regression Models Comparison')
            plt.ylabel('Score')
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
        
        return classification_results, regression_results
    
    def get_feature_importance(self, model_name, feature_names=None):
        """
        Get feature importance from a model (if available)
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).flatten()
        else:
            print(f"Model {model_name} does not have feature importance")
            return None
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importance))]
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def plot_feature_importance(self, model_name, feature_names=None, top_n=10):
        """
        Plot feature importance
        """
        importance_df = self.get_feature_importance(model_name, feature_names)
        
        if importance_df is None:
            return
        
        # Plot top N features
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
        
        return top_features
    
    def save_model(self, model_name, filepath):
        """
        Save a model to disk
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.models[model_name], f)
        
        print(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, filepath, model_name):
        """
        Load a model from disk
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        self.models[model_name] = model
        print(f"Model loaded as {model_name}")
        
        return model

def generate_classification_report(y_true, y_pred, target_names=None):
    """
    Generate a detailed classification report
    """
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    print("\nDetailed Classification Report:")
    print(report_df)
    
    return report_df

