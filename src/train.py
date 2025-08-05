#!/usr/bin/env python3
"""
Iris Dataset Model Training Script

This script trains a Decision Tree Classifier on the Iris dataset,
evaluates its performance, and saves the trained model.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Feature columns
FEATURE_COLUMNS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
TARGET_COLUMN = 'species'

def load_data(filepath):
    """
    Load the dataset from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: The loaded dataset
    """
    try:
        logger.info(f"Loading data from {filepath}")
        data = pd.read_csv(filepath)
        logger.info(f"Data loaded successfully with shape {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

def preprocess_data(data):
    """
    Preprocess the dataset and split into train/test sets.
    
    Args:
        data (pandas.DataFrame): The input dataset
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    logger.info("Preprocessing data and splitting into train/test sets")
    
    # Split data into train and test sets
    train, test = train_test_split(
        data,
        test_size=0.4,
        stratify=data[TARGET_COLUMN],
        random_state=42
    )
    
    # Extract features and target
    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]
    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]
    
    logger.info(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train a Decision Tree Classifier.
    
    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        
    Returns:
        sklearn.tree.DecisionTreeClassifier: The trained model
    """
    logger.info("Training Decision Tree Classifier")
    
    model = DecisionTreeClassifier(max_depth=3, random_state=1)
    model.fit(X_train, y_train)
    
    logger.info("Model training completed")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    
    Args:
        model: The trained model
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): Test target
        
    Returns:
        float: Accuracy score
    """
    logger.info("Evaluating model performance")
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Model accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))
    
    return accuracy

def save_model(model, filepath):
    """
    Save the trained model to disk.
    
    Args:
        model: The trained model
        filepath (str): Path to save the model
    """
    try:
        logger.info(f"Saving model to {filepath}")
        joblib.dump(model, filepath)
        logger.info("Model saved successfully")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        sys.exit(1)

def main():
    """Main function to orchestrate the training process."""
    # Ensure we're in the right directory
    
    # Load and process data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'iris.csv')
    data = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Save model in multiple locations to ensure it's accessible
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_model(model, os.path.join(script_dir, "model.joblib"))  # Save in src directory
    save_model(model, os.path.join(os.path.dirname(script_dir), "model.joblib"))  # Save in root directory
    
    # Save model metadata
    metadata = {
        'accuracy': accuracy,
        'features': FEATURE_COLUMNS,
        'classes': model.classes_.tolist()
    }
    
    # Save metadata as JSON in multiple locations
    import json
    
    # Save metadata in multiple locations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(os.path.dirname(script_dir), 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Training pipeline completed successfully")

if __name__ == "__main__":
    main()
