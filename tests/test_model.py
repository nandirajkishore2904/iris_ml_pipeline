import joblib
import pandas as pd
import numpy as np
import os
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Constants
EXPECTED_COLUMNS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
EXPECTED_CLASSES = ['setosa', 'versicolor', 'virginica']

def test_model_exists():
    """Test if model file exists and loads correctly"""
    # Try different possible paths for the model file
    # Get the absolute path to the project root directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(test_dir)
    src_dir = os.path.join(root_dir, 'src')
    
    model_paths = [
        os.path.join(src_dir, 'model.joblib'),
        os.path.join(root_dir, 'model.joblib'),
        'model.joblib'
    ]
    model = None
    
    for path in model_paths:
        try:
            print(f"Trying to load model from: {path}")
            model = joblib.load(path)
            print(f"Successfully loaded model from: {path}")
            break
        except FileNotFoundError:
            print(f"Model not found at: {path}")
    
    assert model is not None, "Could not find model.joblib in any of the expected locations"
    assert isinstance(model, DecisionTreeClassifier)
    
    # Check if metadata file exists in any of the expected locations
    # Get the absolute path to the project root directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(test_dir)
    src_dir = os.path.join(root_dir, 'src')
    
    metadata_paths = [
        os.path.join(src_dir, 'model_metadata.json'),
        os.path.join(root_dir, 'model_metadata.json'),
        'model_metadata.json'
    ]
    metadata_exists = False
    
    for path in metadata_paths:
        if os.path.exists(path):
            metadata_exists = True
            print(f"Found metadata file at: {path}")
            break
    
    assert metadata_exists, "Model metadata file not found in any of the expected locations"

def test_model_accuracy():
    """Test if model has reasonable accuracy"""
    # Try different possible paths for the model file
    # Get the absolute path to the project root directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(test_dir)
    src_dir = os.path.join(root_dir, 'src')
    
    model_paths = [
        os.path.join(src_dir, 'model.joblib'),
        os.path.join(root_dir, 'model.joblib'),
        'model.joblib'
    ]
    model = None
    
    for path in model_paths:
        try:
            model = joblib.load(path)
            break
        except FileNotFoundError:
            continue
    
    assert model is not None, "Could not find model.joblib in any of the expected locations"
    
    # Try different possible paths for the dataset
    # Get the absolute path to the project root directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(test_dir)
    src_dir = os.path.join(root_dir, 'src')
    
    data_paths = [
        os.path.join(src_dir, 'iris.csv'),
        os.path.join(root_dir, 'iris.csv'),
        'iris.csv'
    ]
    data = None
    
    for path in data_paths:
        try:
            data = pd.read_csv(path)
            break
        except FileNotFoundError:
            continue
    
    assert data is not None, "Could not find iris.csv in any of the expected locations"
    train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
    X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
    y_test = test.species
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Check accuracy
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy >= 0.7, f"Model accuracy {accuracy} is below threshold"

def test_model_structure():
    """Test if model has expected structure"""
    # Try different possible paths for the model file
    # Get the absolute path to the project root directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(test_dir)
    src_dir = os.path.join(root_dir, 'src')
    
    model_paths = [
        os.path.join(src_dir, 'model.joblib'),
        os.path.join(root_dir, 'model.joblib'),
        'model.joblib'
    ]
    model = None
    
    for path in model_paths:
        try:
            model = joblib.load(path)
            break
        except FileNotFoundError:
            continue
    
    assert model is not None, "Could not find model.joblib in any of the expected locations"
    
    # Check max depth is 3 as specified in training
    assert model.max_depth == 3, f"Expected max_depth=3, got {model.max_depth}"
    
    # Check if model can handle 4 input features
    assert model.n_features_in_ == 4, f"Expected 4 features, got {model.n_features_in_}"

def test_data_exists():
    """Test if the dataset file exists"""
    # Get the absolute path to the project root directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(test_dir)
    src_dir = os.path.join(root_dir, 'src')
    
    data_path = os.path.join(src_dir, 'iris.csv')
    assert os.path.exists(data_path), f"Dataset file not found at {data_path}"
    
def test_data_format():
    """Test if the dataset has the expected format"""
    # Get the absolute path to the project root directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(test_dir)
    src_dir = os.path.join(root_dir, 'src')
    
    data_path = os.path.join(src_dir, 'iris.csv')
    data = pd.read_csv(data_path)
    
    # Check columns
    assert set(data.columns) == set(EXPECTED_COLUMNS), f"Expected columns {EXPECTED_COLUMNS}, got {data.columns.tolist()}"
    
    # Check data types
    numeric_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    for col in numeric_columns:
        assert pd.api.types.is_numeric_dtype(data[col]), f"Column {col} should be numeric"
    
    # Check species column contains expected classes
    assert set(data['species'].unique()).issubset(set(EXPECTED_CLASSES)), \
        f"Species column should only contain values from {EXPECTED_CLASSES}"

def test_data_quality():
    """Test data quality (no missing values, valid ranges)"""
    # Get the absolute path to the project root directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(test_dir)
    src_dir = os.path.join(root_dir, 'src')
    
    data_path = os.path.join(src_dir, 'iris.csv')
    data = pd.read_csv(data_path)
    
    # Check for missing values
    assert data.isnull().sum().sum() == 0, "Dataset contains missing values"
    
    # Check for valid ranges
    assert data['sepal_length'].between(4.0, 8.0).all(), "Sepal length out of expected range (4.0-8.0)"
    assert data['sepal_width'].between(2.0, 4.5).all(), "Sepal width out of expected range (2.0-4.5)"
    assert data['petal_length'].between(1.0, 7.0).all(), "Petal length out of expected range (1.0-7.0)"
    assert data['petal_width'].between(0.1, 2.5).all(), "Petal width out of expected range (0.1-2.5)"

def test_model_predictions_shape():
    """Test if model predictions have the expected shape"""
    # Try different possible paths for the model file
    # Get the absolute path to the project root directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(test_dir)
    src_dir = os.path.join(root_dir, 'src')
    
    model_paths = [
        os.path.join(src_dir, 'model.joblib'),
        os.path.join(root_dir, 'model.joblib'),
        'model.joblib'
    ]
    model = None
    
    for path in model_paths:
        try:
            model = joblib.load(path)
            break
        except FileNotFoundError:
            continue
    
    assert model is not None, "Could not find model.joblib in any of the expected locations"
    
    # Create a sample input with the correct shape
    sample_input = pd.DataFrame({
        'sepal_length': [5.1, 6.3, 7.2],
        'sepal_width': [3.5, 2.5, 3.0],
        'petal_length': [1.4, 4.9, 5.8],
        'petal_width': [0.2, 1.5, 1.8]
    })
    
    # Get predictions
    predictions = model.predict(sample_input)
    
    # Check shape
    assert len(predictions) == len(sample_input), "Predictions length doesn't match input length"
    
    # Check that predictions are valid classes
    assert set(predictions).issubset(set(EXPECTED_CLASSES)), \
        f"Predictions should only contain values from {EXPECTED_CLASSES}"
