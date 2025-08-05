import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Set style for plots
plt.style.use('ggplot')

def main():
    print("Generating model performance metrics...")
    
    # Try different possible paths for the model file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    model_paths = [
        os.path.join(script_dir, 'model.joblib'),
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
    
    if model is None:
        raise FileNotFoundError("Could not find model.joblib in any of the expected locations")
    
    # Try different possible paths for the dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    data_paths = [
        os.path.join(script_dir, 'iris.csv'),
        os.path.join(root_dir, 'src', 'iris.csv'),
        'iris.csv'
    ]
    data = None
    
    for path in data_paths:
        try:
            print(f"Trying to load data from: {path}")
            data = pd.read_csv(path)
            print(f"Successfully loaded data from: {path}")
            break
        except FileNotFoundError:
            print(f"Data not found at: {path}")
    
    if data is None:
        raise FileNotFoundError("Could not find iris.csv in any of the expected locations")
    
    # Split the data the same way as in training
    train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
    X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_test = test.species
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    class_names = model.classes_
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Save the confusion matrix
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    plt.savefig(os.path.join(root_dir, 'metrics.png'))
    print("Saved confusion matrix to metrics.png")
    
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=class_names)
    
    # Save classification report to a file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    with open(os.path.join(root_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Append classification report to the markdown report
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)
        with open(os.path.join(root_dir, 'report.md'), 'a') as f:
            f.write("### Classification Report\n")
            f.write("```\n")
            f.write(report)
            f.write("```\n\n")
        print("Classification report appended to report.md")
    except FileNotFoundError:
        # Create the file if it doesn't exist
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)
        with open(os.path.join(root_dir, 'report.md'), 'w') as f:
            f.write("# Model Performance Report\n\n")
            f.write("### Classification Report\n")
            f.write("```\n")
            f.write(report)
            f.write("```\n\n")
        print("Created report.md with classification report")
        f.write("### Classification Report\n")
        f.write("```\n")
        f.write(report)
        f.write("```\n\n")
    
    print("Classification report saved to classification_report.txt and appended to report.md")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        importances = model.feature_importances_
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=features, y=importances)
        plt.title('Feature Importance')
        plt.tight_layout()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)
        plt.savefig(os.path.join(root_dir, 'feature_importance.png'))
        
        with open(os.path.join(root_dir, 'report.md'), 'a') as f:
            f.write("### Feature Importance\n")
            f.write("![Feature Importance](feature_importance.png)\n\n")
        
        print("Feature importance plot saved to feature_importance.png")

if __name__ == "__main__":
    main()
