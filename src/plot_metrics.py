import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Set style for plots
plt.style.use('ggplot')

def main():
    print("Generating model performance metrics...")
    
    # Load the trained model
    model = joblib.load('../model.joblib')
    
    # Load the dataset
    data = pd.read_csv('iris.csv')
    
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
    plt.savefig('../metrics.png')
    print("Saved confusion matrix to metrics.png")
    
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=class_names)
    
    # Save classification report to a file
    with open('../classification_report.txt', 'w') as f:
        f.write(report)
    
    # Append classification report to the markdown report
    with open('../report.md', 'a') as f:
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
        plt.savefig('../feature_importance.png')
        
        with open('../report.md', 'a') as f:
            f.write("### Feature Importance\n")
            f.write("![Feature Importance](feature_importance.png)\n\n")
        
        print("Feature importance plot saved to feature_importance.png")

if __name__ == "__main__":
    main()