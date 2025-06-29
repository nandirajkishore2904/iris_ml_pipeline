# Iris ML Pipeline with CI/CD

This repository contains a machine learning pipeline for the Iris dataset with continuous integration and continuous deployment (CI/CD) using GitHub Actions.

## Project Structure

```
iris_ml_pipeline/
├── .github/workflows/    # GitHub Actions workflows
│   └── sanity-test.yaml  # CI workflow for model testing
├── src/                  # Source code
│   ├── iris.csv          # Iris dataset
│   ├── train.py          # Model training script
│   └── plot_metrics.py   # Script to generate performance visualizations
├── tests/                # Unit tests
│   └── test_model.py     # Model and data validation tests
└── requirements.txt      # Python dependencies
```

## Setup Instructions

### 1. Clone this Repository

```bash
git clone <repository-url>
cd iris_ml_pipeline
```

### 2. Set Up GitHub Repository with Two Branches

1. Create a new repository on GitHub
2. Push the code to the main branch:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <repository-url>
git push -u origin main
```

3. Create and switch to the dev branch:

```bash
git checkout -b dev
git push -u origin dev
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Training Pipeline Locally

```bash
cd src
python train.py
python plot_metrics.py
cd ..
```

### 5. Run Tests Locally

```bash
pytest tests/
```

## CI/CD Workflow

The CI/CD pipeline is configured in `.github/workflows/sanity-test.yaml` and includes:

1. **Code Quality Checks**:
   - Auto-formatting with Black
   - Linting with Flake8

2. **Model Training and Evaluation**:
   - Training the model on the Iris dataset
   - Generating performance metrics and visualizations

3. **Testing**:
   - Running unit tests for model validation
   - Running data validation tests

4. **Reporting**:
   - Generating a comprehensive report with CML
   - Posting results as a comment on the PR

## Making Changes and Creating Pull Requests

1. Make changes on the dev branch:

```bash
git checkout dev
# Make your changes
git add .
git commit -m "Description of changes"
git push origin dev
```

2. Create a Pull Request from dev to main on GitHub:
   - Go to your repository on GitHub
   - Click on "Pull requests" > "New pull request"
   - Select base:main and compare:dev
   - Click "Create pull request"

3. The CI/CD pipeline will automatically run on your PR, testing your changes and posting results as a comment.

4. Review the results and merge the PR if all checks pass.

## Model Performance Metrics

The CI/CD pipeline generates:
- Confusion matrix visualization
- Classification report
- Feature importance plot

These are saved as artifacts and included in the PR comment for easy review.

## Troubleshooting

If you encounter issues with the CI/CD pipeline:

1. Check the GitHub Actions logs for detailed error messages
2. Ensure all dependencies are correctly specified in requirements.txt
3. Verify that file paths in the code match the repository structure
4. Check that the GitHub repository has the necessary permissions set