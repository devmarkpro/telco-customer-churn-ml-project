# Telco Customer Churn ML Project
<a href="https://www.kaggle.com/code/devmarkpro/churn-prediction-lr-rf-svm-catboost-eda" target="_blank"><img align="left" alt="Kaggle" title="Open in Kaggle" src="https://kaggle.com/static/images/open-in-kaggle.svg"></a><br>


This repository contains a machine learning project focused on predicting customer churn for a telecommunications company using various supervised learning models.

## Dataset

The dataset, `Telco-Customer-Churn.csv`, contains customer information such as demographics, account details, and service usage for a telecom provider. The target variable is `Churn`, indicating whether a customer has left the company. The dataset is located in the `data/` folder.

- **Source:** [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Approach

The project follows a structured data science workflow:
- **Exploratory Data Analysis (EDA):**  
  Data cleaning, handling missing values, and visualizing distributions and relationships to understand key churn drivers.
- **Feature Engineering:**  
  Encoding categorical variables, scaling numerical features, and creating new features based on domain knowledge.
- **Modeling:**  
  Multiple supervised learning algorithms are compared, including Logistic Regression, Decision Trees, Random Forests, Gradient Boosting, and CatBoost.
- **Evaluation:**  
  Models are evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. Hyperparameter tuning is performed for optimal results.
- **Reporting:**  
  Key findings and model comparisons are summarized in the final report.

## Project Structure

- **customer_churn_prediction_model_comparison.ipynb**  
  The main notebook that walks through the entire churn prediction pipeline, including EDA, preprocessing, model training, evaluation, and comparison.

- **notebooks/**  
  Contains modular notebooks, each dedicated to a specific section of the main workflow (e.g., EDA, feature engineering, model training, etc.), for easier navigation and experimentation.

- **data/**  
  - `Telco-Customer-Churn.csv`: The primary dataset used for analysis and modeling.

- **results/**  
  Stores model metrics (e.g., accuracy, precision, recall, F1-score, ROC-AUC) generated after running the notebooks.

- **final-report.pdf**  
  The final project report summarizing the methodology, experiments, results, and conclusions.

- **conda-env.yml**  
  Conda environment file listing all dependencies required to run the notebooks.

## Getting Started

### 1. Prerequisites

- [Install Miniconda or Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

### 2. Clone the Repository

```bash
git clone <repo-url>
cd telco-customer-churn-ml-project
```

### 3. Set Up the Environment

Create and activate the conda environment:

```bash
conda env create -f conda-env.yml
conda activate csca-5622-supervised-learning
```

### 4. Data

The dataset is located at `data/Telco-Customer-Churn.csv`. No additional downloads are required.

### 5. Running the Notebooks

- Start JupyterLab or Jupyter Notebook:

  ```bash
  jupyter lab
  # or
  jupyter notebook
  ```

- Open and run `customer_churn_prediction_model_comparison.ipynb` for the full workflow, or explore individual sections in the `notebooks/` folder.

- Model metrics will be saved in the `results/` directory after each run.

### 6. Report

See `final-report.pdf` for a summary of the project, including key findings and model performance.

## Notes

- The main notebook is lengthy and covers the end-to-end process; for focused exploration, use the modular notebooks in the `notebooks/` folder.
- All dependencies are managed via the provided conda environment file.

## License

This project is for educational purposes.