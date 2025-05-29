# Predictive Maintenance for Automotive Manufacturing

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

An end-to-end machine learning project to predict equipment failure in the automotive industry. This repository contains the code and notebooks for data preprocessing, model training, and interpretation using XGBoost, SMOTE, and SHAP.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Workflow](#workflow)
- [Results](#results)
- [License](#license)

---

## Project Overview

In manufacturing, unplanned equipment downtime is a major source of financial loss and operational inefficiency. This project tackles this problem by developing a predictive maintenance solution. By analyzing sensor data from manufacturing equipment, the goal is to build a machine learning model that can accurately predict component failures before they occur.

This allows for a strategic shift from **reactive maintenance** (fixing things after they break) to **proactive maintenance** (scheduling repairs based on failure predictions), thereby increasing uptime, reducing costs, and improving overall plant efficiency.

## Key Features

- **End-to-End ML Pipeline:** Covers all steps from data cleaning and exploration to model evaluation and interpretation.
- **Advanced Modeling:** Utilizes **XGBoost**, a high-performance gradient boosting algorithm, for classification.
- **Imbalance Handling:** Implements **SMOTE (Synthetic Minority Over-sampling Technique)** to address the highly imbalanced nature of failure data.
- **Model Interpretability:** Employs **SHAP (SHapley Additive exPlanations)** to explain model predictions, ensuring transparency and providing actionable insights into failure drivers.

## Tech Stack

- **Python 3.9+**
- **Libraries:**
  - Pandas & NumPy for data manipulation
  - Matplotlib & Seaborn for data visualization
  - Scikit-learn for preprocessing and evaluation
  - imblearn for SMOTE implementation
  - XGBoost for the classification model
  - SHAP for model interpretation
  - Jupyter for notebooks

## Installation

To set up and run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/predictive-maintenance-automotive.git](https://github.com/your-username/predictive-maintenance-automotive.git)
    cd predictive-maintenance-automotive
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can explore the project in two ways:

1.  **Run through the Jupyter Notebooks:**
    For a step-by-step walkthrough of the entire process, open and run the notebooks in the `notebooks/` directory in order.
    ```bash
    jupyter notebook
    ```

2.  **Run the Python scripts (for automation):**
    *(This is an example; modify according to your scripts)*
    ```bash
    # Run the preprocessing script
    python src/preprocess.py

    # Run the model training script
    python src/train.py
    ```

## Workflow

1.  **Data Exploration (EDA):** The `1_Data_Exploration.ipynb` notebook loads the raw data, visualizes feature distributions, and analyzes the correlation between different sensors to understand the dataset's characteristics.

2.  **Data Preprocessing:** Data is cleaned by handling missing values. Features are scaled using `StandardScaler` to prepare them for the model.

3.  **Handling Class Imbalance:** The target variable (failure/no failure) is highly imbalanced. To prevent the model from being biased towards the majority class, **SMOTE** is used on the training data to generate synthetic samples of the minority (failure) class.

4.  **Model Training:** An **XGBoost Classifier** is trained on the preprocessed, balanced data. Hyperparameters were tuned to optimize for the **F1-score**, a metric that balances precision and recall, which is crucial for this type of problem.

5.  **Model Evaluation:** The model's performance is evaluated on an unseen test set. Key metrics include the **F1-score**, **Precision**, **Recall**, and the **Area Under the Precision-Recall Curve (AUPRC)**.

6.  **Model Interpretation with SHAP:** To understand *why* the model makes certain predictions, SHAP is used. The SHAP summary plot below shows the most influential features for predicting failures.

    *(Example of how to embed an image)*
    ![SHAP Summary Plot](path/to/your/shap_summary_plot.png)

    This analysis reveals that features like `pressure`, `vibration_amplitude`, and `temperature` are the top drivers of equipment failure.

## Results

- The final XGBoost model achieved an **F1-score of 0.85** on the test set, indicating a strong balance between correctly identifying true failures (recall) and not raising too many false alarms (precision).
- The SHAP analysis successfully identified critical failure indicators, providing the maintenance team with a clear, data-driven basis for their proactive inspections.


