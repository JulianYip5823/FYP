# Explainable Prediction of Airline Satisfaction
A hybrid machine learning and sentiment analysis framework to forecast passenger recommendations with transparent decision-making logic.

## Project Overview
- **Objective**: To develop a predictive framework that integrates textual sentiment scores with structured flight data to accurately forecast passenger satisfaction and identify critical service failure points.
- **Problem Statement**: Traditional sentiment analysis often fails to comprehend modern human language nuances, including mixed emotions and sarcasm, leading to a gap between numerical ratings and actual customer sentiment.
- **Key Contribution**: Implementation of an Explainable AI (XAI) dashboard that visualizes specific word contributions to each prediction, bridging the transparency gap in "black-box" models.

## Tech Stack
- **Language**: Python 3.11.3
- **Key Libraries**: **NLP**: NLTK (VADER), SpaCy (ABSA), Scikit-Learn (TF-IDF)
  - **Machine Learning**: Scikit-Learn, imbalanced-learn (SMOTE)
  - **Visualization**: Matplotlib, Seaborn
  - **Deployment**: Streamlit

## Data Source
- **Origin**: Combined dataset of approximately 21,000 reviews, combined from 2 primary Kaggle sources.
- **Format**: A mix of structured data (numerical ratings for seat comfort, staff, etc.) and unstructured data (textual passenger reviews)

## Methodology (CRISP-DM)
1. **Data Preprocessing**: Standardized attribute names, handled missing values, and normalized rating scales (1-5, 1-10) to a uniform 0-5 scale
2. **Exploratory Text Analysis**: Utilized Word Clouds and N-gram frequency analysis to identify recurring feedback patterns like service quality and seat comfort
3. **Feature Engineering**:
    - **Sentiment Scoring**: Generated Document-Level scores (Pos, Neg, Neu, Compound) using VADER
    - **Text Vectorization**: Converted text to numerical format using TF-IDF for trigrams
    - **Class Balancing**: Applied SMOTE to address the target class imbalance (majority "Not Recommended")
4. **Modeling & Evaluation**: Evaluated Logistic Regression, Random Forest, Linear SVC, and SGD Classifier
5. **Deployment**: Developed an interactive web app for real-time sentiment prediction and XAI visualization

## Key Results
The project evaluation was split into two phases: establishing a baseline with **Manual Tuning** and optimizing performance via **Automated Hyperparameter Tuning**

### 1. Base Models (Manual Tuning)
*Parameters were manually selected based on the CRISP-DM framework to establish initial performance benchmarks.*

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 74.34% | 45.18% | 79.43% | 0.5759 |
| SGD Classifier | 75.31% | 46.28% | 79.31% | 0.5844 |
| Random Forest | 74.90% | 43.19% | 53.64% | 0.4785 |
| Linear SVC | 76.45% | 47.93% | 76.59% | 0.5894 |

### 2. Optimized Models (Hyperparameter Tuned)
*Utilized `GridSearchCV` to optimize the trade-off between bias and variance, with a specific focus on maximizing Recall for the minority "Recommended" class.*

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Champion)** | **75.61%** | **46.62%** | **79.24%** | **0.5870** |
| SGD Classifier | 74.90% | 45.79% | 80.30% | 0.5832 |
| Random Forest | 75.45% | 44.03% | 54.37% | 0.4866 |
| Linear SVC | 56.34% | 32.51% | 92.52% | 0.4811 |

### 🔍 Key Evaluation Summary
- **Champion Model:** The **Tuned Logistic Regression** was selected for deployment. While Linear SVC achieved a higher recall (92.52%), its accuracy drop to 56.34% indicated instability and a high rate of False Positives.
- **Tuning Impact:** Hyperparameter tuning improved the Logistic Regression model's ability to correctly identify an additional **62 recommendations** compared to the base version, significantly reducing the "missed" positive feedback.

## Strengths & Limitations
1. **Pros**:
   - **Interpretability**: Unlike "black-box" models, this framework explains why a review was classified as negative
   - **Low Latency**: Optimized for real-time prediction on standard hardware
2. **Cons**:
   - **Domain Specificity**: Trained specifically on airline vocabulary; requires retraining for other industries (e.g., hotels)
   - **Sarcasm Detection**: Lexicon-based approaches (VADER) occasionally struggle with highly sarcastic textual data

## Repository Structure

The project follows a modular structure to ensure clear separation between data, processing logic, and the final application.

### Root Directory
* **`app.py`**: The main execution script for the Streamlit web application.
* **`requirements.txt`**: A comprehensive list of Python libraries and dependencies required to run the environment.
* **`README.md`**: Project documentation, setup instructions, and evaluation summary.
* **`data/`**: Directory containing both raw Kaggle datasets and processed CSV files.
* **`notebooks/`**: A series of numbered Jupyter Notebooks detailing the CRISP-DM lifecycle.
* **`models/`**: Storage for the trained Logistic Regression pickle file and the TF-IDF vectorizer.

## Acknowledgements
- **Author**: Yip Yuan Tu
- **Supervisor**: Dr. Fatin Izzati Ramli
- **Institution**: Asia Pacific University (APU)
