# Hypothesis Testing Web Application

This repository contains a Python-based web application for performing various hypothesis tests, built using **Streamlit**. The application is designed to facilitate hypothesis testing by providing an intuitive user interface for data input, assumption checks, and test execution. It supports a wide range of statistical tests for both numerical and categorical data.

---

## Features

- **Data Input:**
  - Upload data via CSV files.
  - Manually input data arrays.

- **Assumption Checks:**
  - Normality Check using Shapiro-Wilk Test.
  - Variance Homogeneity Check using Levene's Test.

- **Hypothesis Tests Supported:**
  - For Numerical Data:
    - Independent T-Test
    - Paired T-Test
    - Repeated Measures ANOVA
    - One-Way ANOVA
    - Non-parametric alternatives: Mann-Whitney U Test, Wilcoxon Signed-Rank Test, Friedman Test, Kruskal-Wallis Test
  - For Categorical Data:
    - Chi-Square Test
    - Fisher's Exact Test
    - McNemar Test
    - Cochran's Q Test
    - Marginal Homogeneity Test

- **Interactive Hypothesis Testing Map:**
  - Visual guide for selecting the appropriate hypothesis test based on data type and assumptions.

---

## Installation

To set up the application, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Hypothesis_Testing_WebApp.git
    cd Hypothesis_Testing_WebApp
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

1. Run the application locally:
    ```bash
    streamlit run Hypothesis_Testing_WebApp.py
    ```

2. Open the app in your web browser using the URL displayed in the terminal.

3. Follow these steps in the application:
   - Upload your dataset or enter data manually.
   - Select the data type (Numerical or Categorical).
   - Perform assumption checks for numerical data.
   - Select and execute the desired hypothesis test.
   - View detailed results and insights.

---

## Screenshots

- **Hypothesis Testing Map:**
  A visual guide embedded in the app sidebar to assist users in choosing the correct test.

---

## Dependencies

The application requires the following Python libraries:

- `streamlit`
- `pandas`
- `numpy`
- `scipy`
- `statsmodels`

Refer to the `requirements.txt` file for installation.

---

## About the Project

- **Developed by:** Serdar Hosver
- **Course:** ADS 511: Statistical Inference Methods
- **Institution:** TED University

The project aims to simplify hypothesis testing for students, researchers, and practitioners by automating the process and providing detailed statistical insights.

---

## Contribution

Contributions are welcome! If you have suggestions for improvements or additional features, feel free to create a pull request or open an issue.


