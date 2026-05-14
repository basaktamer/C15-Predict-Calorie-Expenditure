---
title: Calorie Expenditure Predictor
emoji: 🔥
colorFrom: purple
colorTo: red
sdk: streamlit
sdk_version: 1.31.0
python_version: '3.10'
app_file: app.py
pinned: false
license: mit
---

# Calorie Expenditure Prediction - Kaggle Playground Series S5E5

This repository contains a machine learning application deployed to Hugging Face Spaces that predicts calories burned during physical activity. This project was developed as part of a "Become a Pro" Data Science challenge.

## Project Overview
The goal of this project is to provide high-precision caloric expenditure estimates based on physiological and exercise-related metrics. The model was trained on a dataset of 750,000 records, focusing on memory efficiency and non-linear relationship modeling.

## Technical Details
- **Model:** XGBRegressor (Extreme Gradient Boosting)
- **Performance:** 
  - **R-Squared:** 0.9962
  - **RMSE:** 3.80
- **Environment:** Python 3.10
- **Key Techniques:** 
  - Manual Categorical Mapping (Female: 1, Male: 0) to optimize memory.
  - Histogram-based tree method (`tree_method='hist'`) for handling large-scale tabular data within 8GB RAM constraints.

## Features Used
- `Sex` (Encoded)
- `Age`
- `Height`
- `Weight`
- `Duration`
- `Heart_Rate`
- `Body_Temp`

## Local Setup
To run this project locally on VS Code with Python 3.10:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate