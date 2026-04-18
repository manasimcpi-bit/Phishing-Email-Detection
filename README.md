# Phishing Email Detection and Cybersecurity Attack Detection System

## Project Overview
This project focuses on detecting phishing emails and suspicious cyber attack patterns using machine learning and natural language processing techniques. The system compares multiple models, including Logistic Regression, Random Forest, Support Vector Machine, and BERT, and evaluates their performance using standard classification metrics.

## Project Objectives
- Detect phishing or malicious patterns from uploaded data
- Compare the performance of multiple machine learning models
- Test model generalization using another dataset
- Build a graphical user interface for practical prediction
- Present results visually through charts, confusion matrices, and tables

## Models Used
- Logistic Regression
- Random Forest
- Support Vector Machine
- BERT (advanced experimental model)

## Tools and Technologies
- Python
- VS Code on Windows
- Flask
- pandas
- numpy
- matplotlib
- scikit-learn
- joblib
- transformers
- torch

## Week 1 Progress
- Defined the cybersecurity problem
- Identified and compared datasets
- Selected the final dataset
- Set up the development environment
- Initialized Git and project structure

## Week 2 Progress
- Loaded and explored the selected dataset
- Examined columns, data types, and missing values
- Performed initial exploratory data analysis

## Week 3 Progress
- Cleaned text data
- Standardized labels
- Removed missing and duplicate values
- Prepared data for training

## Week 4 Progress
- Applied TF-IDF feature extraction
- Trained Logistic Regression, Random Forest, and SVM
- Evaluated baseline model performance

## Week 5 Progress
- Compared Logistic Regression, Random Forest, SVM, and BERT
- Evaluated models using accuracy, precision, recall, and F1-score
- Tested trained models on another phishing-related dataset
- Created comparison tables and charts
- Updated project results and documentation

## Week 6 Progress
- Started building the graphical user interface using Flask
- Integrated trained models into the application
- Added CSV upload and prediction workflow
- Generated visual outputs such as charts and confusion matrices
- Displayed detailed prediction results in table format
- Continued BERT integration as an advanced model component
- Applied trained models to another dataset as required by supervisor feedback

## Current Application Features
- Upload CSV file for prediction
- View model status and performance summary
- Detect attacks using Logistic Regression, Random Forest, and SVM
- Display visual analytics including bar chart, probability distribution, and risk distribution
- Show confusion matrices when labels are available
- Preview detailed prediction results for the first 50 records
