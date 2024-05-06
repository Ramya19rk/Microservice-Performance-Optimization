# Microservice-Performance-Optimization

Problem Statement: Optimizing microservice performance is essential for maintaining system efficiency and user satisfaction in modern distributed systems. However, identifying and addressing performance bottlenecks in microservices remains a significant challenge. Our project aims to leverage trace data analysis techniques to uncover insights, identify bottlenecks, and optimize microservice performance for enhanced system reliability and scalability.

NAME : RAMYA KRISHNAN A

DOMAIN : DATA SCIENCE

Linked in URL : www.linkedin.com/in/ramyakrishnan19

# Libraries for Preprocessing

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.style as style
    import plotly.express as px
    import seaborn as sns
    from collections import Counter
    from sklearn.preprocessing import LabelEncoder
    from scipy.stats import zscore
    from datetime import date
    import streamlit as st
    from streamlit_option_menu import option_menu

# Libraries for ML Process

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import VotingClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    import pickle

# Project Overview

## Data Cleaning Process

Analyze and Handle Null Values:​

Convert Data Types as Needed:​

## Exploratory Data Analysis (EDA)

Distribution of durationNano across method names.

Univariate Analysis: durationNano distribution, serviceName frequency.

Bivariate Analysis: durationNano vs. serviceName, durationNano vs. timestamp.

Multivariate Analysis: durationNano, serviceName, and Name.

Temporal Analysis: Trend of durationNano over time.

Distribution Analysis: durationNano distribution within serviceName categories.

Service-Level and Method-Level Analysis.

Anomaly Detection and Pattern Recognition.

## Data Preprocessing

Label encoding for categorical values.

Outlier detection and log transformation.

Correlation heatmap.

## Streamlit Page

### Home menu:​

Provide an overview of the domain, technology used, and project.​

Display relevant images.​

<img width="1440" alt="Screenshot 2024-05-06 at 2 24 30 PM" src="https://github.com/Ramya19rk/Microservice-Performance-Optimization/assets/145639838/3e696f83-101f-4b83-b597-2aa4ad7d6b17">


### Prediction menu:​

Offer a form-like structure for predicting durationNano.​

<img width="1440" alt="Screenshot 2024-05-06 at 11 01 58 AM" src="https://github.com/Ramya19rk/Microservice-Performance-Optimization/assets/145639838/1a8a14b4-eb39-4af0-a54c-a7dcb5a79fbd">


### EDA Analysis menu:​

Include a select box for various EDA analyses.

<img width="1440" alt="Screenshot 2024-05-06 at 11 02 05 AM" src="https://github.com/Ramya19rk/Microservice-Performance-Optimization/assets/145639838/f8f7d576-719a-4d2a-b714-bd3426ee1d62">






    
