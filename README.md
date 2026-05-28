# Predicting Using Heterogeneous Graphs

## Overview

This project explores the use of Heterogeneous Graph Neural Networks (HGNNs) for predictive analytics using relational business data from a Bike Store dataset.

The main objective is to model relationships between customers, orders, products, and brands as a heterogeneous graph and apply Graph Neural Network techniques to predict business outcomes such as delayed orders.

The project combines:

* Graph Machine Learning
* Feature Engineering
* Heterogeneous Graph Modeling
* Predictive Analytics
* Temporal and Exploratory Data Analysis

---

## Technologies

* Python
* PyTorch
* PyTorch Geometric (PyG)
* NetworkX
* Pandas
* Scikit-Learn
* XGBoost
* SQLAlchemy
* Matplotlib / Seaborn

---

## Project Structure

```text
Predicting_using_heterogeneous_graphs/
│
├── data/          # Raw and processed datasets
├── docs/          # Documentation and reports
├── images/        # Generated charts and graph schemas
├── notebooks/     # Jupyter notebooks and experiments
├── scripts/       # Utility scripts
├── src/           # Core project source code
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Graph Structure

The heterogeneous graph is composed of multiple node and edge types:

### Node Types

* Customer
* Order
* Product
* Brand

### Relationships

* Customer → Order
* Product → Order
* Brand → Product

This structure enables the model to capture relational dependencies and business interactions across the dataset.

---

## Features Engineering

Examples of engineered features include:

* Customer state encoding
* Product price normalization
* Order month seasonality
* Temporal train/test split
* Order delay target variable

---

## Exploratory Analysis

The project also includes:

* Class imbalance analysis
* Seasonality analysis
* Confusion matrix evaluation
* Heterogeneous graph schema visualization

Generated charts are available in the `images/` folder.

---

## Model

The project uses Graph Neural Networks with PyTorch Geometric, including:

* GraphSAGE
* Heterogeneous Graph Modeling
* Message Passing Architecture

Traditional Machine Learning models such as XGBoost are also used for comparison purposes.

---

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Example:

```bash
python src/Predicting_heterogeneous_graphs.py
```

or run the notebooks inside the `notebooks/` directory.

---

## Author

Angëlo Sagnori

MBA Data Science & Analytics – USP/ESALQ
SAP Data & Analytics | Data Engineering
