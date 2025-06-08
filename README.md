# Bike Sharing Demand Prediction with AutoGluon

This project demonstrates the use of AutoGluon for predicting bike sharing demand using machine learning. The project is part of the AWS Machine Learning Scholarship Program.

## Project Overview

The goal of this project is to predict the number of bikes that will be rented at each hour of the day, given various features such as weather conditions, time of day, and seasonal information. This prediction model can help bike sharing companies optimize their bike distribution and maintenance schedules.

## Features

- Data preprocessing and feature engineering
- Model training using AutoGluon
- Hyperparameter optimization
- Model evaluation and comparison
- Submission generation for Kaggle competition

## Project Structure

```
project/
├── scholarship-project.ipynb    # Main Jupyter notebook
├── train.csv                    # Training dataset
├── test.csv                     # Test dataset
├── sampleSubmission.csv         # Sample submission format
├── submission.csv               # Generated predictions
├── img/                         # Project images and visualizations
└── kaggle_config/              # Kaggle API configuration
```

## Setup Instructions

1. Clone this repository:
```bash
git clone https://github.com/brlikhon/Predict-Bike-Sharing-Demand-with-AutoGluon.git
```

2. Install required dependencies:
```bash
pip install autogluon
pip install kaggle
```

3. Configure Kaggle API:
   - Place your `kaggle.json` file in the `kaggle_config` directory
   - Ensure the file has the correct permissions

4. Run the Jupyter notebook:
```bash
jupyter notebook project/scholarship-project.ipynb
```

## Results

The project includes model evaluation metrics and visualizations showing the performance of different models. Check the `img` directory for detailed performance graphs.

## License

This project is part of the AWS Machine Learning Scholarship Program.

## Author

- [brlikhon](https://github.com/brlikhon)
