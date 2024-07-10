# Neural Network Analysis

This project is part of an assignment to optimize the performance of a neural network by trying various combinations of hyperparameters and evaluating their results.

## Dataset

The dataset used for this project is the Spambase dataset from the UCI ML repository. [Link to the dataset](https://archive.ics.uci.edu/ml/datasets/Spambase)

## Project Structure

- `NeuralNet.py`: Main script containing the implementation of the neural network.
- `model_history_plot.png`: Plot showing the model loss vs. epochs for various hyperparameters.
- `results.csv`: CSV file containing the results of different hyperparameter combinations.

## Dependencies

- Python 3.6+
- numpy
- pandas
- scikit-learn
- matplotlib

## Installation

To install the required dependencies, run:

pip install -r requirements.txt

## Running the Code

To run the code, follow these steps:

1. Ensure you have the necessary dependencies installed:

   pip install -r requirements.txt

2. Run the `NeuralNet.py` script:

   python NeuralNet.py

This will preprocess the data, train the neural network with various hyperparameters, and output the results in 
`results.csv`.The script will also generate a plot of the model loss vs. epochs and save it as `model_history_plot.png`.
