import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings

# Suppress convergence warnings for clean output
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class NeuralNet:
    def __init__(self, dataFile):
        self.y = None
        self.X = None
        self.raw_input = pd.read_csv(dataFile)

    def preprocess(self):
        # Handle missing values
        self.raw_input.fillna(self.raw_input.mean(), inplace=True)

        # Split features and target
        self.X = self.raw_input.iloc[:, :-1]
        self.y = self.raw_input.iloc[:, -1]

        # Standardize the dataset (mean = 0 and variance = 1) excluding the target variable
        scaler = StandardScaler()
        self.X = pd.DataFrame(scaler.fit_transform(self.X), columns=self.X.columns)

        return 0

    def train_evaluate(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Hyperparameters
        activations = ['logistic', 'tanh', 'relu']
        learning_rates = [0.001, 0.01]
        max_iterations = [1000, 2000]  # Increased iterations
        num_hidden_layers = [1, 2]
        num_neurons = [10, 20]

        # Results storage
        results = []

        # Create different models for each combination of hyperparameters
        for activation in activations:
            for lr in learning_rates:
                for epochs in max_iterations:
                    for layers in num_hidden_layers:
                        for neurons in num_neurons:
                            hidden_layer_sizes = tuple([neurons] * layers)
                            model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                                  activation=activation,
                                                  learning_rate_init=lr,
                                                  max_iter=epochs,
                                                  random_state=42,
                                                  early_stopping=True,  # Early stopping to avoid overfitting
                                                  n_iter_no_change=10)  # Stop if no improvement in 10 epochs

                            model.fit(X_train, y_train)
                            train_acc = model.score(X_train, y_train)
                            test_acc = model.score(X_test, y_test)

                            results.append({
                                'activation': activation,
                                'learning_rate': lr,
                                'epochs': epochs,
                                'layers': layers,
                                'neurons': neurons,
                                'train_acc': train_acc,
                                'test_acc': test_acc,
                            })

                            # Plot the model history
                            plt.plot(model.loss_curve_,
                                     label=f'{activation}, lr={lr}, epochs={epochs}, layers={layers}, neurons={neurons}')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Model Loss vs. Epochs for Various Hyperparameters')
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.savefig('model_history_plot.png')  # Save the plot to a file
        plt.close()

        # Print results in tabular format
        results_df = pd.DataFrame(results)
        print(results_df)

        return results_df


if __name__ == "__main__":
    neural_network = NeuralNet("https://raw.githubusercontent.com/smart79/spambase/main/spambase/spambase.csv")
    neural_network.preprocess()
    results_df = neural_network.train_evaluate()
    results_df.to_csv('results.csv', index=False)
