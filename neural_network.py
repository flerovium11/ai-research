import matplotlib.pyplot as plt
from copy import deepcopy
import os
from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional
import random
import numpy as np
from PIL import Image
from IPython.display import clear_output


class DifferentiableFunction(ABC):

    @abstractmethod
    def __call__(self, values: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, values: np.ndarray, targets: Optional[np.ndarray] = None) -> np.ndarray:
        pass


class Sigmoid(DifferentiableFunction):

    @staticmethod
    def __call__(values: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-values))

    @staticmethod
    def derivative(values: np.ndarray, targets: Optional[np.ndarray] = None) -> np.ndarray:
        sig = Sigmoid.__call__(values)
        return sig * (1 - sig)


class ReLU(DifferentiableFunction):

    @staticmethod
    def __call__(values: np.ndarray) -> np.ndarray:
        return np.maximum(0, values)

    @staticmethod
    def derivative(values: np.ndarray, targets: Optional[np.ndarray] = None) -> np.ndarray:
        return np.where(values > 0, 1, 0)


class Softmax(DifferentiableFunction):
    @staticmethod
    def __call__(values: np.ndarray) -> np.ndarray:
        # for numerical stability
        shifted = values - np.max(values, axis=-1, keepdims=True)
        exps = np.exp(shifted)
        return exps / np.sum(exps, axis=-1, keepdims=True)

    @staticmethod
    def derivative(values: np.ndarray, targets: Optional[np.ndarray] = None) -> np.ndarray:
        softmax_vals = Softmax.__call__(values)
        return softmax_vals - targets


class Tanh(DifferentiableFunction):

    @staticmethod
    def __call__(values: np.ndarray) -> np.ndarray:
        return np.tanh(values)

    @staticmethod
    def derivative(values: np.ndarray, targets: Optional[np.ndarray] = None) -> np.ndarray:
        return 1 - np.tanh(values) ** 2


class Linear(DifferentiableFunction):

    @staticmethod
    def __call__(values: np.ndarray) -> np.ndarray:
        return values

    @staticmethod
    def derivative(values: np.ndarray, targets: Optional[np.ndarray] = None) -> np.ndarray:
        return np.ones_like(values)


class LeakyReLU(DifferentiableFunction):

    @staticmethod
    def __call__(values: np.ndarray) -> np.ndarray:
        return np.where(values > 0, values, 0.01 * values)

    @staticmethod
    def derivative(values: np.ndarray, targets: Optional[np.ndarray] = None) -> np.ndarray:
        return np.where(values > 0, 1, 0.01)


class CrossEntropy(DifferentiableFunction):

    @staticmethod
    def __call__(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return -np.sum(targets * np.log(values)) / values.shape[0]

    @staticmethod
    def derivative(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return -targets / values


class MeanSquaredError(DifferentiableFunction):

    @staticmethod
    def __call__(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return 0.5 * np.sum((values - targets) ** 2)

    @staticmethod
    def derivative(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return values - targets


class WeightInitialization(ABC):

    @abstractmethod
    def __call__(input_size: int, output_size: int) -> np.array:
        pass


class XavierInitialization(WeightInitialization):

    @staticmethod
    def __call__(input_size: int, output_size: int) -> np.array:
        limit = np.sqrt(6 / (input_size + output_size))
        return np.random.uniform(-limit, limit, (output_size, input_size))


class HeInitialization(WeightInitialization):

    @staticmethod
    def __call__(input_size: int, output_size: int) -> np.array:
        stddev = np.sqrt(2 / input_size)
        return np.random.normal(0, stddev, (output_size, input_size))


class RandomInitialization(WeightInitialization):

    @staticmethod
    def __call__(input_size: int, output_size: int) -> np.array:
        return np.random.random((output_size, input_size)) * 2 - 1


class StaticInitialization(WeightInitialization):

    @staticmethod
    def __call__(input_size: int, output_size: int, value: float) -> np.array:
        return np.full((output_size, input_size), value)


class Layer:
    def __init__(
        self: any,
        output_size: int,
        activation_function: DifferentiableFunction,
        weight_initialization: WeightInitialization = RandomInitialization(),
    ) -> None:
        self.weighted_sums = np.zeros(output_size)
        self.output = np.zeros(output_size)
        self.node_deltas = np.zeros(output_size)
        self.activation_function = activation_function
        self.output_size = output_size
        self.weight_initialization = weight_initialization

    def initialize_weights(self, input_size: int) -> None:
        self.input_size = input_size
        self.weights = self.weight_initialization(
            input_size + 1, self.output_size)

    def forward(self, inputs: np.array) -> None:
        self.weighted_sums = np.dot(self.weights, np.append(inputs, 1))
        self.output = self.activation_function(self.weighted_sums)

    def backward(self, next_layer: any) -> None:
        self.node_deltas = np.dot(
            next_layer.weights[:, :-1].T, next_layer.node_deltas) * self.activation_function.derivative(self.weighted_sums)

    def update_weights(self, learning_rate: float, inputs: np.array) -> None:
        gradient = np.outer(self.node_deltas, np.append(inputs, 1))
        numerical_gradient = self.numerical_gradient(inputs, gradient, 1e-5)
        print('gradient', gradient)
        print('numerical_gradient', numerical_gradient)

        if self.gradient_clipping:
            gradient = np.clip(
                gradient, -self.gradient_clipping, self.gradient_clipping)

        self.weights -= learning_rate * gradient

    def numerical_gradient(self, inputs: np.array, targets: np.array, epsilon: float) -> np.array:
        gradients = np.zeros(self.weights.shape)
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                self.weights[i, j] += epsilon
                loss_plus = self.activation_function(
                    CrossEntropy.__call__(self.forward(inputs), targets))
                self.weights[i, j] -= 2 * epsilon
                loss_minus = self.activation_function(
                    CrossEntropy.__call__(self.forward(inputs), targets))
                self.weights[i, j] += epsilon
                gradients[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
        return gradients


class NeuralNetwork:
    def __init__(
        self: any,
        input_size: int,
        layers: list[Layer],
        error_function: DifferentiableFunction,
    ) -> None:
        self.layers = layers
        self.inputs = np.zeros(input_size)
        self.error_function = error_function
        self.error = 0
        self.initialize_layers()

    def initialize_layers(self: any) -> None:
        for i, layer in enumerate(self.layers):
            if self.gradient_clipping is not None:
                layer.gradient_clipping = self.gradient_clipping

            if i == 0:
                layer.initialize_weights(self.inputs.shape[0])
            else:
                layer.initialize_weights(self.layers[i - 1].output_size)

    def predict(self: any, inputs: np.array) -> np.array:
        self.inputs = inputs

        for layer in self.layers:
            layer.forward(inputs)
            inputs = layer.output

        return inputs

    def calculate_error(self: any, targets: np.array) -> float:
        self.error = self.error_function(self.layers[-1].output, targets)
        return self.error

    def backpropagate(self: any, targets: np.array) -> None:
        self.calculate_error(targets)

        for i, layer in enumerate(reversed(self.layers)):
            if i == 0:
                layer.node_deltas = self.error_function.derivative(
                    layer.output, targets) * layer.activation_function.derivative(layer.weighted_sums, targets) * layer.output
            else:
                layer.backward(self.layers[-i])

    def update_weights(self: any, learning_rate: float) -> None:
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.update_weights(learning_rate, self.inputs)
            else:
                layer.update_weights(learning_rate, self.layers[i - 1].output)


class Trainer:
    def __init__(
        self: any,
        neural_network: NeuralNetwork,
        epochs: int,
        batch_size: int,
        dropout_rate: float,
        start_learning_rate: float,
        end_learning_rate: float,
    ) -> None:
        self.neural_network = neural_network
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.start_learning_rate = start_learning_rate
        self.end_learning_rate = end_learning_rate

    def train(self: any, inputs: np.array, targets: np.array, val_inputs: np.array, val_targets: np.array) -> list[float]:
        errors = []
        validate_errors = []
        learning_rate = self.start_learning_rate

        # flatten the images
        inputs = inputs.reshape(inputs.shape[0], -1)
        val_inputs = val_inputs.reshape(val_inputs.shape[0], -1)

        # one-hot encode the targets
        targets = np.eye(len(categories))[targets]
        val_targets = np.eye(len(categories))[val_targets]

        for epoch in range(self.epochs):
            clear_output(wait=True)

            indices = np.random.permutation(len(inputs))
            inputs = inputs[indices]
            targets = targets[indices]
            avg_error = 0

            for i in range(0, len(inputs), self.batch_size):
                batch_inputs = inputs[i:i + self.batch_size]
                batch_targets = targets[i:i + self.batch_size]

                for j in range(len(batch_inputs)):
                    self.neural_network.predict(batch_inputs[j])
                    self.neural_network.backpropagate(batch_targets[j])
                    self.neural_network.update_weights(learning_rate)
                    avg_error += self.neural_network.error

                learning_rate = self.start_learning_rate + \
                    (self.end_learning_rate - self.start_learning_rate) * \
                    (epoch / self.epochs)

            errors.append(avg_error / len(inputs))

            avg_validate_error = 0
            for i in range(len(val_inputs)):
                prediction = self.neural_network.predict(val_inputs[i])
                avg_validate_error += self.neural_network.error_function(
                    prediction, val_targets[i])

            validate_errors.append(avg_validate_error / len(val_inputs))

            print(
                f"Epoch {epoch + 1}/{self.epochs} - Validation Error: {validate_errors[-1]}")

            plt.plot(errors, label='Training error')
            plt.plot(validate_errors, label='Validation error')
            plt.legend()
            plt.ylim(0, np.max([*errors, *validate_errors]))
            plt.gcf().set_size_inches(10, 4)
            plt.show()

        return errors, validate_errors

    def test(self: any, inputs: np.array, targets: np.array) -> None:
        error_count = 0
        # flatten the images
        inputs = inputs.reshape(inputs.shape[0], -1)

        # one-hot encode the targets
        targets = np.eye(len(categories))[targets]

        for i in range(len(inputs)):
            prediction = self.neural_network.predict(inputs[i])
            if np.argmax(prediction) != np.argmax(targets[i]):
                error_count += 1

            print(f"Input: {inputs[i]}")
            print(f"Prediction: {prediction}")
            print(f"Target: {targets[i]}")
            print("")

        accuracy = 1 - error_count / len(inputs)
        print(f"Accuracy: {accuracy}")


if __name__ == 'main':
    # TODO: load data from dataset

    img_size = 28

    initialization = RandomInitialization()

    model = NeuralNetwork(
        input_size=img_size * img_size,
        layers=[
            Layer(128, ReLU(), initialization),
            Layer(64, ReLU(), initialization),
            Layer(len(categories), Softmax(), initialization),
        ],
        error_function=CrossEntropy(),
        # gradient_clipping = 1.0,
    )

    trainer = Trainer(
        neural_network=model,
        epochs=50,
        batch_size=1,
        dropout_rate=0.0,
        start_learning_rate=0.1,
        end_learning_rate=0.1,
    )

    errors, validate_errors = trainer.train(deepcopy(train_inputs), deepcopy(
        train_targets), deepcopy(validate_inputs), deepcopy(validate_targets))

    trainer.test(test_inputs, test_targets)

    np.save(f'models/model.npy', model)

    model = np.load('models/model.npy', allow_pickle=True)
    model = model.item()

    # continiously take a random image from the test set, show the models guess and the actual label, if i press enter, the next image is shown

    showcase_test_inputs = deepcopy(test_inputs)
    showcase_test_targets = deepcopy(test_targets)

    # one-hot encode the targets
    showcase_test_targets = np.eye(len(categories))[showcase_test_targets]

    # flatten the images
    showcase_test_inputs = showcase_test_inputs.reshape(
        showcase_test_inputs.shape[0], -1)

    while True:
        i = np.random.randint(0, len(showcase_test_inputs))
        plt.imshow(test_inputs[i], cmap='gray')
        plt.title(f"Prediction: {categories[np.argmax(model.predict(
            showcase_test_inputs[i]))]} - Target: {categories[np.argmax(showcase_test_targets[i])]}")
        plt.show()
        input()
