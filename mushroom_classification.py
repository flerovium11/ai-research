"""
AI Prompt:

Die Pilze haben je zwei Koordinaten, Höhe und Hutdurchmesser, welche zwischen 0 und 1 liegen.

Testdaten:

Pilz 1: (0.1, 0.8) -> giftig
Pilz 2: (0.07, 0.45) -> giftig
Pilz 3: (0.5, 0.75) -> giftig
Pilz 4: (0.8, 0.5) -> giftig
Pilz 5: (0.45, 0.4) -> essbar
Pilz 6: (0.45, 0.1) -> essbar
Pilz 7: (0.85, 0.15) -> essbar
Pilz 8: (0.85, 0.8) -> essbar

Das KNN hat zwei Eingabeneuronen, zwei versteckte Neuronen und ein Ausgabeneuron (>= 0.5 für giftig und umgekehrt).

Trainiere das KNN mit den Testdaten, stelle die Pilze grafisch in einem Koordinatensystem dar und zeige die Entscheidungsgrenze des KNNs an. Und gib die Gewichte & Bias der Neuronen aus.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_ih = np.random.randn(self.input_size, self.hidden_size)
        self.bias_h = np.zeros((1, self.hidden_size))
        self.weights_ho = np.random.randn(self.hidden_size, self.output_size)
        self.bias_o = np.zeros((1, self.output_size))

    def forward(self, X):
        self.hidden = sigmoid(np.dot(X, self.weights_ih) + self.bias_h)
        self.output = sigmoid(
            np.dot(self.hidden, self.weights_ho) + self.bias_o)
        return self.output

    def update_and_save_plot(self):
        xx, yy = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
        Z = self.forward(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(10, 8))
        contourf = plt.contourf(xx, yy, Z, levels=np.linspace(
            0, 1, 20), cmap=plt.cm.RdYlBu_r)
        plt.scatter(X[:4, 0], X[:4, 1], c='darkred', label='Giftiger Pilz')
        plt.scatter(X[4:, 0], X[4:, 1], c='darkblue', label='Ungiftiger Pilz')
        plt.xlabel('Hutdurchmesser')
        plt.ylabel('Höhe')
        plt.title('Pilzklassifikation mit neuronalem Netzwerk')
        plt.legend()
        plt.colorbar(contourf, label='Wahrscheinlichkeit für Giftigkeit')
        plt.axhline(y=0, color='k', linestyle='--')
        plt.axvline(x=0, color='k', linestyle='--')
        plt.savefig('mushroom_classification.png')
        plt.close()

    def visualize_network(self, filename='neural_network_visualization.png'):
        fig, ax = plt.subplots(figsize=(15, 10))

        layer_sizes = [self.input_size, self.hidden_size, self.output_size]
        layer_positions = [1, 3, 5]

        def draw_neuron(pos, layer, index):
            circle = Circle(pos, radius=0.4, fill=False)
            ax.add_patch(circle)

            if layer == 0:
                ax.text(pos[0], pos[1], f'''x{
                        index+1}''', ha='center', va='center')
            else:
                ax_inset = fig.add_axes([pos[0]-0.35, pos[1]-0.35, 0.7, 0.7])
                xx, yy = np.meshgrid(
                    np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
                X_grid = np.c_[xx.ravel(), yy.ravel()]

                if layer == 1:
                    Z = sigmoid(
                        np.dot(X_grid, self.weights_ih[:, index]) + self.bias_h[0, index])
                else:
                    hidden = sigmoid(
                        np.dot(X_grid, self.weights_ih) + self.bias_h)
                    Z = sigmoid(np.dot(hidden, self.weights_ho) + self.bias_o)

                Z = Z.reshape(xx.shape)
                ax_inset.contourf(xx, yy, Z, levels=20, cmap='RdYlBu')
                ax_inset.set_xticks([])
                ax_inset.set_yticks([])

        def draw_connection(start, end, weight):
            arrow = FancyArrowPatch(
                start, end, arrowstyle='->', mutation_scale=10, color='gray', linewidth=abs(weight))
            ax.add_patch(arrow)

        for layer, num_neurons in enumerate(layer_sizes):
            for i in range(num_neurons):
                x = layer_positions[layer]
                y = (i - (num_neurons - 1) / 2) * 1.5
                draw_neuron((x, y), layer, i)

        for layer in range(len(layer_sizes) - 1):
            for i in range(layer_sizes[layer]):
                for j in range(layer_sizes[layer + 1]):
                    start = (
                        layer_positions[layer], (i - (layer_sizes[layer] - 1) / 2) * 1.5)
                    end = (layer_positions[layer + 1], (j -
                           (layer_sizes[layer + 1] - 1) / 2) * 1.5)
                    weight = self.weights_ih[i,
                                             j] if layer == 0 else self.weights_ho[i, j]
                    draw_connection(start, end, weight)

        ax.set_xlim(0, 6)
        ax.set_ylim(-2, 2)
        ax.axis('off')
        plt.title('Neural Network Structure with Decision Boundaries')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def train(self, X, y, learning_rate=0.1, epochs=5000):
        for _ in range(epochs):
            self.forward(X)

            error = y - self.output
            d_output = error * sigmoid_derivative(self.output)

            error_hidden = np.dot(d_output, self.weights_ho.T)
            d_hidden = error_hidden * sigmoid_derivative(self.hidden)

            self.weights_ho += np.dot(self.hidden.T, d_output) * learning_rate
            self.bias_o += np.sum(d_output, axis=0,
                                  keepdims=True) * learning_rate
            self.weights_ih += np.dot(X.T, d_hidden) * learning_rate
            self.bias_h += np.sum(d_hidden, axis=0,
                                  keepdims=True) * learning_rate

            if _ % 100 == 0:
                self.update_and_save_plot()


X = np.array([[-0.8, 0.6], [-0.86, -0.1], [0, 0.5], [0.6, 0],
              [-0.1, -0.2], [-0.1, -0.8], [0.7, -0.7], [0.7, 0.6]])
y = np.array([[1], [1], [1], [1], [0], [0], [0], [0]])

nn = NeuralNetwork(2, 3, 1)
nn.train(X, y, epochs=10000)

# nn.visualize_network()

print("Gewichte (Eingang zu versteckte Schicht):")
print(nn.weights_ih)
print("\nBias (versteckte Schicht):")
print(nn.bias_h)
print("\nGewichte (versteckte Schicht zu Ausgang):")
print(nn.weights_ho)
print("\nBias (Ausgangsschicht):")
print(nn.bias_o)
