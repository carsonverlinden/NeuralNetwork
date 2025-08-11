import numpy as np
# NeuralNet.py

# ------------------------------------------------------------
# Class: NeuralNet
# Description:
#   Implements a basic fully connected feedforward neural network
#   from scratch using NumPy for matrix operations.
#   Supports:
#   - Configurable number of hidden layers
#   - ReLU activation for hidden layers
#   - Softmax activation for output layer
#   - Cross-entropy loss for classification
#   - Mini-batch gradient descent training
#   - MNIST dataset loading example in __main__
# ------------------------------------------------------------

# ------------------------------------------------------------
    #Initialize network architecture, weights, and biases.

    #Parameters:
    #- input_size: Number of input features (default 784 for MNIST 28x28 images).
    #- hidden_size: List with neuron counts for each hidden layer.
    #- output_size: Number of output classes (default 10 for MNIST digits).

    #Process:
    #1. Store architecture structure in self.layers.
    #2. Initialize weights with small random values (Gaussian scaled by 0.01).
    #3. Initialize biases to zeros for all layers.
# ------------------------------------------------------------
class NeuralNet:
    def __init__ (self, input_size = 784, hidden_size = [1024, 512, 256, 128, 64, 32], output_size = 10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights = [] #list of weight matrices
        self.biases = [] #list of bias vectors
        self.layers = [input_size] + hidden_size + [output_size]

        #Initialize weights and biases for each layer
        for i in range(len(self.layers) - 1):
        #Input to the hidden layers
            self.weights.append(0.01 * np.random.randn(self.layers[i], self.layers[i+1]))
            self.biases.append(np.zeros((1, self.layers[i+1])))



    # ------------------------------------------------------------
        #Normalize input data to the range [0, 1] using min-max normalization.

        #Parameters:
        #- X: Input data array of shape [num_samples, num_features].

        #Returns:
        #- Normalized data with same shape as X.
    # ------------------------------------------------------------
    def normalize_data(X):
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        return (X - X_min) / (X_max - X_min + 1e-8)



    # ------------------------------------------------------------
        #ReLU activation function: max(0, z)
        #- Introduces non-linearity to the model.
        #- Sets all negative values to 0 while keeping positive values unchanged.
        # Parameters:
        #- z: Pre-activation values (numpy array) 
        #Returns:
        #- Activated values after applying ReLU.
    # ------------------------------------------------------------
    def relu(self, z):
        return np.maximum(0, z)



    # ------------------------------------------------------------
        #Derivative of ReLU activation function.
        #- Returns 1 for positive z, 0 for non-positive z
        #Parameters:
        #- z: Pre-activation values (numpy array)
        #Returns:
        #- Derivative values (1 or 0) for each element in z.
    # ------------------------------------------------------------
    def relu_derivative(self, z):
        #Derivative of ReLU activation function
        #Returns 1 for positive z, 0 for non-positive z
        return np.where(z > 0, 1, 0)



    # ------------------------------------------------------------
        #Softmax activation function:
        #- Converts raw output logits into probability distributions.
        #- Each output is between 0 and 1, and the sum across outputs = 1.
        #Implementation details:
        #- Subtracts np.max(z) for numerical stability to prevent overflow.
        #Parameters:
        #- z: Pre-softmax logits (numpy array of shape [batch_size, output_size])
        #Returns:
        #- Probability distribution across classes for each input sample.
    # ------------------------------------------------------------
    def softmax(self, z):
        # Softmax function to convert logits to probabilities
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)



    # ------------------------------------------------------------
        #Perform a forward pass through the network.
        #Steps:
        #1. Start with the input layer.
        #2. For each layer:
            #- Multiply by weights, add biases (linear transformation).
            #- Apply ReLU for all hidden layers (non-linear activation).
        #3. For the output layer:
            #- Skip ReLU and apply softmax to get probabilities.
        #Parameters:
        #- inputs: Input data matrix of shape [batch_size, input_size]
        #Returns:
        #- Final output probabilities of shape [batch_size, output_size].
    # ------------------------------------------------------------
    def forward(self, inputs):
        #Store the outputs of each layer (for possible future use)
        activations = [inputs]
        zs = []
        a = inputs
        #Iterate through each layer's weights and biases
        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            zs.append(z)
            # Now we need to apply ReLU activation to all links or "layers" to the network except the last
            if i < len(self.weights) - 1:
                a = self.relu(z)
            else:
                a = self.softmax(z)
            activations.append(a)
            # Raw data outputs or (logits) need to be converted to probabilities using softmax
        return activations, zs



    # ------------------------------------------------------------
        #Compute the cross-entropy loss between predicted and true labels.
        #This loss function is commonly used for multi-class classification tasks.
        #Parameters:
        #- y_pred: Predicted probabilities from the model (output of softmax).
        #- y_true: True labels (one-hot encoded or integer labels).
        #Returns:
        # - Cross-entropy loss value.
    # ------------------------------------------------------------
    def cross_entropy_loss(self, y_pred, y_true):
        #y_pred: [batch_size, num_classes], y_true: [batch_size] (integer labels)
        m = y_pred.shape[0]
        #Clip predictions to avoid log(0)
        p = np.clip(y_pred, 1e-12, 1. - 1e-12)
        log_likelihood = -np.log(p[range(m), y_true])
        loss = np.sum(log_likelihood)/ m
        return loss



    # ------------------------------------------------------------
        #Perform backpropagation to compute gradients and update parameters.

        #Steps:
        #1. Compute output layer error (delta).
        #2. Propagate errors backward through layers.
        #3. Compute weight and bias gradients.
        #4. Update parameters using gradient descent.

        #Parameters:
        #- activations: Outputs of each layer from forward pass.
        #- zs: Pre-activation values from forward pass.
        #- y_true: True class labels (integer-encoded).
        #- learning_rate: Step size for gradient descent updates.
    # ------------------------------------------------------------
    def backwardProp(self, activations, zs, y_true, learning_rate = 0.01):
        #forward pass to get predictions
        #Initializers
        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)
        m = activations[0].shape[0]

        #Output layer error protection
        delta = activations[-1].copy()
        #Adjust the delta for the output layer
        delta[range(m), y_true] -= 1
        delta /= m

        #Compute gradients for each layer
        for i in reversed(range(len(self.weights))):
            grads_w[i] = np.dot(activations[i].T, delta)
            grads_b[i] = np.sum(delta, axis = 0, keepdims = True)
            if i != 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(zs[i-1])

        #Update the weights and biases using the computed gradients
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grads_w[i]
            self.biases[i] -= learning_rate * grads_b[i]



    # ------------------------------------------------------------
        #Train the network using mini-batch gradient descent.

        #Parameters:
        #- X: Training data [num_samples, num_features].
        #- y: True labels (integer-encoded) [num_samples].
        #- epochs: Number of training iterations over the full dataset.
        #- learning_rate: Step size for weight updates.
        #- batch_size: Number of samples per training batch.
    # ------------------------------------------------------------
    def train(self, X, y, epochs = 1000, learning_rate = 0.01, batch_size = 32):
        n = X.shape[0]
        for epoch in range(epochs):
            #shuffle the data
            indices = np.arange(n)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            #Forward pass
            for start in range(0, n, batch_size):
                end = start + batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]
                activations, zs = self.forward(X_batch)
                loss = self.cross_entropy_loss(activations[-1], y_batch)
                self.backwardProp(activations, zs, y_batch, learning_rate)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")



# ------------------------------------------------------------
    # Main program:
    # Loads MNIST dataset, preprocesses it, trains the network.
    # Evaluates accuracy on the test set.
# ------------------------------------------------------------
if __name__ == "__main__":
    from tensorflow.keras.datasets import mnist

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 28*28)
    X_test = X_test.reshape(-1, 28*28)

    X_train = NeuralNet.normalize_data(X_train)
    X_test = NeuralNet.normalize_data(X_test)

    net = NeuralNet(input_size=784, hidden_size=[128, 64], output_size = 10)
    net.train(X_train, y_train, epochs = 30, learning_rate = 0.01, batch_size = 64)

    activations, _ = net.forward(X_test)
    y_pred = np.argmax(activations[-1], axis = 1)
    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")