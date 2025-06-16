# prompt: Build an Artificial Neural Network by implementing the Backpropagation algorithm and test the
# same using appropriate data sets.

import numpy as np

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Function to initialize weights and biases
def initialize_network(input_size, hidden_size, output_size):
    weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
    bias_hidden = np.random.uniform(size=(1, hidden_size))
    weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))
    bias_output = np.random.uniform(size=(1, output_size))
    return weights_input_hidden, bias_hidden, weights_hidden_output, bias_output

# Forward propagation
def forward_propagation(X, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)
    return hidden_layer_output, predicted_output

# Backward propagation
def backward_propagation(X, y, hidden_layer_output, predicted_output, weights_hidden_output, learning_rate):
    # Calculate output layer error and delta
    output_error = y - predicted_output
    output_delta = output_error * sigmoid_derivative(predicted_output)

    # Calculate hidden layer error and delta
    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

    # Calculate weight updates
    weights_hidden_output_update = hidden_layer_output.T.dot(output_delta) * learning_rate
    weights_input_hidden_update = X.T.dot(hidden_delta) * learning_rate

    # Calculate bias updates
    bias_output_update = np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    bias_hidden_update = np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    return weights_input_hidden_update, bias_hidden_update, weights_hidden_output_update, bias_output_update

# Update weights and biases
def update_weights_and_biases(weights_input_hidden, bias_hidden, weights_hidden_output, bias_output,
                              weights_input_hidden_update, bias_hidden_update, weights_hidden_output_update, bias_output_update):
    weights_input_hidden += weights_input_hidden_update
    bias_hidden += bias_hidden_update
    weights_hidden_output += weights_hidden_output_update
    bias_output += bias_output_update
    return weights_input_hidden, bias_hidden, weights_hidden_output, bias_output

# Training the neural network
def train_neural_network(X, y, input_size, hidden_size, output_size, learning_rate, epochs):
    weights_input_hidden, bias_hidden, weights_hidden_output, bias_output = initialize_network(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        # Forward pass
        hidden_layer_output, predicted_output = forward_propagation(X, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)

        # Backward pass
        weights_input_hidden_update, bias_hidden_update, weights_hidden_output_update, bias_output_update = backward_propagation(
            X, y, hidden_layer_output, predicted_output, weights_hidden_output, learning_rate
        )

        # Update weights and biases
        weights_input_hidden, bias_hidden, weights_hidden_output, bias_output = update_weights_and_biases(
            weights_input_hidden, bias_hidden, weights_hidden_output, bias_output,
            weights_input_hidden_update, bias_hidden_update, weights_hidden_output_update, bias_output_update
        )

        # Calculate and print loss every 1000 epochs
        if epoch % 1000 == 0:
            loss = np.mean(np.square(y - predicted_output))
            print(f"Epoch {epoch}, Loss: {loss}")

    return weights_input_hidden, bias_hidden, weights_hidden_output, bias_output

# --- Example Usage with a simple XOR dataset ---

# Input dataset (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Output dataset (XOR problem)
y = np.array([[0], [1], [1], [0]])

# Define hyperparameters
input_size = 2
hidden_size = 4  # You can adjust the hidden layer size
output_size = 1
learning_rate = 0.1
epochs = 10000

# Train the neural network
weights_input_hidden, bias_hidden, weights_hidden_output, bias_output = train_neural_network(
    X, y, input_size, hidden_size, output_size, learning_rate, epochs
)

# --- Testing the trained network ---

print("\nTesting the trained network:")
test_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

for input_data in test_X:
    # Perform forward propagation with the trained weights and biases
    hidden_layer_output, predicted_output = forward_propagation(
        input_data, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output
    )
    print(f"Input: {input_data}, Predicted Output: {predicted_output[0]:.4f}")

