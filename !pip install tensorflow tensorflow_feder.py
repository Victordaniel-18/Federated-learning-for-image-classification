!pip install tensorflow tensorflow_federated matplotlib numpy
!pip install --upgrade jax jaxlib
!pip install triton
!pip install tensorflow
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize images to [0,1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to int32 and flatten
y_train = y_train.flatten().astype(np.int32)
y_test = y_test.flatten().astype(np.int32)

# Convert to float32
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# Create federated data
def create_federated_data(x, y, num_clients=10):
    client_data = []
    data_per_client = len(x) // num_clients
    for i in range(num_clients):
        start = i * data_per_client
        end = (i + 1) * data_per_client
        dataset = tf.data.Dataset.from_tensor_slices((x[start:end], y[start:end]))
        dataset = dataset.shuffle(1000).batch(32)
        client_data.append(dataset)
    return client_data

NUM_CLIENTS = 10
federated_train_data = create_federated_data(x_train, y_train, NUM_CLIENTS)

# Define Keras model
def create_keras_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Model function for TFF
def model_fn():
    keras_model = create_keras_model()
    input_spec = federated_train_data[0].element_spec
    return tff.learning.models.from_keras_model(
        keras_model=keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

# Federated averaging algorithm
iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn=model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
)

# Initialize training
state = iterative_process.initialize()

# Training loop
NUM_ROUNDS = 5
accuracy_list = []

for round_num in range(1, NUM_ROUNDS + 1):
    result = iterative_process.next(state, federated_train_data)
    state = result.state
    metrics = result.metrics

    # Directly extract from the known structure
    accuracy = metrics['client_work']['train']['sparse_categorical_accuracy']
    accuracy_list.append(accuracy)

    print(f'Round {round_num}, Train Accuracy: {accuracy:.4f}')


# Plot accuracy curve
plt.plot(range(1, NUM_ROUNDS + 1), accuracy_list, marker='o')
plt.xlabel('Round')
plt.ylabel('Train Accuracy')
plt.title('Federated Learning Accuracy Over Rounds')
plt.grid(True)
plt.show()
