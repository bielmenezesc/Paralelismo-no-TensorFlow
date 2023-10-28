import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Função de pré-processamento a ser aplicada usando .map()
def preprocess_image(image, label):
    # Normalização dos pixels para um intervalo de [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Carregar os dados do MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Definir modelo
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Dividir o conjunto de treinamento em mini lotes e aplicar pré-processamento
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.map(preprocess_image)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)  # Otimização do pipeline

# Treinar o modelo com paralelismo de dados
import time

start_time = time.time()
model.fit(train_dataset, epochs=5)
end_time = time.time()

print("Tempo de treinamento com paralelismo de dados:", end_time - start_time, "segundos")


















