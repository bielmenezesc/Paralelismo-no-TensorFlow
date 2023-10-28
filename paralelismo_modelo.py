import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import time

# Carregar os dados do MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Definir o contexto de distribuição estratégica
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Definir modelo dentro do escopo de estratégia distribuída
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Treinar o modelo com paralelismo usando distribuição de estratégia
start_time = time.time()
model.fit(train_images, train_labels, epochs=5, batch_size=64)
end_time = time.time()

print("Tempo de treinamento com paralelismo de modelo:", end_time - start_time, "segundos")












