# pip install apache-beam

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam import DoFn, ParDo
import apache_beam as beam
import time

# Função de pré-processamento
class PreprocessData(DoFn):
    def process(self, element):
        images, labels = element
        # Realizar pré-processamento dos dados aqui
        yield images, labels

# Função para treinar o modelo
class TrainModel(DoFn):
    def process(self, element, model):
        images, labels = element
        model.fit(images, labels, epochs=5, batch_size=64)

# Carregar os dados do MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalização dos dados
train_images = train_images / 255.0
test_images = test_images / 255.0

# Definir modelo
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Iniciar contagem de tempo
start_time = time.time()

# Iniciar pipeline do Apache Beam
with beam.Pipeline(options=PipelineOptions()) as pipeline:
    # Pré-processamento dos dados usando Apache Beam
    preprocessed_data = (
        pipeline
        | 'ReadData' >> beam.Create([(train_images, train_labels)])
        | 'PreprocessData' >> ParDo(PreprocessData())
    )

    # Treinamento do modelo usando Apache Beam
    _ = preprocessed_data | 'TrainModel' >> ParDo(TrainModel(), model=model)

# Calcular tempo total
end_time = time.time()
print("Tempo de treinamento com paralelismo de tarefa:", end_time - start_time, "segundos")
