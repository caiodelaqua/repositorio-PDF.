import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# Carregar o dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Exibir algumas informações sobre o dataset
print(f'Tamanho do conjunto de treinamento: {x_train.shape}')
print(f'Tamanho do conjunto de teste: {x_test.shape}')

# Normalizar os dados para o intervalo [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Visualizar algumas imagens do conjunto de treinamento
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
plt.show()

# Construir um modelo simples usando Keras com tangente hiperbólica
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='tanh'),  # Troca de ReLU para tanh
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo com 10 épocas
model.fit(x_train, y_train, epochs=10)

# Avaliar o modelo no conjunto de teste
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'\nAcurácia no conjunto de teste: {test_acc}')

# Fazer previsões no conjunto de teste
predictions = model.predict(x_test)

# Função para plotar as imagens com suas previsões
def plot_image_with_prediction(image, true_label, predicted_label):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap=plt.cm.binary)
    color = 'green' if predicted_label == true_label else 'red'  # Troca para verde
    plt.xlabel(f"Verdadeiro: {true_label}, Predito: {predicted_label}", color=color)

# Mostrar uma amostra de 5 imagens do conjunto de teste com suas previsões
num_images = 5  # Mostrar apenas 5 imagens
plt.figure(figsize=(10, 10))
for i in range(num_images):
    plt.subplot(5, 2, i + 1)
    true_label = y_test[i]
    predicted_label = np.argmax(predictions[i])
    plot_image_with_prediction(x_test[i], true_label, predicted_label)
plt.tight_layout()
plt.show()