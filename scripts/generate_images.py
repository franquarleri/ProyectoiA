import tensorflow as tf
from tensorflow.keras import layers
import os

# Definición del generador
generator = tf.keras.Sequential([
    layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)),
    layers.BatchNormalization(),
    layers.LeakyReLU(),

    layers.Reshape((7, 7, 256)),

    layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.LeakyReLU(),

    layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.LeakyReLU(),

    layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
])

# Generar una imagen de ejemplo
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

# Asegurar que la forma del tensor sea tridimensional
generated_image = tf.squeeze(generated_image, axis=0)  # Eliminar la dimensión de lote si es necesario hola navho

# Asegurar que la forma de la imagen sea tridimensional
if len(generated_image.shape) == 2:
    generated_image = tf.expand_dims(generated_image, axis=-1)  # Agregar la dimensión del canal si es necesario

# Crear la carpeta 'images' si no existe
os.makedirs('images', exist_ok=True)

# Guardar la imagen generada en la carpeta 'images'
tf.keras.preprocessing.image.save_img('images/generat_image.png', generated_image * 0.5 + 0.5)




