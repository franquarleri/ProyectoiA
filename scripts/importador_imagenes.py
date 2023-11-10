import tensorflow as tf
import tensorflow_datasets as tfds

# Carga el conjunto de datos OpenImages V4
openimages_v4 = tfds.load('open_images_v4', split='train_sample')

# Filtra las imágenes que no son de mujeres
def is_female(image, label):
  """Devuelve `True` si la imagen es de una mujer.

  Args:
    image: La imagen a evaluar.
    label: La etiqueta de la imagen.

  Returns:
    `True` si la imagen es de una mujer.
  """

  return label['label'] == 'person' and label['attributes']['gender'] == 'female'

def is_in_age_range(image, label):
  """Devuelve `True` si la imagen es de una mujer de entre 15 y 45 años.

  Args:
    image: La imagen a evaluar.
    label: La etiqueta de la imagen.

  Returns:
    `True` si la imagen es de una mujer de entre 15 y 45 años.
  """

  return label['label'] == 'person' and label['attributes']['gender'] == 'female' and label['attributes']['age'].between(15, 45)

filtered_dataset = openimages_v4.filter(is_female).filter(is_in_age_range)

# Convertir el conjunto de datos a un conjunto de datos de TensorFlow
dataset = filtered_dataset.as_dataset()

# Entrenar el generador de imágenes condicionales
model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(256, 256, 3)),
  tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
  tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
  tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1024),
  tf.keras.layers.Dense(784),
  tf.keras.layers.Reshape((28, 28, 1)),
])

model.compile(optimizer='adam', loss='mse')
model.fit(dataset, epochs=10)

# Generar una imagen
generated_image = model.predict(tf.random.normal((1, 28, 28, 1)))

# Guardar la imagen
tfio.write_file('generated_image.jpg', generated_image)
