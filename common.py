import random
import tensorflow as tf
import tensorflow_probability as tfp
import keras_cv

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

random.seed(0)
tf.random.set_seed(0)

num_fonts = 56443
num_chars = 62
image_shape = [64, 64]
input_shape = (*image_shape, 3)

uniform_sampler = keras_cv.UniformFactorSampler(0.0, image_shape[0]/6)
blurrer = keras_cv.layers.RandomGaussianBlur(image_shape, uniform_sampler)

model = tf.keras.applications.MobileNetV3Small(
    input_shape=input_shape,
    classes=num_chars,
    classifier_activation=None,
    weights=None,
    include_preprocessing=False
)

@tf.function
def augment(image, label):
    image = tf.tile(image/255.*2.-1, (1, 1, 3))
    tf.Assert(tf.reduce_all(image <= 1), [image])
    tf.Assert(tf.reduce_all(image >= -1), [image])

    random_vector_first = tf.random.normal((*image_shape,1), stddev=0.25)
    random_vector = tf.random.normal((*image_shape,1), stddev=0.25)
    
    image += random_vector_first
    image = blurrer.augment_image(image, blurrer.get_random_transformation())
    image += random_vector
    image = tfp.math.clip_by_value_preserve_gradient(image, -1., 1.)
    # tf.Assert(tf.reduce_any(image > 0.), [image])
    # tf.Assert(tf.reduce_any(image < 0.), [image])
    # tf.Assert(tf.reduce_all(image <= 1), [image])
    # tf.Assert(tf.reduce_all(image >= -1), [image])

    return image, label
