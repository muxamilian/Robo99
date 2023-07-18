import random
import tensorflow as tf
import tensorflow_probability as tfp

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
    # tf.debugging.assert_less_equal(image, 1.)
    # tf.debugging.assert_greater_equal(image, -1.)
    # Because tf.random is broken on Apple devices on the GPU
    # with tf.device('/cpu:0'):
    new_res = tf.random.uniform((), minval=2, maxval=image_shape[0]+1, dtype=tf.int32)
    random_vector_downscaled = tf.random.normal((1,new_res,new_res,1), stddev=0.5)
    random_vector = tf.random.normal((1,*image_shape,1), stddev=0.5)
    
    image = tf.expand_dims(image, 0)
    image = tf.image.resize(image, (new_res,new_res))
    image += random_vector_downscaled
    image = tf.image.resize(image, image_shape)
    image += random_vector
    image = tf.squeeze(image, 0)
    image = tfp.math.clip_by_value_preserve_gradient(image, -1., 1.)

    return image, label
