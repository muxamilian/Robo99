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

all_results = []

for i in range(4, image_shape[0]+1): 
   if image_shape[0] % i == 0:
      all_results.append(i)

normalized_probabilities = [1 / len(all_results) if i in all_results else 0. for i in range(image_shape[0]+1)]
print('all_results', all_results, 'normalized_probabilities', normalized_probabilities)

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

    # new_res = tf.random.uniform((), minval=4, maxval=image_shape[0]+1, dtype=tf.int32)
    new_res = tf.squeeze(tf.random.categorical(tf.math.log([normalized_probabilities]), 1, dtype=tf.int32))
    random_vector_downscaled = tf.random.normal((1,new_res,new_res,1), stddev=0.5)
    random_vector = tf.random.normal((1,*image_shape,1), stddev=0.5)
    
    image = tf.expand_dims(image, 0)
    # image = tf.image.resize(image, (new_res,new_res), method=tf.image.ResizeMethod.AREA)
    image = tf.reshape(image, [1, new_res, image_shape[0]//new_res, new_res, image_shape[1]//new_res, 3])
    image = tf.reduce_mean(image, axis=4)
    image = tf.reduce_mean(image, axis=2)
    image += random_vector_downscaled
    image = tf.image.resize(image, image_shape, tf.image.ResizeMethod.BILINEAR)
    image += random_vector
    image = tf.squeeze(image, 0)
    image = tfp.math.clip_by_value_preserve_gradient(image, -1., 1.)
    # tf.Assert(tf.reduce_any(image > 0.), [image])
    # tf.Assert(tf.reduce_any(image < 0.), [image])
    # tf.Assert(tf.reduce_all(image <= 1), [image])
    # tf.Assert(tf.reduce_all(image >= -1), [image])

    return image, label
