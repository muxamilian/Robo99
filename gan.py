import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
from datetime import datetime
import os
from common import *

# dataset_name = 'dataset'
dataset_name = 'dataset-capitals'
if dataset_name == 'dataset-capitals':
  num_chars = 26

noise_dim = 128
batch_size = 64

multiplier = 3

epochs = 50
lr = 1e-4

def make_generator_model():
  gen = tf.keras.Sequential(
    [
      tf.keras.Input(shape=(noise_dim+1+num_chars,)),
      layers.Reshape ((1, 1, noise_dim+1+num_chars)),
      layers.Conv2DTranspose(multiplier*256, kernel_size=4, strides=4, padding='same', use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.2),
      layers.Conv2DTranspose(multiplier*128, kernel_size=4, strides=2, padding='same', use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.2),
      layers.Conv2DTranspose(multiplier*64, kernel_size=4, strides=2, padding='same', use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.2),
      layers.Conv2DTranspose(multiplier*32, kernel_size=4, strides=2, padding='same', use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.2),
      layers.Conv2DTranspose(1, kernel_size=4, strides=2, activation='tanh', padding='same')
    ],
    name="generator",
  )

  return gen

def make_discriminator_model():
  disc = tf.keras.Sequential(
    [
      layers.Input(shape=(*image_shape, 1+num_chars)),
      layers.Conv2D(multiplier*32 + 3*num_chars, (4, 4), padding='same', strides=2),
      layers.LeakyReLU(alpha=0.2),
      layers.Conv2D(multiplier*64 + 2*num_chars, (4, 4), padding='same', strides=2, use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.2),
      layers.Conv2D(multiplier*128 + 1*num_chars, (4, 4), padding='same', strides=2, use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.2),
      layers.Conv2D(multiplier*256, (4, 4), padding='same', strides=2, use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.2),
      layers.Conv2D(1, (4, 4), padding='same', strides=4, use_bias=False),
      layers.Flatten(),
    ],
    name="discriminator",
  )

  return disc


@tf.function
def augment_simple(image, label):
    image = image/255.*2.-1
    # tf.Assert(tf.reduce_all(image <= 1), [image])
    # tf.Assert(tf.reduce_all(image >= -1), [image])
    image = tfp.math.clip_by_value_preserve_gradient(image, -1., 1.)
    # tf.Assert(tf.reduce_any(image > 0.), [image])
    # tf.Assert(tf.reduce_any(image < 0.), [image])
    return image, label

dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_name,
    seed=0,
    shuffle=True,
    batch_size=None,
    color_mode='grayscale',
    image_size=image_shape                                            
).map(augment_simple).batch(batch_size, drop_remainder=True)

logdir = "gan_logs/" + datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
os.makedirs(logdir, exist_ok=True)
file_name = __file__.split('/')[-1]
with open(__file__) as fr:
   content = fr.read()
with open(logdir + file_name, 'w') as fw:
   fw.write(content)
file_writer = tf.summary.create_file_writer(logdir)
seed = tf.concat([tf.zeros([num_chars, noise_dim]), tf.one_hot(tf.range(0, num_chars), depth=num_chars)], axis=-1)

class CustomCallback(tf.keras.callbacks.Callback):

  def on_epoch_begin(self, epoch, logs):
    if epoch == 0:
      self.model.val_data = next(dataset.__iter__())[0]

      self.model.lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(lr, epochs*batches_per_epoch)

      self.model.generator_optimizer = tf.keras.optimizers.legacy.SGD(self.model.lr_decayed_fn)
      self.model.discriminator_optimizer = tf.keras.optimizers.legacy.SGD(self.model.lr_decayed_fn)
      # self.model.generator_optimizer = tf.keras.optimizers.legacy.Adam(self.model.lr_decayed_fn)#, beta_1=0.0)
      # self.model.discriminator_optimizer = tf.keras.optimizers.legacy.Adam(self.model.lr_decayed_fn)#, beta_1=0.0)
      # self.model.generator_optimizer = tf.keras.optimizers.legacy.RMSprop(self.model.lr_decayed_fn)
      # self.model.discriminator_optimizer = tf.keras.optimizers.legacy.RMSprop(self.model.lr_decayed_fn)

      with file_writer.as_default():
        tf.summary.image("In imgs", (self.model.val_data+1)/2, step=epoch, max_outputs=64)
            
    with file_writer.as_default():
      tf.summary.scalar('lr', self.model.lr_decayed_fn(epoch*batches_per_epoch), step=epoch)

    self.model.gen_loss_tracker.reset_states()
    self.model.disc_loss_tracker.reset_states()
    self.model.class_loss_tracker.reset_states()

  def on_epoch_end(self, epoch, logs=None):
    generated_images = self.model.generate()

    with file_writer.as_default():
      for tradeoff, results in generated_images:
        tf.summary.image(f"Out imgs {round(tradeoff, 3)}", (results+1)/2, step=epoch, max_outputs=64)

      for key in logs:
        tf.summary.scalar(key, logs[key], step=epoch)


class CustomModel(tf.keras.Model):

  def __init__(self):
    super(CustomModel, self).__init__()

    self.gen_loss_tracker = tf.keras.metrics.Mean(name="gen_loss")
    self.disc_loss_tracker = tf.keras.metrics.Mean(name="disc_loss")
    self.class_loss_tracker = tf.keras.metrics.Mean(name="class_loss")
    self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')
    self.sparse_categorical_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    self.generator = make_generator_model()
    print(self.generator.summary())
    self.discriminator = make_discriminator_model()
    print(self.discriminator.summary())
    self.classifier_model = model
    # self.classifier_model.load_weights('logs/20230704-173202/weights.47')
    self.classifier_model.load_weights('logs/20230721-120604/weights.29')

  def generate(self):
    results_tuple = []
    for tradeoff in tf.linspace(0., 1., 6):
        generated_images = self.generator(tf.concat([seed, tf.tile(tf.reshape(tradeoff, [1, 1]), (seed.shape[0], 1))], axis=-1), training=False)
        results_tuple.append((float(tradeoff), generated_images))
    return results_tuple

  def discriminator_loss(self, real_output, fake_output):
      real_loss = tf.reduce_mean(self.cross_entropy(tf.ones_like(real_output), real_output))
      fake_loss = tf.reduce_mean(self.cross_entropy(tf.zeros_like(fake_output), fake_output))
      total_loss = real_loss + fake_loss
      return total_loss

  def generator_loss(self, fake_output):
      return self.cross_entropy(tf.ones_like(fake_output), fake_output)

  @tf.function
  def train_step(self, batch):
      images, labels = batch
      # with tf.device('/cpu:0'):
      loss_tradeoff = tf.random.uniform([batch_size], minval=0, maxval=1, dtype=tf.float32)
      # loss_tradeoff = tf.zeros([batch_size], dtype=tf.float32)
      noise = tf.random.normal([batch_size, noise_dim])

      # tf.assert_equal(tf.shape(labels), (batch_size,))
      labels_one_hot = tf.one_hot(labels, depth=num_chars)

      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
          generated_images = self.generator(tf.concat([noise, labels_one_hot, tf.reshape(loss_tradeoff, (batch_size, 1))], axis=-1), training=True)

          tiled_labels = tf.tile(tf.reshape(labels_one_hot, [batch_size, 1, 1, num_chars]), [1, *image_shape, 1])
          real_output = self.discriminator(tf.concat([images, tiled_labels], axis=-1), training=True)
          fake_output = self.discriminator(tf.concat([generated_images, tiled_labels], axis=-1), training=True)

          gen_loss = self.generator_loss(fake_output)
          augmented_imgs_list = []
          for img in tf.unstack(generated_images, axis=0):
            rescaled_img = (img+1)/2*255
            new_img = augment(rescaled_img, tf.zeros([]))[0]
            augmented_imgs_list.append(new_img)
          augmented_imgs = \
            tf.stack(augmented_imgs_list, axis=0)
          tf.debugging.Assert(tf.reduce_all(augmented_imgs <= 1), [augmented_imgs])
          tf.debugging.Assert(tf.reduce_all(augmented_imgs >= -1), [augmented_imgs])
          class_loss = \
            self.sparse_categorical_cross_entropy(
              labels, 
              self.classifier_model(augmented_imgs, training=False))
          # class_loss = 0
          disc_loss = self.discriminator_loss(real_output, fake_output)
          # tf.assert_equal(tf.shape(gen_loss)[0], batch_size)
          # tf.assert_equal(tf.shape(class_loss)[0], batch_size)
          gen_grad_loss = tf.reduce_mean((1-loss_tradeoff)*gen_loss + loss_tradeoff*class_loss)
          # gen_grad_loss = tf.reduce_mean(gen_loss)

      gradients_of_generator = gen_tape.gradient(gen_grad_loss, self.generator.trainable_variables)
      gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

      self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
      self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

      self.gen_loss_tracker.update_state(tf.reduce_mean(gen_loss))
      self.disc_loss_tracker.update_state(disc_loss)
      self.class_loss_tracker.update_state(tf.reduce_mean(class_loss))

      return {
        "gen_loss": self.gen_loss_tracker.result(), 
        "disc_loss": self.disc_loss_tracker.result(), 
        "class_loss": self.class_loss_tracker.result()
      }

model = CustomModel()

model.compile(run_eagerly=True)

# steps_per_epoch = 1000
batches_per_epoch = len(dataset)

model.fit(
    dataset, 
    steps_per_epoch=batches_per_epoch,
    epochs=int(epochs*(len(dataset)/batches_per_epoch)),
    shuffle=False,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(logdir, "weights.{epoch:02d}"), verbose=1, save_weights_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=logdir),
        CustomCallback()
    ]
)