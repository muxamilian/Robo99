import subprocess
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
from datetime import datetime
import os
from common import *
import argparse
import cv2
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

dataset_name = 'dataset'
# dataset_name = 'dataset-capitals'
if dataset_name == 'dataset-capitals':
  num_chars = 26

noise_dim = 128
batch_size = 64

multiplier = 6

epochs = 15
lr = 1e-4

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--mode', default='train')      # option that takes a value

args = parser.parse_args()
print(args)

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

seed = tf.concat([tf.zeros([num_chars, noise_dim]), tf.one_hot(tf.range(0, num_chars), depth=num_chars)], axis=-1)

class CustomCallback(tf.keras.callbacks.Callback):

  def on_epoch_begin(self, epoch, logs):
    if epoch == 0:
      self.model.val_data = next(dataset.__iter__())[0]

      self.model.lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(lr, epochs*batches_per_epoch)

      # self.model.generator_optimizer = tf.keras.optimizers.legacy.SGD(self.model.lr_decayed_fn)
      # self.model.discriminator_optimizer = tf.keras.optimizers.legacy.SGD(self.model.lr_decayed_fn)
      # self.model.generator_optimizer = tf.keras.optimizers.legacy.Adam(self.model.lr_decayed_fn)#, beta_1=0.0)
      # self.model.discriminator_optimizer = tf.keras.optimizers.legacy.Adam(self.model.lr_decayed_fn)#, beta_1=0.0)
      self.model.generator_optimizer = tf.keras.optimizers.legacy.RMSprop(self.model.lr_decayed_fn)
      self.model.discriminator_optimizer = tf.keras.optimizers.legacy.RMSprop(self.model.lr_decayed_fn)

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
    # print(self.generator.summary())
    self.discriminator = make_discriminator_model()
    # print(self.discriminator.summary())
    self.classifier_model = model
    # self.classifier_model.load_weights('logs/20230704-173202/weights.47')
    # self.classifier_model.load_weights('logs/20230721-120604/weights.29')
    # self.classifier_model.load_weights('logs/20230723-175517/weights.23')
    self.classifier_model.load_weights('logs/20230723-175517/weights.26')
    # print(self.classifier_model.summary())

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

def analyze(raw_imgs, out_path, font_name):
  accuracies = []
  big_square = np.zeros([64*8, 64*8], dtype=np.uint8)
  for j, img in enumerate(raw_imgs):
    big_square[(j//8)*64:((j//8)+1)*64, (j%8)*64:((j%8)+1)*64] = np.squeeze(img.numpy().astype(np.uint8))
  for _ in range(100):
    imgs, _ = list(zip(*[augment(img, tf.zeros([])) for img in raw_imgs]))
    imgs = tf.stack(imgs, axis=0)
    predictions = tf.math.argmax(model.classifier_model(imgs, training=False), axis=-1, output_type=tf.int32)
    tf.Assert(predictions.shape[0] == imgs.shape[0], [f'{len(predictions.shape)}, {len(imgs)}'])
    accuracy = tf.reduce_mean(tf.cast(tf.range(len(imgs)) == predictions, dtype=tf.float32))
    accuracies.append(accuracy.numpy().item())
  accuracy = np.mean(accuracies)
  cv2.imwrite(f'{out_path}/{font_name}.png', big_square)
  return accuracy

if args.mode == 'train':
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
elif args.mode == 'export':
    # Current max accuracy is 0.9861290323734283
    # weights.01 0.974354835152626
    # weights.02 0.9675806391239167
    # weights.03 0.9683870869874954
    # weights.04 0.973387091755867
    # weights.05 0.9633870851993561
    # weights.06 0.9817741948366165
    # weights.07 0.9861290323734283
    # weights.08 0.979354836344719
    # weights.09 0.979838707447052
    # weights.10 0.9772580629587173
    # weights.11 0.9820967727899551
    # weights.12 0.979032256603241
    # weights.13 0.9829032278060913
    # weights.14 0.978225804567337
    # weights.15 0.9759677386283875
    path = 'gan_logs/20230728-142502/'
    all_weights = [item[:-len('.index')] for item in os.listdir(path) if '.index' in item]
    out_dir = path.split('/')[-1]
    out_path = 'out_imgs/' + out_dir
    os.makedirs(out_path, exist_ok=True)
    for i, weight_file_name in tqdm(enumerate(all_weights)): 
      if 'weights.07' not in weight_file_name:
        continue
      print(f'{weight_file_name=}')
      p = subprocess.Popen(f'ffmpeg -y -f image2pipe -vcodec png -r 20 -i - -f apng -plays 0 -r 20 {out_path}/{weight_file_name}.png'.split(' '), stdin=subprocess.PIPE)
      for tradeoff in tqdm(np.linspace(0., 1., 50)):
        print(f'{tradeoff=}')
        model.load_weights(path + weight_file_name)
        out_imgs = model.generator(tf.concat([seed, tf.tile(tf.reshape(tradeoff.item(), [1, 1]), (seed.shape[0], 1))], axis=-1), training=False)
        out_imgs = tf.squeeze(out_imgs)
        os.makedirs(f'{out_path}/{weight_file_name}', exist_ok=True)
        raw_imgs = []
        for j, item in enumerate(tf.unstack(out_imgs, axis=0)):
          converted = 255-np.round((item.numpy()+1.)/2.*255).astype(np.uint8)
          cv2.imwrite(f'{out_path}/{weight_file_name}/{j}.png', converted)
          raw_imgs.append(tf.reshape(tf.constant(converted, dtype=tf.float32), (64, 64, 1)))
        analyze(raw_imgs, out_path, f'{weight_file_name}_{tradeoff}')
        im = PIL.Image.open(f'{out_path}/{weight_file_name}_{tradeoff}.png')
        im.save(p.stdin, 'PNG')
    p.stdin.close()
    p.wait()
elif args.mode == 'classify_fonts':
  out_file_name = 'font_accuracies.txt'
  all_accuracies = []
  current_max = 0
  if os.path.isfile(out_file_name):
    with open(out_file_name) as fr:
      all_accuracies = [float(item) for item in [item.strip() for item in fr.read().split('\n')] if len(item) > 0]
    if len(all_accuracies) > 0:
      current_max = max(all_accuracies)
  with open(out_file_name, 'a', buffering=1) as fw:
    fonts_to_analyze = num_fonts
    all_fonts = list(range(num_fonts))
    random.seed(0)
    random.shuffle(all_fonts)
    for font_index in tqdm(range(fonts_to_analyze)):
      if font_index < len(all_accuracies):
        print(f'Skipping {font_index}')
        continue
      raw_imgs = []
      for char_index in range(num_chars):
        read_img = cv2.imread(f'dataset/{str(char_index).zfill(2)}/{all_fonts[font_index]}.png')
        assert np.all(read_img[:,:,0] == read_img[:,:,1]) and np.all(read_img[:,:,0] == read_img[:,:,2])
        # print(f'{read_img.shape=}')
        raw_img = tf.reshape(tf.constant(read_img[:,:,0], dtype=tf.float32), (64, 64, 1))
        raw_imgs.append(raw_img)
      font_name = str(font_index).zfill(2)
      accuracy = analyze(raw_imgs, f'out_imgs/', font_name)
      current_max = max(current_max, accuracy)
      print(f'{font_name=}, {accuracy=}, {current_max=}')
      all_accuracies.append(accuracy)
      fw.write(f'{accuracy}\n')
  plt.figure()
  counts, bins = np.histogram(all_accuracies, 40)
  counts = counts/np.sum(counts)
  print(f'{np.max(all_accuracies)=}, {np.median(all_accuracies)=}, {np.mean(all_accuracies)=}')
  print('counts', counts, 'bins', bins)
  plt.stairs(counts, bins, fill=True)
  plt.yticks([], [])
  plt.tight_layout()
  plt.savefig('histogram.pdf')
