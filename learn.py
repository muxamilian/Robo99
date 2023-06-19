import h5py
import random
import tensorflow as tf
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np

random.seed(0)
tf.random.set_seed(0)

batch_size = 64
fonts = h5py.File('fonts.hdf5')['fonts']

num_fonts = fonts.shape[0]
num_chars = fonts.shape[1]
image_shape = fonts.shape[2:]
input_shape = (*image_shape, 3)

@tf.function
def augment(image, label):
    # images = tf.tile(images/255.*2.-1, (1, 1, 1, 3))
    # tf.assert_less(images, 1.)
    # tf.assert_greater(images, -1.)
    # for image in tf.unstack(images):
    image = tf.tile(image/255.*2.-1, (1, 1, 3))
    tf.debugging.assert_less_equal(image, 1.)
    tf.debugging.assert_greater_equal(image, -1.)
    with tf.device('/cpu:0'):
        new_res = tf.random.uniform((), minval=2,maxval=image_shape[0]+1, dtype=tf.int32)
        random_vector_downscaled = tf.random.normal((1,new_res,new_res,1), stddev=0.5)
        random_vector = tf.random.normal((1,*image_shape,1), stddev=0.5)
    # Because tf.random is broken on Apple devices on the GPU
    image = tf.expand_dims(image, 0)
    image = tf.image.resize(image, (new_res,new_res))
    image += random_vector_downscaled
    image = tf.image.resize(image, image_shape)
    image += random_vector
    image = tf.squeeze(image, 0)
    # images = tf.stack(images)
    image = tf.clip_by_value(image, -1., 1.)

    return image, label

# def generator():
#     while True:
#         all_combinations = list(range(num_fonts * num_chars))
#         random.shuffle(all_combinations)
#         for batch_index in range(len(all_combinations)//batch_size):
#             batch = []
#             labels = []
#             for i in range(batch_size):
#                 current_index = batch_index*batch_size + i
#                 current_combination = all_combinations[current_index]
#                 font_index = current_combination // num_chars
#                 label = current_combination % num_chars
#                 current_font = tf.tile(tf.expand_dims(tf.constant(fonts[font_index,label,:,:]/255.*2.-1, dtype=tf.float32), -1), (1, 1, 3))
#                 current_font = augment(current_font)
#                 batch.append(current_font)
#                 labels.append(label)
    
#             result = tf.stack(batch)
#             assert tf.reduce_max(result) <= 1.
#             assert tf.reduce_min(result) >= -1.
#             yield result, tf.constant(labels, dtype=tf.float32)

validation_split = 0.1
datasets = tf.keras.utils.image_dataset_from_directory(
    'dataset',
    validation_split=validation_split,
    seed=0,
    subset='both',
    shuffle=True,
    batch_size=None,
    color_mode='grayscale',
    image_size=image_shape                                            
)

datasets = [dataset.map(augment).batch(batch_size, drop_remainder=True) for dataset in datasets]

train_ds, val_ds = datasets

model = tf.keras.applications.MobileNetV3Large(
    input_shape=input_shape,
    classes=num_chars,
    classifier_activation=None,
    weights=None,
    include_preprocessing=False
)

logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
model.compile(
    # optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.01),
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
    )

print(model.summary())

# data, labels = next(generator())
first_data = next(val_ds.__iter__())
# print('first_data', first_data)
data, labels = first_data
plt.figure(figsize=(8,8))
for i in range(min(64,batch_size)):
    plt.subplot(8,8,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(data[i].numpy()/2+0.5)
    plt.xlabel(str(labels[i].numpy().item()))
plt.tight_layout()
# plt.show()

model.fit(
    # generator(),
    train_ds, 
    epochs=10,
    shuffle=False,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(logdir, "weights.{epoch:02d}"), verbose=1, save_weights_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=logdir)
    ]
)