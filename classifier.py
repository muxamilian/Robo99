import tensorflow as tf
from datetime import datetime
import os
import matplotlib.pyplot as plt

from common import *

batch_size = 64

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
    train_ds, 
    epochs=10,
    shuffle=False,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(logdir, "weights.{epoch:02d}"), verbose=1, save_weights_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=logdir)
    ]
)