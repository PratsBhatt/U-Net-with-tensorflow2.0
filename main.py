import tensorflow as tf
import tensorflow_datasets as tfds
import unet_model
from plotImage import display

# Hyperparameters
BATCH_SIZE = 8
BUFFER_SIZE = 1000
EPOCHS = 20

# First step is to load the dataset
# Disable the progress bar.
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize_with_pad(datapoint['image'], 572, 572)
    input_mask = tf.image.resize_with_pad(datapoint['segmentation_mask'], 388, 388)

    # Randomly choosing the images to flip right left.
    # We need to split both the input image and the input mask as the mask is in correspondence to the input image.
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    # Normalizing the input image.
    input_image = tf.cast(input_image, tf.float32) / 255.0

    # Returning the input_image and the input_mask
    return input_image, input_mask

# We do not need to do any image augmentation here for the validation dataset.
def load_image_test(datapoint):
    input_image = tf.image.resize_with_pad(datapoint['image'], 572, 572)
    input_mask = tf.image.resize_with_pad(datapoint['segmentation_mask'], 388, 388)

    # Normalizing the input image.
    input_image = tf.cast(input_image, tf.float32) / 255.0

    return input_image, input_mask


# Transform the dataset using the map function.
train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)


# Shuffle, batch, cache and prefetch the dataset.
train_dataset = train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).cache()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)


# Check out how does the training set look like.
for image, mask in train.take(1):
  sample_image, sample_mask = image, mask
  display([sample_image, sample_mask])

# Create the model.
model = unet_model.unet_model()
# Build the model with the input shape
# Image is RGB, so here the input channel is 3.
model.build(input_shape=(None, 3, 572, 572))
model.summary()

# Write model saving callback.
model_save_callback = tf.keras.callbacks.ModelCheckpoint(
    './model_checkpoint', monitor='val_loss', verbose=0, save_best_only=True,
    save_weights_only=False, mode='auto', save_freq='epoch')

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model_history = model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset, callbacks=[model_save_callback])
