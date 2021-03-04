# sys.path.append("/Users/tamarervin/CH_Project/CHD")
import numpy as np
import h5py as h5
import numpy.random as random
import matplotlib.pyplot as plt

import tensorflow as tf
import analysis.ml_analysis.ch_detect.ml_functions as ml_funs
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#### ---- parameters ---- ####
# test size of dataset
test_size = 0.33
BATCH_SIZE = 32
BUFFER_SIZE = 16
EPOCHS = 50

# h5 file name
train_h5 = '/Volumes/CHD_DB/map_data_small.h5'
test_h5 = 'h5_datasets/data_test.h5'
model_h5 = 'map_unet_model.h5'

### metrics
# Image size that we are going to use
IMG_SIZE = 128
# Our images are RGB (3 channels)
N_CHANNELS = 3
# Number of classes
N_CLASSES = 2

#### ---- dealing with dataset ---- ####
### split data into train and test set
# read hdf5 file
hf = h5.File(train_h5, 'r')

# split data into training and test set
train, test = train_test_split(list(hf.keys()), test_size=test_size, random_state=0)

# create list of training images and masks
train_image = []
train_mask = []
for date in train:
    g = hf.get(date)
    train_image.append(np.array(g['euv_image']))
    train_mask.append(np.array(g['chd_data']))

# create list of validation images and masks
val_image = []
val_mask = []
for date in test:
    g = hf.get(date)
    val_image.append(np.array(g['euv_image']))
    val_mask.append(np.array(g['chd_data']))

hf.close()

### create tensorflow dataset
# convert to dictionaries
train_dict = {key: value for key, value in zip(["image"], [train_image])}
train_dict['segmentation_mask'] = train_mask
val_dict = {key: value for key, value in zip(["image"], [val_image])}
val_dict['segmentation_mask'] = val_mask

del train_image
del train_mask
del val_image
del val_mask

# create tensorflow datasets
train_dataset = tf.data.Dataset.from_tensor_slices(train_dict)
val_dataset = tf.data.Dataset.from_tensor_slices(val_dict)

# map the dataset using the ML functions
train = train_dataset.map(ml_funs.load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val = val_dataset.map(ml_funs.load_image_val)

TRAIN_LENGTH = len(train_dataset)
VAL_LENGTH = len(val_dataset)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_dataset = val.batch(BATCH_SIZE)


#### ---- training model ---- ####
# convolution function
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',
               padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


# U-Net definition
def get_unet(input_img, n_filters=16, dropout=0.1, batchnorm=True):
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


# compile model
input_image = Input((IMG_SIZE, IMG_SIZE, N_CHANNELS), name='img')
model = get_unet(input_image, n_filters=16, dropout=0.05, batchnorm=True)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

model.summary()

# callbacks
callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint(model_h5, verbose=1, save_best_only=True, save_weights_only=True)
]

# fit model
results = model.fit(train_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks,
                    validation_data=val_dataset)

# plot loss curve
plt.figure(figsize=(8, 8))
plt.title("Learning Curve")
plt.plot(results.history["loss"], label="Training Loss", color='orchid')
plt.plot(results.history["val_loss"], label="Validation Loss", color='lightblue')
plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r",
         label="Best Model")
plt.xlabel("Epochs")
plt.ylabel("Log Loss")
plt.legend()

#### ---- test the model ---- ####
# load the best model
model.load_weights(model_h5)


# plotting function
def plot_sample(X, y, preds, binary_preds, ix=None, title=None):
    """Function to plot the results"""
    if ix is None:
        ix = random.randint(0, len(X))

    fig, ax = plt.subplots(1, 3, figsize=(20, 10))

    ax[0].imshow(X[ix].squeeze(), vmin=0, vmax=1)
    ax[0].set_title('EUV Image')

    ax[1].imshow(y[ix].squeeze())
    ax[1].set_title('CHD')

    use_chd = np.logical_and(y[ix].squeeze() > 0, preds[ix].squeeze() > 0)
    pred = np.zeros(shape=preds[ix].squeeze().shape)
    pred[use_chd] = preds[ix].squeeze()[use_chd]
    ax[2].imshow(pred)
    ax[2].set_title('CHD Predicted')

    for a in ax.flat:
        a.set(xlabel="x (solar radii)", ylabel="y (solar radii)")
        a.set_xticks([])
        a.set_yticks([])
    # use_chd = np.logical_and(y[ix].squeeze() > 0, binary_preds[ix].squeeze() > 0)
    # pred_binary = np.zeros(shape=binary_preds[ix].squeeze().shape)
    # pred_binary[use_chd] = binary_preds[ix].squeeze()[use_chd]
    # ax[3].imshow(pred_binary, vmin=0, vmax=1)
    # # if has_mask:
    # # ax[3].contour(binary_preds[ix].squeeze(), colors='k', levels=[0.25])
    # ax[3].set_title('CHD Predicted Binary')

    if title is not None:
        fig.suptitle(title)

    return None


### training data
hf = h5.File(train_h5, 'r')

# split data into training and test set
train, test = train_test_split(list(hf.keys()), test_size=test_size, random_state=0)

# create list of training images and masks
train_image = []
train_mask = []
for date in train:
    g = hf.get(date)
    train_image.append(np.array(g['euv_image'])[tf.newaxis, ...])
    train_mask.append(np.array(g['chd_data'])[tf.newaxis, ...])

hf.close()

# convert to dictionaries
train_dict = {key: value for key, value in zip(["image"], [train_image])}
train_dict['segmentation_mask'] = train_mask

del train_image
del train_mask

# load the best model
model.load_weights(model_h5)

# prediction
X_train = train_dict['image']
y_train = train_dict['segmentation_mask']
preds_train = model.predict(X_train, verbose=1)

# binary prediction
preds_train_t = (preds_train > 0.01).astype(np.uint8)

# plot prediction with training data
plot_sample(X_train, y_train, preds_train_t, preds_train_t, ix=0, title="Model Prediction with Training Data")

### validation data
# read hdf5 file
hf = h5.File(train_h5, 'r')

# split data into training and test set
train, test = train_test_split(list(hf.keys()), test_size=test_size, random_state=0)

# create list of validation images and masks
val_image = []
val_mask = []
for date in test:
    g = hf.get(date)
    val_image.append(np.array(g['euv_image'])[tf.newaxis, ...])
    val_mask.append(np.array(g['chd_data'])[tf.newaxis, ...])

hf.close()

val_dict = {key: value for key, value in zip(["image"], [val_image])}
val_dict['segmentation_mask'] = val_mask

del val_image
del val_mask

# validation data prediction
X_valid = val_dict['image']
y_valid = val_dict['segmentation_mask']
preds_val = model.predict(X_valid, verbose=1)

# binary prediction
preds_val_t = (preds_val > 0.01).astype(np.uint8)

plot_sample(X_valid, y_valid, preds_val_t, preds_val_t, ix=0, title="Model Prediction with Validation Data")

#### ---- check on new data!! ---- ####
# load new data
hf = h5.File(test_h5, 'r')
print("Loading HDF5 dataset")

image = []
mask = []
for date in list(hf.keys()):
    g = hf.get(date)
    image.append(np.array(g['euv_image'])[tf.newaxis, ...])
    mask.append(np.array(g['chd_data'])[tf.newaxis, ...])

hf.close()

# convert to dictionaries
data_dict = {key: value for key, value in zip(["image"], [image])}
data_dict['segmentation_mask'] = mask

del image
del mask

# use model to predict
X_test = data_dict['image']
y_test = data_dict['segmentation_mask']
preds_test = model.predict(X_test, verbose=1)
preds_test_t = (preds_test > 0.01).astype(np.uint8)

# plot images
plot_sample(X_test, y_test, preds_test_t, preds_test_t, ix=0, title="Model Prediction with Test Data")
