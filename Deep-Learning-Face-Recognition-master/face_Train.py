from keras.layers import Input, Lambda, Dense, Flatten, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

train_path = '/content/drive/MyDrive/Facy Team/Resource/Images/Images - Deep Learning/800-200 Images/Train/'
test_path = '/content/drive/MyDrive/Facy Team/Resource/Images/Images - Deep Learning/800-200 Images/Test/'

# add preprocessing layer to the front of VGG
vgg = VGG16(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
  layer.trainable = False
  
def new(bottom_model, num_classs):
  top_model = bottom_model.output
  #top_model = GlobalAveragePooling2D()(top_model)
  # add a fully-connected layer
  top_model = Flatten()(top_model)
  #top_model = Dense(512, activation='relu')(top_model)
  #top_model = Dropout(0.25)(top_model)
  #top_model = Dense(256, activation='relu')(top_model)
  #top_model = Dropout(0.25)(top_model)
  top_model = Dense(num_classs , activation='softmax')(top_model)
  return top_model

#FCLayer = new(vgg, len(folders))
#model = Model(inputs=vgg.input, outputs=FCLayer)
  
  # useful for getting number of classes
folders = glob(train_path + '*')

# create a model object
x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   brightness_range = [0.5, 1.5],
                                   rotation_range = 5,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

# fit the model
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs = 5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
  )

#print(r.history)
# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

model.save('model.h5')