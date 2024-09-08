!pip install keras-preprocessing


from keras.utils import to_categorical
from keras_preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import os
import pandas as pd
import numpy as np

!pip install keras tensorflow keras-preprocessing


import os
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras_preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split


from google.colab import drive
drive.mount('/content/drive')


import zipfile

# Path to the archive
zip_path = '/content/drive/MyDrive/archive.zip'

# Unzip the archive to a directory
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('/content/dataset')


data_dir = '/content/dataset/images'  # The path to your dataset
categories = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
img_size = 48


import os
import numpy as np
from keras_preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical

def load_data_from_directory(data_dir, categories, img_size, subset):
    data = []
    labels = []
    subset_dir = os.path.join(data_dir, subset)
    for category in categories:
        path = os.path.join(subset_dir, category)
        class_num = categories.index(category)
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = load_img(img_path, target_size=(img_size, img_size))
                img_array = img_to_array(img)
                data.append(img_array)
                labels.append(class_num)
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")
    return np.array(data), np.array(labels)

# Load training data
train_data, train_labels = load_data_from_directory(data_dir, categories, img_size, 'train')

# Load test data
test_data, test_labels = load_data_from_directory(data_dir, categories, img_size, 'test')

# Normalize the images
train_data = train_data / 255.0
test_data = test_data / 255.0

# Convert labels to categorical
train_labels = to_categorical(train_labels, num_classes=len(categories))
test_labels = to_categorical(test_labels, num_classes=len(categories))


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(categories), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(train_data, train_labels, validation_split=0.2, epochs=25, batch_size=32)


loss, accuracy = model.evaluate(test_data, test_labels)
print(f'Test Accuracy: {accuracy}')


def predict_emotion(img_path):
    img = load_img(img_path, target_size=(img_size, img_size))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return categories[np.argmax(prediction)]

# Example usage with a new image in Google Drive
new_image_path = '/content/dataset/images/test/angry/10121.jpg'
print(predict_emotion(new_image_path))


from keras.models import load_model # type: ignore

# Assuming you've trained your model and stored it in a variable called 'model'
model.save('emotiondetectionmodel.h5')


from keras.models import load_model # type: ignore

# Load the saved model
loaded_model = load_model('emotiondetectionmodel.h5')