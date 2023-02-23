import keras,os, cv2
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.layers import Dense,Activation,Flatten,Dropout,Conv2D,MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

#========================================Preprocessing=============================================
data_path='dataset1'
categories=os.listdir(data_path)
labels=[i for i in range(len(categories))]

label_dict=dict(zip(categories,labels)) #empty dictionary

print(label_dict)
print(categories)
print(labels)

img_size = 224
data = []
target = []

for category in categories:
    folder_path = os.path.join(data_path, category)
    img_names = os.listdir(folder_path)

    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Coverting the image into gray scale
            resized = cv2.resize(gray, (img_size, img_size))
            # resizing the gray scale into 100x100, since we need a fixed common size for all the images in the dataset
            img_median = cv2.medianBlur(resized, 9)  # Add median filter to image
            data.append(img_median)
            target.append(label_dict[category])

            #cv2.imshow("image", img_median)

            #cv2.waitKey(0)
            #print(len(data))
            # appending the image and the label(categorized) into the list (dataset)

        except Exception as e:
            print('Exception:', e)
            # if any exception rasied, the exception will be printed here. And pass to the next image

#========================================End Preprocessing=============================================

#===============================Recale and assign catagorical lables====================================
import numpy as np
data=np.array(data)/255.0
data=np.reshape(data,(data.shape[0],img_size,img_size,1))
target=np.array(target)
from keras.utils import np_utils
new_target=np_utils.to_categorical(target)

np.save('images.npy',data)
np.save('lables.npy',new_target)


data=np.load('images.npy')
new_target=np.load('lables.npy')
#========================================Create The Model=============================================
model = Sequential()

model.add(Conv2D(input_shape=data.shape[1:],filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=2, activation="softmax"))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

#===========================================Model Compile===========================================
from tensorflow.keras.optimizers import Adam
opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['acc'])

######## Splittiong data into traning and testing
from sklearn.model_selection import train_test_split
train_data,test_data,train_target,test_target=train_test_split(data,new_target,test_size=0.1)

########
from keras.callbacks import ModelCheckpoint, EarlyStopping
#ModelCheckpoint helps us to save the model by monitoring a specific parameter of the model.
checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')

#EarlyStopping helps us to stop the training of the model early if there is no increase
#early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
#pass train and test data to fit_generator.
history=model.fit(train_data,train_target,epochs=1,callbacks=[checkpoint],validation_split=0.2)


#==========================================visualise The Result====================================

import matplotlib.pyplot as plt
N = 1

plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Acc")
plt.legend(loc="center right")
plt.savefig("CNN_Model")
plt.show()






