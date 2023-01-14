import cv2 # For capturing video and working on it
import pandas as pd # For handling data
import numpy as np # For mathematical processing
import matplotlib.pyplot as plt # For Plotting and working on Graphs
import keras # High level API fot building model
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from tensorflow.keras.utils import to_categorical


# Loading Datasets
train=pd.read_csv(r'handsign\sign_mnist_train.csv')
test=pd.read_csv(r'handsign\sign_mnist_test.csv')

# Dividing Datasets into input and output data
x_train=train.iloc[:,1:]
y_train_=train.iloc[:,0]
x_test=test.iloc[:,1:]
y_test=test.iloc[:,0]

# # Converting Data into Categorical Data
y_train=to_categorical(y_train_)
# y_test=to_categorical(y_test)

# Reshaping Dataset into necessary shape
x_train = x_train.values.reshape(-1,28,28,1)
x_test = x_test.values.reshape(-1,28,28,1)

# # plotting graphs
# f,ax=plt.subplots(2,5)
# f.set_size_inches(10,10)
# k=0
# for i in range(2):
#     for j in range(5):
#         ax[i,j].imshow(x_train[k].reshape(28,28),cmap='gray')
#         k=k+1
#     plt.tight_layout()

# plt.figure(2)
# plt.bar(y_train_.value_counts().index,y_train_.value_counts().values)
# plt.xlabel('label')
# plt.ylabel('count')
# plt.title('Count-Label')
# plt.show()

# Normalization
x_train=x_train/255
x_test=x_test/255

# Learning Rate
learning_rate=keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.1,patience=2,min_lr=0.0001)


#Model Building
model = keras.models.Sequential()
model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Flatten())
model.add(Dense(units = 512 , activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(units = 25 , activation = 'softmax'))
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])


model.summary()


model.fit(x_train,y_train,epochs = 8,batch_size=128,callbacks=[learning_rate])

pred=model.predict(x_test)
pred=np.argmax(pred, axis=1)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
print(accuracy_score(y_test,pred)*100)

# Capturing Video
cap=cv2.VideoCapture(0)
letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
while True:
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    roi=frame[0:300,0:300]
    cv2.rectangle(frame,(0,0),(300,300),(0,255,0),2)
    resized=cv2.resize(roi,(28,28),interpolation=cv2.INTER_AREA)
    resized=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
    dadada=model.predict_classes(np.expand_dims(np.expand_dims(resized,axis=2),axis=0))[0]
    cv2.putText(frame,letters[dadada],(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,122),1)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',resized)
    if(cv2.waitKey(1) & 0xFF==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
