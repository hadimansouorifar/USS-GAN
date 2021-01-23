import random
import cv2
import numpy as np
from PIL import Image
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import pickle

z_dim=100
x_train=[]
size=100
x_train = np.zeros((size, z_dim),dtype='float')
y_train = np.zeros((size),dtype='int')
counter = np.zeros((9),dtype='int')
model = ResNet50()
for i in range(0,size):
    noise = np.random.normal(0, 1, size=(1, z_dim))
    x_train[i]=noise
    # noise=noise*255
    im = np.reshape(noise, (10, 10))
    res = cv2.resize(im, dsize=(240, 240), interpolation=cv2.INTER_CUBIC)
    img = Image.fromarray(np.uint8((res) * 255))
    #img.show()
    img.save('ss' + str(i) + '.png')


    image = load_img('ss' + str(i) + '.png', target_size=(224, 224))
    x = np.expand_dims(image, axis=0)
    x = preprocess_input(x)
    print(x.shape)
    prediction = model.predict(x)

    # convert the probabilities to class labels
    label = decode_predictions(prediction)
    # retrieve the most likely result, e.g. highest probability
    label1 = label[0][0][1]
    label2 = label[0][1][1]
    label3 = label[0][2][1]

    if (label1=='mushroom') and (label2=='maze') and (label3=='dough'):
        y_train[i]=1
        print('1')
        counter[1]=counter[1]+1
    elif (label1=='mushroom') and (label2=='maze') and (label3=='agaric'):
        y_train[i] = 2
        print('2')
        counter[2]=counter[2]+1
    elif (label1=='mushroom') and (label2=='maze') and (label3=='pretzel'):
        y_train[i] = 2
        print('2')
        counter[2] = counter[2] + 1
    elif (label1=='mushroom') and (label2=='maze') and (label3=='coil'):
        y_train[i] = 2
        print('2')
        counter[2] = counter[2] + 1
    elif (label1=='mushroom') and (label2=='maze') and (label3=='sea_slug'):
        y_train[i] = 2
        print('2')
        counter[2] = counter[2] + 1
    elif (label1=='mushroom') and (label2=='maze') and (label3=='bubble'):
        y_train[i] = 2
        print('2')
        counter[2] = counter[2] + 1
    elif (label1=='mushroom') and (label2=='dough') and (label3=='maze'):
        y_train[i] = 2
        print('2')
        counter[2] = counter[2] + 1
    else:
        y_train[i] = 3
        print('3')
        counter[3]=counter[3]+1


    print(label1)
    print(label2)
    print(label3)

print(counter)
#clf=svm.SVC(kernel='rbf')
clf = GaussianNB()
clf.fit(x_train, y_train)
ss_model=clf

# Predict the response for test dataset
noise = np.random.normal(0, 1, size=(10, z_dim))
y_pred = clf.predict(x_train)
print(metrics.accuracy_score(y_train, y_pred))
y_pred = clf.predict(noise)
print(y_pred)

filename = 'SS-model'
outfile = open(filename,'wb')
pickle.dump(ss_model,outfile)
outfile.close()

infile = open(filename,'rb')
ss = pickle.load(infile)
infile.close()

y_pred = ss.predict(noise)
print(y_pred)