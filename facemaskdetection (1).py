#import the required modules
import cv2
import clx.xms
import requests
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import learning_curve

#getting the data and saving it
haar_data= cv2.CascadeClassifier(r'C:\Users\aksha\OneDrive\Desktop\face mask detection\haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0)
counter=0
data=[]
while True:
    flag, img= capture.read()
    if flag:
        faces=haar_data.detectMultiScale(img)
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
        face= img[y:y+h, x:x+w, :]
        face=cv2.resize(face, (50,50))
        print(len(data))
        if len(data)<400:
            data.append(face)
    cv2.imshow('result',img)
    if cv2.waitKey(2)==27 or len(data)>= 200:
        break
capture.release()
cv2.destroyAllWindows()
np.save('without_mask.npy', data)
np.save('proper_mask.npy', data)
np.save('improper_mask.npy', data)

#Loading the data
proper_mask=np.load('proper_mask.npy')
without_mask=np.load('without_mask.npy')
improper_mask=np.load('improper_mask.npy')

#Converting data into 2D
proper_mask=proper_mask.reshape(200, 50*50*3)
without_mask=without_mask.reshape(200, 50*50*3)
improper_mask=improper_mask.reshape(200, 50*50*3)

#Preparing the training data
X= np.r_[proper_mask, without_mask, improper_mask]
labels= np.zeros(X.shape[0])
labels[200:400]= 1.0
labels[400:]= 2.0

#Reducing the dimensions of training data
x_train, x_test, y_train, y_test = train_test_split(X,labels, test_size=0.25)
pca = PCA(n_components=3)
x_train=pca.fit_transform(x_train)

#training and testing the data
svm=SVC()
svm.fit(x_train, y_train)
x_test=pca.transform(x_test)
y_pred=svm.predict(x_test)
accuracy_score(y_test, y_pred)

#Code for face detection and sending notification to individual
haar_data= cv2.CascadeClassifier(r'C:\Users\aksha\OneDrive\Desktop\face mask detection\haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0)
data=[]
counter=0
font= cv2.FONT_HERSHEY_COMPLEX
client = clx.xms.Client(service_plan_id='ff90c71a89d247f29ed6d673994e1d63',token='7d1fea48c6974453a319819895f4b6db')
create = clx.xms.api.MtBatchTextSmsCreate()
create.sender = '447537455170'
create.recipients = {'918072984663'}
while True:
    flag, img= capture.read()
    if flag:
        faces=haar_data.detectMultiScale(img)
    for x,y,w,h in faces:
        if(x,y,w,h.any() and counter ==0):
                try:
                   batch = client.create_batch(create)
                except (requests.exceptions.RequestException, clx.xms.exceptions.ApiException) as ex:
                    print('Failed to communicate with XMS: %s' % str(ex))
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
        face= img[y:y+h, x:x+w, :]
        face=cv2.resize(face, (50,50))
        face=face.reshape(1,-1)
        face=pca.transform(face)
        pred=svm.predict(face)
        n=names[int(pred)]
        cv2.putText(img,n,(x,y),font,1,(0,255,0),2)
        print(n)
if n==1:
    create.body('no mask')
if n==2:
    create.body('improper mask')
        
    cv2.imshow('result',img)
    if cv2.waitKey(2)==27:
        break
capture.release()
cv2.destroyAllWindows()

#Code for plotting accuracy curve
Face = np.r_[proper_mask, without_mask, improper_mask]
X = Face.data
train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(), X, labels, cv=2, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
print(train_mean)
plt.subplots(1, figsize=(10,10))
plt.plot(train_sizes, train_mean, color="black",  label="Training set")
plt.plot(train_sizes, test_mean,'--', color="blue", label="Test set")

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

plt.title("Learning Curve")
plt.xlabel("training set size(cases)"), plt.ylabel("Accuracy"), plt.legend(loc="best")
plt.tight_layout()
plt.show()
