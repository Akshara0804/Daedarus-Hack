# Daedarus-Hack
**COVID-19 Face mask detection system**

**Abstract**</br>
In today’s times wearing a face mask has become compulsory to protect ourselves from COVID-19. We tried to develop a face mask detection system integrated with a CCTV surveillance camera which identifies a human face and detects whether the person is wearing the mask properly, wearing the mask improperly or not wearing a mask and displays the message on the screen and sends a notification to the administrator if he is improperly wearing the mask or not wearing the mask. 

**Novelty**</br>
1)Can be utilised to keep a watch on people entering a particular place like banks, schools, malls etc without employing human work force and exposing them to infection.
2)The output of the face detection system is integrated with an application programming interface to send the message to the administrator.

**Technology Stack**</br>
Platform: Jupyter notebook
Image processing and management tech stack:
       a) Open CV
       b) Numpy
       c) Matplotlib
       d) Scikit image
API: Sinch
Programming language: Python

**Implementation**</br>
The machine learning algorithm used in this application is support vector machine algorithm and for face detection Voila Jones object detection framework(Haar-cascade image classifier) is used.
First we create the data set to be trained the load the data. Then we train it through SVM algorithm and integrate the face mask detection system with Sinch API. Once we run the application, when the face mask detector detects either no mask or improper mask it sends a signal to the API which in-turn sends a notification to the administrator.

**Business Scope**</br>
Can be implemented in areas such as schools, banks, malls, offices, hospitals, restaurants, airports, railway station etc. where people  go in their everyday life.


**Team members**</br>
1)Akshara Ganeshram</br>
2)Umang Jain</br>
3)Mahima Rajapriyar</br>









