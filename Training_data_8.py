import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import matplotlib.pyplot as plt


df = pd.read_csv('data_set_for_training.csv')
img_size = (100, 100)
X = []

for file_path in df['file_name']:
    file_path= "Train_images\\" + file_path 
    img = imread(file_path, as_gray=True)
    img_resized = resize(img, img_size)
    X.append(img_resized.flatten())


X = np.array(X)
X_gender = pd.get_dummies(df['Gender'])
X_combined = np.hstack((X, X_gender))

X_train, X_test, y_train, y_test = train_test_split(X_combined, df['PHQ8_Binary'], test_size=0.2, random_state =42)


param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}

gnb = GaussianNB()
clf = GridSearchCV(gnb, param_grid=param_grid, cv=5)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy}')

with open('model.pickle', 'wb') as f:
    pickle.dump(clf, f)