import pickle
import matplotlib.pyplot as plt
import numpy as np

import sklearn.linear_model

with open('logistic-regression-data.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)

model = sklearn.linear_model.LogisticRegression()

model.fit(data['training_x'], data['training_y'])

score = model.score(data['validation_x'], data['validation_y'])
print(score)

proba_val = np.array(model.predict_proba(data['validation_x']))
proba_train = np.array(model.predict_proba(data['training_x']))

print(proba_train[:, 1])

proba_train_true = []
proba_train_false = []
for y, pred in zip(data['training_y'], proba_train[:, 1]):
    if y == 1:
        proba_train_true += [pred]
    else:
        proba_train_false += [pred]
proba_val_true = []
proba_val_false = []
for y, pred in zip(data['validation_y'], proba_val[:, 1]):
    if y == 1:
        proba_val_true += [pred]
    else:
        proba_val_false += [pred]

plt.figure()
plt.title('Training Data')
plt.hist(proba_train_true, rwidth=0.5)
plt.hist(proba_train_false, rwidth=0.5)
plt.legend(['True', 'False'])

plt.figure()
plt.title('Validation Data')
plt.hist(proba_val_true, rwidth=0.5)
plt.hist(proba_val_false, rwidth=0.5)
plt.legend(['True', 'False'])

plt.show()
