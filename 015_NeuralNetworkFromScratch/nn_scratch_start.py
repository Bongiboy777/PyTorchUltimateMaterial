
#%% packages
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#%% data prep
# source: https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
df = pd.read_csv('heart.csv')
df.head()

#%% separate independent / dependent features
X = np.array(df.loc[ :, df.columns != 'output'])
y = np.array(df['output'])

print(f"X: {X.shape}, y: {y.shape}")

#%% Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#%% scale the data
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

# input -> hidden -> activation -> pred


#%% network class
class network():
    
    def __init__(self, xtrain,ytrain, xtest, ytest):
        self.xtrain = xtrain
        self.xtest = xtest
        self.ytrain = ytrain
        self.ytest = ytest
        self.b = np.random.randn()
        self.w = np.random.randn(len(xtrain[0]))

        self.losses = []
        self.testLosses = []


        print(f'Init weights: {self.w}')
        print(f'Init bias: {self.b}')

        pass
    
    def hidden_1(self, xfeatures):
        hidden =  np.dot(xfeatures , self.w)+ self.b
        print(f'hidden1: {hidden}')
        return hidden
        

    def activation(self, x):
        activated = 1 / (1 + np.exp(-x))
        # print(f'activated: {activated}')
        return activated
    
    def activation_derivative(self, x):
        return (self.activation(x) * (1 - self.activation(x)))

    def foreward(self, xfeatures):
        hidden = self.hidden_1(xfeatures)
        prediction = self.activation(hidden)
        # print(f'prediction: {prediction}')
        return prediction
    

    def backward(self, xfeatures, diff):
        # dloss_dpred -> dpred_dactivation -> dactivated_hidden -> dhidden_input(wx + b)

        dL_dpred = 2 * (diff)
        hidden = self.hidden_1(xfeatures)

        dpred_dhidden = self.activation_derivative(hidden)
        dhidden_db = 1
        dhidden_dw = xfeatures


        dL_db = dhidden_db * dpred_dhidden * dL_dpred
        dL_dw = dhidden_dw * dpred_dhidden * dL_dpred

        return dL_dw, dL_db
        pass


    def iterate(self, xfeatures, ytruth, lr):
        prediction = self.foreward(xfeatures)
        diff = prediction - ytruth
        loss = diff ** 2
        
        self.losses.append(loss)

        dL_dw, dL_db = self.backward(xfeatures, diff)

        self.w = self.w - dL_dw * lr
        self.b =  self.b - dL_db * lr

        # print(f'diff: {loss}')

    def train(self, epochs, lr):
        for epoch in range(epochs):
            for index, sample in enumerate(self.xtrain):
                ytruth = self.ytrain[index]
                self.iterate(sample, ytruth, lr)
            
            testLoss = 0
            for i, t in enumerate(self.xtest):
                test_pred = self.foreward(t)
                test_truth = self.ytest[i]

                test_loss = (test_pred - test_truth) ** 2
                print(f'test loss: {test_loss}')
                testLoss += test_loss
            self.testLosses.append(test_loss)
lr = 0.01
epochs = 100

net = network(X_train_scale, y_train, X_test_scale, y_test)
net.train(epochs, lr)

# print(net.losses)
sns.lineplot(x=np.array(range(len(net.testLosses))), y=(net.testLosses))

# %% check losses

print(net.testLosses)
# %% iterate over test data

print(net.b)

print(net.w)

# %% Calculate Accuracy

# %% Baseline Classifier

# %% Confusion Matrix
total = X_test_scale.shape[0]
correct = 0
y_preds = []
for i in range(total):
    y_true = y_test[i]
    y_pred = np.round(net.foreward(X_test_scale[i]))
    y_preds.append(y_pred)
    correct += 1 if y_true == y_pred else 0
# %% Calculate Accuracy
acc = correct / total
f'{acc * 100:.2f}%'


# %%

# %%

# %% Baseline Classifier
from collections import Counter
Counter(y_test)
# %% Confusion Matrix
confusion_matrix(y_true = y_test, y_pred = y_preds)
# %%
