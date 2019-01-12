from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
import keras.backend as K
import pandas as pd
import matplotlib.pyplot as plt

def get_dataset():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pd.read_csv(url, names=names)
    return dataset

def get_model(input_size,output_size):
    K.clear_session()
    model = Sequential()
    
    model.add(Dense(32, input_shape=(input_size,), activation='relu'))
    
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(output_size,activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

df = get_dataset()

df.head()

X = df.drop(['class'],axis=1).values
num = {name:i for i,name in enumerate(df['class'].unique())}

y = [num[i] for i in df['class'].values]

y_cat = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X,y_cat,test_size=0.15)

model = get_model(X_train.shape[1],y_train.shape[1])

model.summary()

model.fit(X_train, y_train, 
          validation_split=0.3, 
          batch_size=16, 
          epochs=100, 
          verbose=1)

result = model.evaluate(X_test,y_test)

history = model.history

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print(result)