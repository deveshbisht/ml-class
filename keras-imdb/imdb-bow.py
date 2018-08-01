import imdb
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.preprocessing import text
import wandb
from sklearn.linear_model import LogisticRegression

wandb.init()
config = wandb.config
config.vocab_size = 1000

(X_train, y_train), (X_test, y_test) = imdb.load_imdb()


tokenizer = text.Tokenizer(num_words=config.vocab_size)
tokenizer.fit_on_texts(X_train)
#X_train = tokenizer.texts_to_matrix(X_train, mode= "tfidf")
#X_test = tokenizer.texts_to_matrix(X_test, mode= "tfidf")

X_train = tokenizer.texts_to_matrix(X_train )
X_test = tokenizer.texts_to_matrix(X_test )

bow_model = LogisticRegression()
#bow_model = Sequential()
bow_model.fit(X_train, y_train)
#img_width = X_train.shape[1]
#img_height = X_train.shape[2]
#bow_model.add (Flatten(input_shape=(img_width, img_height))
#bow_model.add (Dense(num_classes, activation='softmax'))
#bow_model.compile(loss='categorical_crossentropy', optimizer='adam',
#                metrics=['accuracy'])
               
# Fit the model
#bow_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test),
#                    labels=["A","b","C"])])
            #callbacks=[WandbCallback(data_type="image",    
pred_train = bow_model.predict(X_train)
acc = np.sum(pred_train==y_train)/len(pred_train)

pred_test = bow_model.predict(X_test)
val_acc = np.sum(pred_test==y_test)/len(pred_test)
wandb.log({"val_acc": val_acc, "acc": acc})