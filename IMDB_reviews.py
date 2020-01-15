import tensorflow as tf 
from tensorflow import keras 
import numpy as np 

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words= 10000)

word_index = data.get_word_index()
word_index = {k:(v + 3) for k,v in word_index.items()}
word_index['<PAD>'] = 0   # pour gérer la longueur des reviews (faire en sorte que ca soit la même longueur)
word_index['<START>'] = 1
word_index['<UNK>'] = 2   # pour les valeurs UNKNOWN
word_index['<UNUSED>'] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Gestion de la longueur des reviews (pour les avoir identiques) / Pads sequences to the same length.
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value = word_index['<PAD>'], padding= 'post', maxlen= 250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value = word_index['<PAD>'], padding= 'post', maxlen= 250)

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])  # si le mot est inconnu on le transforme en "?"

# model :  Group similar words (différence d'angles sur les vecteurs)> ici on construit 10.000 vecteurs de mots
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))   # 16 : dimensions. Convertir l'input en vecteur (ex: [ word n°102] ==> [0.2, 0.9, 0.3.. x16])
model.add(keras.layers.GlobalAveragePooling1D()) # Réalise la moyenne de vecteurs 
model.add(keras.layers.Dense(16, activation= "relu")) 
model.add(keras.layers.Dense(1, activation= "sigmoid"))  #reviews good or bad

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_train = train_data[10000:]
y_train = train_labels[10000:]

x_val = train_data[:10000]  # Validation data (10000 reviews on 25 000)
y_val = train_labels[:10000]  # Validation data (10000 reviews on 25 000)

fitModel = model.fit(x_train, y_train, batch_size=512, epochs= 40, validation_data=(x_val,y_val), verbose=1)
results = model.evaluate(test_data, test_labels)

#prediction
test_review = test_data[0]
prediction= model.predict([test_review]) 
print ("Review: ")
print (decode_review(test_review))
print ("Prediction: " + str(prediction[0]))
print ("Actual: " + str(test_labels[0]))

#Enregistrer le modèle
model.save("model.h5")  # h5 est une extension

#model = keras.models.load_model("model.h5")


#-----------------------------------------------------------------
#print (len(train_data[0]))
#print (decode_review(train_data[0]))
#word_index = {k:(v + 3) for k,v in word_index.items()}
#print (len(word_index))
#print (len(train_data[0]))
