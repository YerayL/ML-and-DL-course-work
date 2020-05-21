from keras import layers, models

def create_embedding_model(vocab_size, max_length):
    model=models.Sequential()
    model.add(layers.Embedding(vocab_size, 100, input_length=max_length))
    model.add(layers.Conv1D(32, 8, activation="relu"))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation="relu"))
    model.add(layers.Dense(1,  activation="sigmoid"))
    return model

