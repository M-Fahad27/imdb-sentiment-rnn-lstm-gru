from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence


max_features = 10000
max_len = 500

(xtrain, ytrain), (xtest, ytest) = imdb.load_data(num_words=max_features)

xtrain = sequence.pad_sequences(xtrain, maxlen=max_len)
xtest = sequence.pad_sequences(xtest, maxlen=max_len)


def build_model(model_type):
    model = models.Sequential()

    model.add(layers.Embedding(max_features, 32, input_length=max_len))

    if model_type == "RNN":
        model.add(layers.SimpleRNN(32))
    elif model_type == "LSTM":
        model.add(layers.LSTM(32))
    elif model_type == "GRU":
        model.add(layers.GRU(32))

    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


models_list = ["RNN", "LSTM", "GRU"]
results = {}

for m in models_list:
    print(f"\nTraining {m} Model\n")

    model = build_model(m)

    model.fit(xtrain, ytrain, epochs=10, batch_size=64, validation_split=0.2, verbose=1)

    loss, acc = model.evaluate(xtest, ytest, verbose=0)

    results[m] = acc
    print(f"{m} Test Accuracy: {acc:.4f}")


print("\nFinal Comparison\n")

for model_name, accuracy in results.items():
    print(f"{model_name}: {accuracy:.4f}")

best_model = max(results, key=results.get)
print(f"\nBest Performing Model: {best_model}")
