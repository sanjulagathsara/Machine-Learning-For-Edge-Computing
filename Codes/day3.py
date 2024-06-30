from tensorflow import keras
import matplotlib.pyplot as plt

X = [-2, -1, 0, 1, 2, 3]  #  Features
Y = [-8, -5, -2, 1, 4, 7]  # Labels

plt.plot(X, Y)

model = keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])]) # Dense = Layer of Neurons

model.summary()

model.compile(optimizer = keras.optimizers.SGD(learning_rate=0.01), loss = 'mean_squared_error')

print(model.predict([10.0]))

w, b = model.get_weights()
print(f'Before Training... \n weight = {w} \n bias = {b}')

hist = model.fit(X, Y, epochs = 500)

plt.title('Training ' + 'loss')
plt.plot(hist.history['loss'])
plt.show()

