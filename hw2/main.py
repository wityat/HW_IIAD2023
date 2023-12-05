from os.path import join

from hw2.models import *
from hw2.preprocessing import MnistDataloader

input_path = 'C:\\Users\\Витя Кенг\\PycharmProjects\\HW_IIAD2023\\hw2\\mnist'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath,
                                   test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()


def to_one_hot(y, num_classes):
    return np.eye(num_classes)[y]


y_train = to_one_hot(y_train, 10)
y_test = to_one_hot(y_test, 10)

nn = NeuralNetwork([
    Linear(784, 64),  # 784 - количество признаков (пикселей) в данных MNIST
    ReLU(),
    Linear(64, 10),  # 10 - количество классов в данных MNIST
    Softmax()
])


nn.loss = MSE()
nn.fit(x_train, y_train, epochs=500, batch_size=512, learning_rate=0.001)

y_pred = nn.predict(x_test)
accuracy = (y_pred == np.argmax(y_test, axis=1)).mean()
print(f'Test accuracy: {accuracy}')
