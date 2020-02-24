""""""
import tensorflow as tf
import numpy as np


def load_data():
    """
    https://gist.github.com/ogyalcin/51fbb2750d7001c30298e27a21969729#file-preparing_mnist-py
    :return:
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshaping the array to 4-dims so that it can work with the Keras API
    x_train = x_train.reshape(-1, 28, 28)
    x_test = x_test.reshape(-1, 28, 28)
    input_shape = (28, 28, 1)
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255

    set_size_train = min(len(np.where(y_train == 0)[0]), len(np.where(y_train == 1)[0]))
    set_size_test = min(len(np.where(y_test == 0)[0]), len(np.where(y_test == 1)[0]))

    x_train_zeros = x_train[np.where(y_train == 0), :][0]
    x_train_ones = x_train[np.where(y_train == 1), :][0]
    x_test_zeros = x_train[np.where(y_test == 0), :][0]
    x_test_ones = x_train[np.where(y_test == 1), :][0]

    x_train = np.concatenate([x_train_zeros[:set_size_train, ...], x_train_ones[:set_size_train, ...]], axis=0)
    x_test = np.concatenate([x_test_zeros[:set_size_test, ...], x_test_ones[:set_size_test, ...]], axis=0)
    y_train = np.concatenate((np.zeros((set_size_train, 1)), np.ones((set_size_train, 1))), axis=0)
    y_test = np.concatenate((np.zeros((set_size_test, 1)), np.ones((set_size_test, 1))), axis=0)
    print('x_train shape:', x_train.shape)
    print('Number of images in x_train', x_train.shape[0])
    print('Number of images in x_test', x_test.shape[0])

    return x_train, y_train, x_test, y_test


def get_aposterior_probabilities(n_samples, n_classes):
    rand_matrix = np.random.randint(
        low=0,
        high=5000,
        size=(n_samples, n_classes)
    )
    sum_row = np.repeat(rand_matrix.sum(axis=1).reshape((-1, 1)), repeats=rand_matrix.shape[1], axis=1)
    aposterior_probabilities = rand_matrix / sum_row
    return aposterior_probabilities


def _get_params(x_train, aposterior_probs):
    sum_prob = np.expand_dims(aposterior_probs.sum(axis=0), axis=1)
    sum_prob = np.repeat(sum_prob, repeats=28 * 28, axis=1).reshape((2, x_train.shape[1], x_train.shape[2]))

    probabilities = np.expand_dims(aposterior_probs.T, axis=2)
    probabilities = np.repeat(probabilities, repeats=28 * 28, axis=2).reshape((
        2,
        x_train.shape[0],
        x_train.shape[1],
        x_train.shape[2],
    ))

    x = np.expand_dims(x_train, axis=0)
    x = np.repeat(x, repeats=aposterior_probs.shape[1], axis=0)
    return np.multiply(x, probabilities).sum(axis=1) / sum_prob


def _get_con_probs(x_train, params):
    params = np.expand_dims(params, axis=0)
    params = np.repeat(params, repeats=x_train.shape[0], axis=0)
    params = params.reshape((params.shape[0], params.shape[1], -1))

    x_train = np.expand_dims(x_train, axis=1)
    x_train = np.repeat(x_train, repeats=2, axis=1)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], -1))

    power_w = np.power(params, x_train).prod(axis=-1)
    power_d = np.power(1 - params, (1 - x_train)).prod(axis=-1)
    return np.multiply(power_w, power_d)


def _get_aposterior(_conditions, _aprior_probs):
    sum_prob = _conditions[:, 0] * _aprior_probs[0] + _conditions[:, 1] * _aprior_probs[1]

    aposterior_first = (_conditions[:, 0] * _aprior_probs[0]) / sum_prob
    aposterior1_second = (_conditions[:, 1] * _aprior_probs[1]) / sum_prob

    return np.stack((aposterior_first, aposterior1_second), axis=1)


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = load_data()
    aposterior_probs = get_aposterior_probabilities(X_train.shape[0], 2)

    for epoch in range(10):
        aprior_probs = aposterior_probs.mean(axis=0)
        parameters = _get_params(X_train, aposterior_probs)
        conditions = _get_con_probs(X_train, parameters)
        aposterior_probs = 1 - _get_aposterior(conditions, aprior_probs)

        prediction = _get_con_probs(X_test, parameters).argmax(axis=1)
        _error_value = (prediction - Y_test.flatten()).sum() / len(prediction)
        if _error_value < 0:
            _error_value = ((1 - prediction) - Y_test.flatten()).sum() / len((1 - prediction))
        print("\nerror value for {} iteration : ".format(epoch + 1), _error_value)
