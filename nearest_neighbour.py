import collections
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from KMinHeap import kminheap


def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


def gensmallm_with_corruption(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])
    indices_to_corrupt = random.sample(range(len(y)), int(len(y) * 0.2))

    for i in indices_to_corrupt:
        lables = [1, 3, 4, 6]
        lables.remove(y[i])
        new_lable = lables[random.randrange(3)]
        y[i] = new_lable

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


# todo: complete the following functions, you may add auxiliary functions or define class to help you

def learnknn(k: int, x_train: np.array, y_train: np.array):
    """

    :param k: value of the nearest neighbour parameter k
    :param x_train: numpy array of size (m, d) containing the training sample
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    :return: classifier data structure
    """
    return (k, x_train, y_train)


def mostCommonNeighbor(indices: list):
    list_counter = collections.Counter(indices)
    knn = list_counter.most_common(1)
    return knn[0][0]


def predictknn(classifier, x_test: np.array):
    """

    :param classifier: data structure returned from the function learnknn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """
    k, x_train, y_train = classifier

    Ytestprediction = []
    for x in x_test:
        kheap = kminheap(k)
        for i in range(0, len(x_train)):
            kheap.insert(distance.euclidean(x, x_train[i]), i)

        lables = [y_train[x[1]] for x in kheap.elements]
        knn = mostCommonNeighbor(lables)
        Ytestprediction.append(knn)

    return np.transpose(np.array(Ytestprediction))


def simple_test():
    data = np.load('mnist_all.npz')

    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], 100)

    x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 50)

    classifer = learnknn(5, x_train, y_train)

    preds = predictknn(classifer, x_test)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array"
    assert preds.shape[0] == x_test.shape[0] and preds.shape[
        1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"

    # get a random example from the test set
    i = np.random.randint(0, x_test.shape[0])

    # this line should print the classification of the i'th test sample.
    print(f"The {i}'th test sample was classified as {preds[i]}")


def ex_2a():
    data = np.load('mnist_all.npz')
    acc_errors = []
    maximums = []
    minimums = []
    (X_test, y_test) = gensmallm([data['test1'], data['test3'], data['test4'], data['test6']], [1, 3, 4, 6],
                                 len(data['test1']) + len(data['test3']) + len(data['test4']) + len(data['test6']))

    for m in range(1, 101, 10):
        errors = []
        for i in range(0, 10):
            (X_train, y_train) = gensmallm([data['train1'], data['train3'], data['train4'], data['train6']],
                                           [1, 3, 4, 6], m)
            k = 1
            classifier = learnknn(k, X_train, y_train)
            y_testpredict = predictknn(classifier, X_test)
            errors.append(np.mean(y_test != y_testpredict))

        maximums.append(np.amax(np.array(errors)))
        minimums.append(np.amin(np.array(errors)))
        acc_errors.append(np.average(np.array(errors)))

    p1 = plt.bar(range(1, 101, 10), maximums, 4)
    p2 = plt.bar(range(1, 101, 10), minimums, 4)
    main_plt = plt.plot(range(1, 101, 10), acc_errors, color='red')
    plt.legend((main_plt[0], p1[0], p2[0]), ('Average error', 'Maximum', 'Minimum'))
    plt.xlabel('Sample Size')
    plt.ylabel('Error')
    plt.title('2.a: Error as function of sample size (k=1)')
    plt.locator_params(nbins=10)
    plt.show()


def ex_2e():
    data = np.load('mnist_all.npz')
    acc_errors = []
    maximums = []
    minimums = []
    (X_test, y_test) = gensmallm([data['test1'], data['test3'], data['test4'], data['test6']], [1, 3, 4, 6],
                                 len(data['test1']) + len(data['test3']) + len(data['test4']) + len(data['test6']))

    for k in range(1, 12):
        errors = []
        for i in range(0, 10):
            (X_train, y_train) = gensmallm([data['train1'], data['train3'], data['train4'], data['train6']],
                                           [1, 3, 4, 6], 100)
            classifier = learnknn(k, X_train, y_train)
            y_testpredict = predictknn(classifier, X_test)
            errors.append(np.mean(y_test != y_testpredict))

        maximums.append(np.amax(np.array(errors)))
        minimums.append(np.amin(np.array(errors)))
        acc_errors.append(np.average(np.array(errors)))

    p1 = plt.bar(range(1, 12), maximums, 0.5)
    p2 = plt.bar(range(1, 12), minimums, 0.5)
    main_plt = plt.plot(range(1, 12), acc_errors, color='red')
    plt.legend((main_plt[0], p1[0], p2[0]), ('Average Error', 'Maximum', 'Minimum'))
    plt.xlabel('k')
    plt.ylabel('Error')
    plt.title('2.e: Error as function of k (sample size=100)')
    plt.xticks(range(1, 12))
    plt.show()


def ex_2f():
    data = np.load('mnist_all.npz')
    acc_errors = []
    maximums = []
    minimums = []
    (X_test, y_test) = gensmallm_with_corruption([data['test1'], data['test3'], data['test4'], data['test6']],
                                                 [1, 3, 4, 6],
                                                 len(data['test1']) + len(data['test3']) + len(data['test4']) + len(
                                                     data['test6']))

    for k in range(1, 12):
        errors = []
        for i in range(10):
            (X_train, y_train) = gensmallm_with_corruption(
                [data['train1'], data['train3'], data['train4'], data['train6']],
                [1, 3, 4, 6], 100)
            classifier = learnknn(k, X_train, y_train)
            y_testpredict = predictknn(classifier, X_test)
            errors.append(np.mean(y_test != y_testpredict))

        maximums.append(np.amax(np.array(errors)))
        minimums.append(np.amin(np.array(errors)))
        acc_errors.append(np.average(np.array(errors)))

    p1 = plt.bar(range(1, 12), maximums, 0.5)
    p2 = plt.bar(range(1, 12), minimums, 0.5)
    main_plt = plt.plot(range(1, 12), acc_errors, color='red')
    plt.legend((main_plt[0], p1[0], p2[0]), ('Average Error', 'Maximum', 'Minimum'))
    plt.xlabel('k')
    plt.ylabel('Error')
    plt.title('2.f: Error as function of k (sample size=100) with 20% corrupted lables')
    plt.xticks(range(1, 12))
    plt.show()


def ex_5c():
    err = lambda a: min(a, 1 - a)
    f = lambda a: 2 * a * (1 - a)
    x_axis = [x / 1000 for x in range(1, 1001)]
    y_err = [err(x / 1000) for x in range(1, 1001)]
    y_f = [f(x / 1000) for x in range(1, 1001)]

    plt.plot(x_axis, y_err)
    plt.plot(x_axis, y_f, color='red')
    plt.legend(('err(h^*,D)=min{a,1-a}', 'f(a)=2a(1-a)'))
    plt.xlabel('a')
    plt.title('5.c: Error expectation and Bayes-optimal error')
    plt.show()


if __name__ == '__main__':
    # ex_2a()
    # ex_2e()
    # ex_2f()
    ex_5c()
    print("end")
