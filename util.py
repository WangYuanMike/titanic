import numpy as np
import pandas as pd


def load_train_samples(sample_file):
    samples_df = pd.read_csv(sample_file)
    samples_np = samples_df.values
    x_index = np.r_[0, 2:samples_np.shape[1]]
    x = samples_np[:, x_index]
    y = samples_np[:, 1]
    return x, y


def load_test_samples(sample_file):
    samples_df = pd.read_csv(sample_file)
    x = samples_df.values
    return x


def add_x0(x):
    x_plus = np.ones((x.shape[0], x.shape[1]+1))
    x_plus[:, :-1] = x
    x = x_plus
    return x


def unit_test():
    train_x, train_y = load_train_samples("./train.csv")
    print("Load train data:")
    print(train_x.shape)
    print(train_x[:5, :])
    print(train_y.shape)
    print(train_y[:5])
    print()

    test_x = load_test_samples("./test.csv")
    print("Load test data:")
    print(test_x.shape)
    print(test_x[:5])
    print()


if __name__ == '__main__':
    unit_test()
