import numpy as np
import pandas as pd
import unittest


# 0            1       2*    3    4    5      6      7*      8     9*     10
# PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
# 2, 1, "Cumings, Mrs. John Bradley (Florence Briggs Thayer)", female, 38, 1, 0, PC17599, 71.2833, C85, C3

# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked


def load_train_samples(sample_file):
    samples_df = pd.read_csv(sample_file)
    x = samples_df.drop(['Survived'], axis=1)
    y = samples_df.loc[:, 'Survived']
    return x, y


def load_test_samples(sample_file):
    x = pd.read_csv(sample_file)
    return x


def add_x0(x):
    x['x0'] = 1
    return x


def remove_fields(x):
    x = x.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    return x


def transform_sex(x):
    def convert_sex(row):
        if row['Sex'] == 'female':
            return 0
        elif row['Sex'] == 'male':
            return 1

    x['Sex'] = x.apply(lambda row: convert_sex(row), axis=1)
    return x


class TestUtil(unittest.TestCase):
    train_file = "./train.csv"
    test_file = "./test.csv"

    def test_load_train_data(self):
        train_x, train_y = load_train_samples(TestUtil.train_file)
        self.assertEqual(train_x.shape, (891, 11))
        self.assertEqual(train_y.shape, (891,))

    def test_load_test_data(self):
        test_x = load_test_samples(TestUtil.test_file)
        self.assertEqual(test_x.shape, (418, 11))

    def test_remove_fields(self):
        train_x, _ = load_train_samples(TestUtil.train_file)
        train_x = remove_fields(train_x)
        self.assertEqual(train_x.shape, (891, 7))

    def test_add_x0(self):
        train_x, _ = load_train_samples(TestUtil.train_file)
        train_x = add_x0(train_x)
        self.assertEqual(train_x.shape, (891, 12))
        self.assertEqual(train_x.loc[0, 'x0'], 1)

    def test_tranform_sex(self):
        train_x, _ = load_train_samples(TestUtil.train_file)
        train_x = transform_sex(train_x)
        self.assertEqual(train_x.shape, (891, 11))
        self.assertEqual(train_x.loc[10, 'Sex'], 0)
        self.assertEqual(train_x.loc[21, 'Sex'], 1)
        self.assertEqual(train_x.loc[888, 'Sex'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
