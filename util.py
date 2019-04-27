import pandas as pd
import unittest
import subprocess
import shlex


desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)


# 0            1       2*    3    4    5      6      7*      8     9*     10
# PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
# 2, 1, "Cumings, Mrs. John Bradley (Florence Briggs Thayer)", female, 38, 1, 0, PC17599, 71.2833, C85, C3

# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked


def load_train_samples(sample_file="./train.csv"):
    samples_df = pd.read_csv(sample_file)
    x = samples_df.drop(['Survived'], axis=1)
    y = samples_df.loc[:, 'Survived']
    return x, y


def load_test_samples(sample_file="./test.csv"):
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


def transform_embarked(x):
    x = pd.get_dummies(x, columns=['Embarked'], prefix=['Embarked'])
    return x


def impute(x):
    x.fillna(x.mean(), inplace=True)
    return x


def total_transform(x, need_x0=False):
    x = remove_fields(x)
    x = transform_sex(x)
    x = transform_embarked(x)
    x = impute(x)
    if need_x0:
        x = add_x0(x)
    return x


def generate_submission(test_x_raw, predict_y):
    df = test_x_raw[["PassengerId"]]
    df = df.assign(Survived=pd.Series(predict_y).values)
    return df


def submit_test_result(submission_file, message):
    command = "kaggle competitions submit -c titanic -f " + submission_file + " -m \"" + message + "\""
    res = subprocess.check_output(shlex.split(command))
    for line in res.splitlines():
        print(line)


class TestUtil(unittest.TestCase):
    def test_load_train_data(self):
        train_x, train_y = load_train_samples()
        self.assertEqual(train_x.shape, (891, 11))
        self.assertEqual(train_y.shape, (891,))

    def test_load_test_data(self):
        test_x = load_test_samples()
        self.assertEqual(test_x.shape, (418, 11))

    def test_remove_fields(self):
        train_x, _ = load_train_samples()
        train_x = remove_fields(train_x)
        self.assertEqual(train_x.shape, (891, 7))

    def test_add_x0(self):
        train_x, _ = load_train_samples()
        train_x = add_x0(train_x)
        self.assertEqual(train_x.shape, (891, 12))
        self.assertEqual(train_x.loc[0, 'x0'], 1)

    def test_transform_sex(self):
        train_x, _ = load_train_samples()
        shape = train_x.shape
        train_x = transform_sex(train_x)
        self.assertEqual(train_x.shape, shape)
        self.assertEqual(train_x.loc[10, 'Sex'], 0)
        self.assertEqual(train_x.loc[21, 'Sex'], 1)
        self.assertEqual(train_x.loc[888, 'Sex'], 0)

    def test_transform_embark(self):
        train_x, _ = load_train_samples()
        shape = train_x.shape
        train_x = transform_embarked(train_x)
        self.assertEqual(train_x.shape, (shape[0], shape[1]+2))
        self.assertEqual(train_x.loc[1, 'Embarked_C'], 1)
        self.assertEqual(train_x.loc[3, 'Embarked_S'], 1)
        self.assertEqual(train_x.loc[5, 'Embarked_Q'], 1)

    def test_impute(self):
        train_x, _ = load_train_samples()
        train_x = impute(train_x)
        self.assertAlmostEqual(train_x.loc[5, 'Age'], 29.699118, 5)

    def test_total_transform(self):
        train_x, _ = load_train_samples()
        #print('\n', train_x[:6])
        train_x = total_transform(train_x)
        #print('\n', train_x[:5])

    def test_submit_test_result(self):
        if False:
            submit_test_result("lr_prediction.csv", "test kaggle API in python")


if __name__ == '__main__':
    unittest.main(verbosity=2)
