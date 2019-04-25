from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from util import load_train_samples, total_transform


def main():
    train_x, train_y = load_train_samples()
    train_x = total_transform(train_x, need_x0=True)
    lr = LogisticRegression()
    scores = cross_val_score(lr, train_x, train_y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


if __name__ == '__main__':
    main()
