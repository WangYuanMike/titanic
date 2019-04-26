from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from util import load_train_samples, load_test_samples, total_transform, generate_gender_submission


def grid_search(verbose=True):
    best_c = 1
    best_score_mean = 0.0
    best_score_std = 1.0
    c_list = list(range(-5, 6))
    for c in c_list:
        C = pow(10, c)
        train_x, train_y = load_train_samples()
        train_x = total_transform(train_x, need_x0=True)
        lr = LogisticRegression(C=C, solver='liblinear')
        scores = cross_val_score(lr, train_x, train_y, cv=10)
        if verbose:
            print("C=%.2e, Accuracy: %0.4f (+/- %0.3f)" % (C, scores.mean(), scores.std() * 2))
        if scores.mean() > best_score_mean or (scores.mean() == best_score_mean and scores.std() < best_score_std):
            best_score_mean = scores.mean()
            best_score_std = scores.std()
            best_c = c
    return pow(10, best_c)


def generate_test_result(C=1, verbose=True):
    train_x, train_y = load_train_samples()
    train_x = total_transform(train_x, need_x0=True)
    lr = LogisticRegression(C=C, solver='liblinear')
    lr.fit(train_x, train_y)

    test_x_raw = load_test_samples()
    test_x = total_transform(test_x_raw, need_x0=True)
    predict_y = lr.predict(test_x)

    gender_submission_df = generate_gender_submission(test_x_raw, predict_y)
    gender_submission_df.to_csv("lr_prediction.csv", index=False)
    if verbose:
        print()
        print("C = %.2e" % C)
        print(gender_submission_df)
    print("gender submission file lr_prediction.csv is generated.")


def main():
    generate_test_result(grid_search())


if __name__ == '__main__':
    main()
