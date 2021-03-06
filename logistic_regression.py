from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from util import load_train_samples, load_test_samples, total_transform, generate_submission, submit_test_result


def grid_search(verbose=True):
    best_C = 1
    best_score_mean = 0.0
    best_score_std = 1.0
    c_list = list(range(-5, 6))
    for c in c_list:
        C = pow(10, c)
        train_x, train_y = load_train_samples()
        train_x = total_transform(train_x)
        lr = LogisticRegression(C=C, solver='liblinear')
        scores = cross_val_score(lr, train_x, train_y, cv=10)
        if verbose:
            print("C=%.2e, Accuracy: %0.4f (+/- %0.3f)" % (C, scores.mean(), scores.std() * 2))
        if scores.mean() > best_score_mean or (scores.mean() == best_score_mean and scores.std() < best_score_std):
            best_score_mean = scores.mean()
            best_score_std = scores.std()
            best_C = C
    if verbose:
        print("best_C = %.2e" % best_C)
        print("best_cv_score = %.4f" % best_score_mean)
    return best_C, best_score_mean


def generate_test_result(C=1, verbose=True):
    train_x, train_y = load_train_samples()
    train_x = total_transform(train_x)
    lr = LogisticRegression(C=C, solver='liblinear')
    lr.fit(train_x, train_y)

    test_x_raw = load_test_samples()
    test_x = total_transform(test_x_raw)
    predict_y = lr.predict(test_x)

    submission_file = "lr_prediction.csv"
    submission_df = generate_submission(test_x_raw, predict_y)
    submission_df.to_csv(submission_file, index=False)
    if verbose:
        print()
        print("C = %.2e" % C)
        print(submission_df)
    print("submission file " + submission_file + " is generated.")
    return submission_file


def main(auto_submit_score=0.8):
    best_C, best_cv_score = grid_search()
    submission_file = generate_test_result(best_C, verbose=False)
    if best_cv_score >= auto_submit_score:
        submit_test_result(submission_file, "logistic regression - submit within python")


if __name__ == '__main__':
    main()
