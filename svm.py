from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from util import load_train_samples, load_test_samples, total_transform, generate_submission, submit_test_result, frange
import matplotlib.pyplot as plt
import numpy as np


def grid_search(verbose=True):
    train_x, train_y = load_train_samples()
    train_x = total_transform(train_x)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_x = min_max_scaler.fit_transform(train_x)

    plt.figure(figsize=(8, 6))

    best_C = 1
    best_gamma = 1
    best_score_mean = 0.0
    best_score_std = 1.0
    c_list = frange(-3, 16, 0.5)          # frange(-5, 16, 1)
    gamma_list = frange(-8, 3, 0.5)      # frange(-15, 4, 1)
    Z = np.zeros((len(gamma_list), len(c_list)))
    for i, c in enumerate(c_list):
        for j, g in enumerate(gamma_list):
            C = pow(2, c)
            gamma = pow(2, g)
            svm = SVC(C=C, kernel='rbf', gamma=gamma, verbose=False)
            scores = cross_val_score(svm, train_x, train_y, cv=5)
            Z[j, i] = scores.mean()
            if verbose:
                print("c=%.1f, g=%.1f, Accuracy: %0.4f (+/- %0.3f)" % (c, g, scores.mean(), scores.std() * 2))
            if scores.mean() > best_score_mean or (scores.mean() == best_score_mean and scores.std() < best_score_std):
                best_score_mean = scores.mean()
                best_score_std = scores.std()
                best_C = C
                best_gamma = gamma

    if verbose:
        print("best_C = %.2e" % best_C)
        print("best_gamma = %.2e" % best_gamma)
        print("best_cv_score = %.4f" % best_score_mean)

    X, Y = np.meshgrid(c_list, gamma_list)
    cs = plt.contour(X, Y, Z)
    plt.clabel(cs, inline=1)
    plt.xlabel("c - log2(C)")
    plt.ylabel("g - log2(gamma)")
    plt.show()

    return best_C, best_gamma, best_score_mean


def generate_test_result(C=1, gamma=1, verbose=True):
    train_x, train_y = load_train_samples()
    train_x = total_transform(train_x)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_x = min_max_scaler.fit_transform(train_x)

    svm = SVC(C=C, kernel='rbf', gamma=gamma, verbose=False)
    svm.fit(train_x, train_y)

    test_x_raw = load_test_samples()
    test_x = total_transform(test_x_raw)
    test_x = min_max_scaler.fit_transform(test_x)
    predict_y = svm.predict(test_x)

    submission_file = "svm_prediction.csv"
    submission_df = generate_submission(test_x_raw, predict_y)
    submission_df.to_csv(submission_file, index=False)
    if verbose:
        print()
        print("C = %.2e, gamma = %.2e" % (C, gamma))
        print(submission_df)
    print("submission file " + submission_file + " is generated.")
    return submission_file


def main(auto_submit_score=0.85):
    best_C, best_gamma, best_cv_score = grid_search()
    submission_file = generate_test_result(best_C, best_gamma, verbose=False)
    if best_cv_score >= auto_submit_score:
        submit_test_result(submission_file, "svm baseline - submit within python")


if __name__ == '__main__':
    main()
