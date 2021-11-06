# Created by Steven Yuan

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
import pydot
import math


# ===============================
# Helper Functions for load_data
# ===============================
def __process_files(file_name, is_real):
    """
    Return a tuple of headlines and their corresponding labels.
    :param file_name: clean_real.txt or clean_fake.txt
    :param is_real: True if file_name == "clean_real.txt"; False otherwise.
    :return: a tuple of headlines and their corresponding labels
    """
    with open(file_name) as f:
        lines = [line.strip() for line in f.readlines()]
        labels = [is_real for _ in range(len(lines))]
    return lines, labels


def __combine_data(real: tuple, fake: tuple):
    """
    :param real: a list of headlines of real news with corresponding labels
    :param fake: a list of headlines of fake news with corresponding labels
    :return: a matrix that corresponds to the dataset of real and fake headlines,
             a list of labels of all headlines, columns of the matrix(words)
    """
    aggregate_dataset = real[0] + fake[0]
    aggregate_labels = real[1] + fake[1]
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(aggregate_dataset)
    return count_matrix, aggregate_labels, vectorizer.get_feature_names_out()


def __split_data(data_matrix, labels):
    """
    Split the dataset into training, validation, and testing sets
    :param data_matrix: a matrix in which each row represents a headline and
                        each column represents a word(feature) and how many
                        times it appears in each headline
    :param labels: a list of True, False(real, fake) labels of the headlines
    :return: training set with labels, validation set with labels, and
             testing set with labels
    """
    x_train, x_test_and_val, y_train, y_test_and_val = \
        train_test_split(data_matrix, labels,
                         train_size=0.85, test_size=0.15)
    x_val, x_test, y_val, y_test = \
        train_test_split(x_test_and_val, y_test_and_val,
                         train_size=0.50, test_size=0.50)
    return x_train, x_val, x_test, y_train, y_val, y_test


# ======================================
# END of Helper Functions for load_data
# ======================================


def load_data():
    """
    Load data from files and generate required datasets
    :return: a list of feature names(words that appear in the headlines), training, validation, and testing sets,
             and their corresponding labels
    """
    real = __process_files("clean_real.txt", True)
    fake = __process_files("clean_fake.txt", False)
    matrix, aggregate_labels, feature_names_out = __combine_data(real, fake)
    x_train, x_val, x_test, y_train, y_val, y_test = \
        __split_data(matrix, aggregate_labels)
    return feature_names_out, x_train, x_val, x_test, y_train, y_val, y_test


# ==================================
# Helper Functions for select_model
# ==================================
def __generate_prediction(depth: int, criterion, x_train, y_train: list, x_val: list):
    """
    Generate prediction for a validation set given criterion.
    :param depth: max depth of the classification tree
    :param criterion: split criteria (in- formation gain and Gini coefficient)
    :param x_train: training dataset
    :param y_train: training labels
    :param x_val: validation set
    :return:
    """
    dtc = DecisionTreeClassifier(criterion=criterion, max_depth=depth)
    dtc.fit(x_train, y_train)
    return dtc.predict(x_val)


# replaced with accuracy_score function
# def __compute_accuracy(pred_lst, y_val):
#     non_acc_lst = []
#     for i in range(len(y_val)):
#         non_acc_lst.append(abs(pred_lst[i] ^ y_val[i]))
#     return 1 - sum(non_acc_lst) / len(non_acc_lst)


def __print_accuracies(criterion, depth, acc):
    """
    Report accuracies for each criterion for a particular depth
    :param criterion: split criteria (in- formation gain and Gini coefficient)
    :param depth: max depth of the classification tree
    :param acc: accuracy in floating point
    :return:
    """
    print(f"The accuracy for {criterion} criterion with max_depth = {depth}:"
          f" {round(acc * 100, 2)}%")


# ========================================
# END of Helper Functions for select_model
# ========================================


def select_model(depths: list, x_train: list, y_train: list, x_val: list, y_val: list):
    """
    Generate classification trees for each criterion and compare accuracies
    :param depths: max depth of the classification tree
    :param x_train: training dataset
    :param y_train: training labels
    :param x_val: validation dataset
    :param y_val: validation labels
    :return: None
    """
    for i in depths:
        gini_pred = __generate_prediction(i, 'gini', x_train, y_train, x_val)
        info_pred = __generate_prediction(i, 'entropy', x_train, y_train, x_val)
        gini_acc = accuracy_score(gini_pred, y_val)
        info_acc = accuracy_score(info_pred, y_val)
        __print_accuracies("Gini coefficient", i, gini_acc)
        __print_accuracies("information gain", i, info_acc)


def graph_best_tree(words, x_train, y_train):
    """
    Graph the tree with the best accuracy.
    :param words: features
    :param x_train: training dataset
    :param y_train: training labels
    :return: None
    """
    best = DecisionTreeClassifier(criterion="gini", max_depth=55)
    best.fit(x_train, y_train)
    export_graphviz(best, out_file="decision_tree.dot",
                    feature_names=words, class_names=['fake', 'real'],
                    max_depth=2, filled=True)
    parsed_graph, = pydot.graph_from_dot_file("decision_tree.dot")
    parsed_graph.write_png("decision_tree.png")


# ==============================================
# Helper Functions for compute_information_gain
# ==============================================
def __compute_info_content(prob):
    """
    Compute information content given probability of an event.
    :param prob: probability of an event
    :return: - prob * log_2(prob)
    """
    if not prob:
        return 0
    return - prob * math.log(prob, 2)


def __get_root_entropy(y):
    """
    Get H(Y), where Y is the random variable signifying whether the headline is real or fake.
    :param y: training labels
    :return: entropy of Y
    """
    real_count, fake_count = 0, 0
    total = len(y)
    for label in y:
        if label:
            real_count += 1
        else:
            fake_count += 1
    return __compute_info_content(real_count / total) + \
        __compute_info_content(fake_count / total)


THRESHOLD = 0.5


def __get_cond_leaf_entropy(dt_matrix, split_value, y, words, t):
    """
    Return the conditional leaf entropy
    :param dt_matrix: training dataset
    :param split_value: the word that the tree splits on
    :param y: training labels
    :param words: the entire list of features
    :param t: threshold value
    """
    left_real, left_fake, right_real, right_fake = 0, 0, 0, 0
    for i in range(len(dt_matrix.toarray())):
        if dt_matrix[:, words.index(split_value)][i] > t:
            # if dt_matrix[i][np.where(words == split_value)] <= t
            if y[i]:
                right_real += 1
            else:
                right_fake += 1
        else:
            if y[i]:
                left_real += 1
            else:
                left_fake += 1
    sum_left = left_real + left_fake
    sum_right = right_real + right_fake
    sum_all = sum_left + sum_right
    hy_given_left = __compute_info_content(left_real / sum_left) + __compute_info_content(left_fake / sum_left)
    hy_given_right = __compute_info_content(right_real / sum_right) + __compute_info_content(right_fake / sum_right)
    p_left = sum_left / sum_all
    p_right = sum_right / sum_all
    return p_left * hy_given_left + p_right * hy_given_right


# =====================================================
# END of Helper Functions for compute_information_gain
# =====================================================
def compute_information_gain(dt_matrix, split_value, y, words, t):
    """
    Return IG(Y | X) = H(Y) - H(Y | X)
    """
    return __get_root_entropy(y) - __get_cond_leaf_entropy(dt_matrix, split_value, y, words, t)


def print_ig(split_value, ig):
    """
    Print information gain ig with split_value in a verbose manner.
    :param split_value: The keyword that the decision tree splits on
    :param ig: Information gain for the given split
    :return: None
    """
    print(f'''The information gain for attribute "{split_value}" is {round(ig, 4)}''')


# ==================
# The Main Function
# ==================
def main():
    print("Loading dataset...")
    feature_names_out, x_train, x_val, x_test, y_train, y_val, y_test = load_data()

    print("Comparing models...")
    select_model([8, 13, 21, 34, 55], x_train, y_train, x_val, y_val)

    print("Graphing the best tree...")
    graph_best_tree(feature_names_out, x_train, y_train)

    feature_names_out = feature_names_out.tolist()
    print("Computing information gain for the topmost split...")
    print_ig("the", compute_information_gain(x_train, "the", y_train, feature_names_out, THRESHOLD))
    print("Computing information gain for several other keywords...")
    print_ig("donald", compute_information_gain(x_train, "donald", y_train, feature_names_out, THRESHOLD))
    print_ig("trumps", compute_information_gain(x_train, "trumps", y_train, feature_names_out, THRESHOLD))
    print_ig("hillary", compute_information_gain(x_train, "hillary", y_train, feature_names_out, THRESHOLD))
    print("Done!")


if __name__ == "__main__":
    main()
