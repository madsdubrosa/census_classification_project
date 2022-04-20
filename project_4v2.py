
import numpy
from collections import Counter

import pydot
from sklearn.model_selection import train_test_split
from random import randint
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz

from io import StringIO

"""
Common Theme for data partitioning:

dictionary of paritions

self.raw_data = the data you read from

def processed_data(self):
    here we would process the self.raw_data field and SET self.data

self.data = {
    "train": [init_x, init_y, processed_x, processed_y],
    "val": [init_x, init_y, processed_x, processed_y],
    "test": [init_x, init_y, processed_x, processed_y]
}

"""


class CensusDecisionTreeData:
    def __init__(self):
        self.features = ["age", "workclass", "education", "marital_status", "occupation", "relationship", "race", "sex",
                    "capital_gain", "capital_loss", "hours_per_week", "native_country"]
        self.workclass = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]
        self.education = ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]
        self.marital_status = ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
        self.occupation = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]
        self.relationship = ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
        self.race = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
        self.sex = ["Female", "Male"]
        self.native_country = ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]
        self.categorical = {"workclass": self.workclass,
                       "education": self.education,
                       "marital_status": self.marital_status,
                       "occupation": self.occupation,
                       "relationship": self.relationship,
                       "race": self.race,
                       "sex": self.sex,
                       "native_country": self.native_country}
        self.total_features = self.get_total_features()
        self.income_level = ["<=50K", ">50K"]
        self.content = None
        self.initial_training_data = None
        self.initial_y_training_values = None
        self.mean_mode_features = None
        self.complete_training_data = None
        self.total_training_data = None
        self.training_data = None
        self.validation_data = None
        self.initial_validation_data = None
        self.initial_y_validation_values = None
        self.complete_validation_data = None
        self.total_validation_data = None
        self.total_y_training_data = None
        self.total_y_validation_data = None
        self.total_y_test_data = None
        self.test_data = None
        self.initial_test_data = None
        self.initial_y_test_values = None
        self.complete_test_data = None
        self.total_test_data = None

    def get_total_features(self):
        all_features = []
        for feature in self.features:
            if self.categorical.get(feature) is None:
                all_features.append(feature)
            else:
                for category in self.categorical.get(feature):
                    all_features.append(f"{feature}**{category}")
        return all_features

    def read_data(self, filename, file_flag):

        with open(filename, "r") as fh:
            content = fh.readlines()
        self.content = content
        if file_flag == "train":
            self.training_data = content
        elif file_flag == "validation":
            self.validation_data = content
        elif file_flag == "test":
            self.test_data = content
        else:
            print("File flag is not supported")
            return

        return

    def get_initial_data(self, data_flag):
        if data_flag == "train":
            data = self.training_data
        elif data_flag == "validation":
            data = self.validation_data
        elif data_flag == "test":
            data = self.test_data
        else:
            print("Data flag is not supported")
            return

        total_vectors = []
        y_data = []

        for i in range(len(data)):
            vector_content = data[i]
            vector = [item.strip() for item in vector_content.split(",")]
            y_data.append(vector[-1].strip("."))
            total_vectors.append(vector[:len(vector) - 1])

        if data_flag == "train":
            self.initial_training_data = numpy.array(total_vectors)
            self.initial_y_training_values = numpy.array(y_data)
        elif data_flag == "validation":
            self.initial_validation_data = numpy.array(total_vectors)
            self.initial_y_validation_values = numpy.array(y_data)
        elif data_flag == "test":
            self.initial_test_data = numpy.array(total_vectors)
            self.initial_y_test_values = numpy.array(y_data)
        else:
            print("Data flag is not supported")
            return

        return

    def get_mean_and_mode_values(self, data_split):
        if data_split == "train":
            data = self.initial_training_data
        elif data_split == "validation":
            data = self.initial_validation_data
        elif data_split == "test":
            data = self.initial_test_data
        else:
            print("Data split is not supported")
            return

        mean_mode_features = []

        for i, feature in enumerate(self.features):  # feature = "age", i = 0
            column = data[:, i]
            values, counts = numpy.unique(column, return_counts=True)
            values_and_counts = numpy.asarray((values, counts)).T[values != "?", :]
            if self.categorical.get(feature) is None:
                values_and_counts = values_and_counts.astype(int)
                mean_mode_features.append(sum(values_and_counts[:, 0] * values_and_counts[:, 1]) / sum(values_and_counts[:, 1]))
            else:
                values_and_counts = values_and_counts.astype(str)
                mean_mode_features.append(values_and_counts[0, 0])

        self.mean_mode_features = mean_mode_features

        return


    def old_get_mean_and_mode_values(self, data_split):
        if data_split == "train":
            data = self.initial_training_data
        elif data_split == "validation":
            data = self.validation_data
        elif data_split == "test":
            data = []  # TODO
        else:
            print("Data split is not supported")
            return
        category_counters = {}
        continuous = {}
        for feature in self.features:
            if self.categorical.get(feature) is not None:
                category_counters[feature] = Counter()
            else:
                continuous[feature] = [0, 0]

        for i in range(len(data)):
            for j in range(len(data[i])):
                if str(data[i][j]) != "?":
                    if self.categorical.get(self.features[j]) is not None:
                        category_counters[self.features[j]][str(data[i][j])] += 1
                    else:  # NOTE(maddie): no missing values for continuous values seen, but supported below:
                        continuous.get(self.features[j])[0] += int(data[i][j])
                        continuous.get(self.features[j])[1] += 1

        mean_mode_features = []
        for k in range(len(self.features)):
            if self.categorical.get(self.features[k]) is not None:
                mean_mode_features.append(str(category_counters.get(self.features[k]).most_common(1)[0][0]))
            else:
                mean_mode_features.append(int(continuous.get(self.features[k])[0] / continuous.get(self.features[k])[1]))

        self.mean_mode_features = mean_mode_features
        return mean_mode_features

    def fill_missing_values(self, data_flag):
        if data_flag == "train":
            data = self.initial_training_data
        elif data_flag == "validation":
            data = self.initial_validation_data
        elif data_flag == "test":
            data = self.initial_test_data
        else:
            print("Data flag is not supported")
            return

        for i in range(len(data)):
            for j in range(len(self.mean_mode_features)):
                if str(data[i][j]) == "?":
                    data[i][j] = self.mean_mode_features[j]

        if data_flag == "train":
            self.complete_training_data = numpy.array(data)
        elif data_flag == "validation":
            self.complete_validation_data = numpy.array(data)
        elif data_flag == "test":
            self.complete_test_data = numpy.array(data)
        else:
            print("Data flag is not supported")
            return

        return

    def convert_feature_vector(self, data_flag):
        if data_flag == "train":
            data = self.complete_training_data
        elif data_flag == "validation":
            data = self.complete_validation_data
        elif data_flag == "test":
            data = self.complete_test_data
        else:
            print("Data flag is not supported")
            return

        total_data = []
        for i in range(len(data)):
            new_total_feature_vector = numpy.zeros(len(self.total_features))
            curr_index = 0
            for j in range(len(self.features)):
                if self.categorical.get(self.features[j]) is None:
                    new_total_feature_vector[curr_index] = data[i][j]
                else:
                    while f"{self.features[j]}**{data[i][j]}" != self.total_features[curr_index]:
                        curr_index += 1
                    new_total_feature_vector[curr_index] = 1
                curr_index += 1
            total_data.append(new_total_feature_vector)

        if data_flag == "train":
            self.total_training_data = numpy.array(total_data)
        elif data_flag == "validation":
            self.total_validation_data = numpy.array(total_data)
        elif data_flag == "test":
            self.total_test_data = numpy.array(total_data)
        else:
            print("Data flag is not supported")
            return

        return

    def convert_y_values(self, data_flag):
        if data_flag == "train":
            data = self.initial_y_training_values
        elif data_flag == "validation":
            data = self.initial_y_validation_values
        elif data_flag == "test":
            data = self.initial_y_test_values
        else:
            print("Data flag is not supported")
            return

        new_y = numpy.zeros(len(data))
        for i in range(len(data)):
            if data[i] != self.income_level[0]:
                new_y[i] = 1

        if data_flag == "train":
            self.total_y_training_data = numpy.array(new_y)
        elif data_flag == "validation":
            self.total_y_validation_data = numpy.array(new_y)
        elif data_flag == "test":
            self.total_y_test_data = numpy.array(new_y)
        else:
            print("Data flag is not supported")
            return

        return

    def split_training_and_val(self):
        self.training_data, self.validation_data, x, y = train_test_split(self.content, numpy.zeros(len(self.content)), train_size=0.7, random_state=12)
        return


class CensusDecisionTree:
    def __init__(self, training_x, training_y):
        self.x_train = training_x
        self.y_train = training_y
        self.x_test = None
        self.y_test = None
        self.max_depth_all = None
        self.min_samples_leaf_all = None
        self.training_accuracy_all_max_depth = None
        self.training_accuracy_all_min_samples_leaf = None
        self.testing_accuracy_all_max_depth = None
        self.testing_accuracy_all_min_samples_leaf = None
        self.max_depth = None
        self.min_samples_leaf = 1
        self.training_accuracy = None
        self.testing_accuracy = None
        self.tree = DecisionTreeClassifier
        self.y_train_predicted = None
        self.y_test_predicted = None

    def set_testing_data(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test
        return

    def fit_decision_tree_classifier(self):
        self.tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
        self.tree = self.tree.fit(self.x_train, self.y_train)
        return

    def predict_decision_tree(self, data_flag):
        if data_flag == "train":
            data = self.x_train
        elif data_flag == "test":
            data = self.x_test
        else:
            print("Data flag is not supported")
            return

        y_pred = self.tree.predict(data)

        if data_flag == "train":
            self.y_train_predicted = y_pred
        elif data_flag == "test":
            self.y_test_predicted = y_pred
        else:
            print("Data flag is not supported")
            return

        return

    def get_accuracy(self, data_flag):

        if data_flag == "train":
            y_pred = self.y_train_predicted
            y_actual = self.y_train
        elif data_flag == "test":
            y_pred = self.y_test_predicted
            y_actual = self.y_test
        else:
            print("Data flag is not supported")
            return

        k = 0
        for i in range(len(y_pred)):
            if y_pred[i] != y_actual[i]:
                k += 1

        accuracy = 1 - (k / len(y_actual))

        if data_flag == "train":
            self.training_accuracy = accuracy
        elif data_flag == "test":
            self.testing_accuracy = accuracy
        else:
            print("Data flag is not supported")
            return

        return

    def set_params(self, param_flag, param_list):
        if param_flag == "max_depth":
            self.max_depth_all = param_list
            self.max_depth = self.max_depth_all[0]
        elif param_flag == "min_samples_leaf":
            self.min_samples_leaf_all = param_list
            self.min_samples_leaf = self.min_samples_leaf_all[0]
        else:
            print("Param flag is not supported")
            return

    def try_param_combos(self, param_flag):
        if param_flag == "max_depth":
            variable = self.max_depth_all
        elif param_flag == "min_samples_leaf":
            variable = self.min_samples_leaf_all
        else:
            print("Param flag is not supported")
            return

        training_accuracy = []
        testing_accuracy = []
        for i in range(len(variable)):
            # self.tree = DecisionTreeClassifier
            if param_flag == "max_depth":
                self.max_depth = variable[i]
            elif param_flag == "min_samples_leaf":
                self.min_samples_leaf = variable[i]
            self.fit_decision_tree_classifier()
            self.predict_decision_tree("train")
            self.get_accuracy("train")
            training_accuracy.append(self.training_accuracy)
            self.predict_decision_tree("test")
            self.get_accuracy("test")
            testing_accuracy.append(self.testing_accuracy)

        if param_flag == "max_depth":
            self.training_accuracy_all_max_depth = training_accuracy
            self.testing_accuracy_all_max_depth = testing_accuracy
        elif param_flag == "min_samples_leaf":
            self.training_accuracy_all_min_samples_leaf = training_accuracy
            self.testing_accuracy_all_min_samples_leaf = testing_accuracy

        return

    def plot_param_combos(self, param_flag):
        if param_flag == "max_depth":
            variable = self.max_depth_all
            training_accuracy = self.training_accuracy_all_max_depth
            testing_accuracy = self.testing_accuracy_all_max_depth
        elif param_flag == "min_samples_leaf":
            variable = self.min_samples_leaf_all
            training_accuracy = self.training_accuracy_all_min_samples_leaf
            testing_accuracy = self.testing_accuracy_all_min_samples_leaf
        else:
            print("Param flag is not supported")
            return

        plt.rcParams['figure.figsize'] = [5, 5]
        plt.scatter(variable, training_accuracy, color="pink")
        plt.scatter(variable, testing_accuracy, color="blue")
        plt.xlabel(param_flag)
        plt.ylabel("accuracy")
        plt.title(f"{param_flag} / accuracy")
        plt.show()

        return

    def get_max_accuracy_param(self, param_flag):
        if param_flag == "max_depth":
            variable = self.max_depth_all
            training_accuracy = self.training_accuracy_all_max_depth
            testing_accuracy = self.testing_accuracy_all_max_depth
        elif param_flag == "min_samples_leaf":
            variable = self.min_samples_leaf_all
            training_accuracy = self.training_accuracy_all_min_samples_leaf
            testing_accuracy = self.testing_accuracy_all_min_samples_leaf
        else:
            print("Param flag is not supported")
            return

        #shoudl this be based on the training or validataion accuracy
        max_accuracy = numpy.argmax(testing_accuracy)
        value = variable[max_accuracy]

        if param_flag == "max_depth":
            self.max_depth = value
        elif param_flag == "min_samples_leaf":
            self.min_samples_leaf = value
        else:
            print("Param flag is not supported")
            return

        return

    def display_first_three_levels(self):
        sio = StringIO()
        export_graphviz(self.tree, out_file=sio, max_depth=3, filled=True)
        graph = pydot.graph_from_dot_data(sio.getvalue())
        graph[0].write_pdf("income.pdf")
        return




def main():
    filename = "/Users/ldubrosa/maddie-coding/homework/machine_learning/project_4/ps4_data/adult_train.txt"
    # filename = "/Users/mo/src-control/projects/kwellerprep/privates/maddie/maddie-coding/homework/machine_learning/project_4/ps4_data/adult_train.txt"
    data = CensusDecisionTreeData()
    data.read_data(filename, "train")
    data.split_training_and_val()

    # print(numpy.shape(cdt.training_data))
    # print(numpy.shape(cdt.validation_data))

    data.get_initial_data("train")
    data.get_mean_and_mode_values("train")
    # print(data.initial_training_data[8])
    data.fill_missing_values("train")
    # print(data.complete_training_data[8])
    data.convert_feature_vector("train")
    # print(data.total_training_data[8])
    # data.convert_y_values("train")

    data.get_initial_data("validation")
    data.fill_missing_values("validation")
    data.convert_feature_vector("validation")
    # data.convert_y_values("validation")

    tree = CensusDecisionTree(data.total_training_data, data.initial_y_training_values)
    # print(numpy.shape(tree.x_train))
    # print(numpy.shape(tree.y_train))

    # tree.fit_decision_tree_classifier()
    # tree.predict_decision_tree("train")
    # print(tree.y_train[0])
    # print(tree.y_train_predicted[0])
    # tree.get_accuracy("train")
    # print(tree.training_accuracy)

    tree.set_testing_data(data.total_validation_data, data.initial_y_validation_values)
    # tree.predict_decision_tree("test")
    # tree.get_accuracy("test")
    # print(tree.testing_accuracy)

    # max_depths = [i for i in range(1, 31)]
    # min_samples_leaf = [i for i in range(1, 51)]
    #
    # tree.set_params("max_depth", max_depths)
    # tree.set_params("min_samples_leaf", min_samples_leaf)
    #
    # tree.try_param_combos("max_depth")
    # # tree.plot_param_combos("max_depth")
    # tree.try_param_combos("min_samples_leaf")
    # # tree.plot_param_combos("min_samples_leaf")
    #
    # tree.get_max_accuracy_param("max_depth")
    # tree.get_max_accuracy_param("min_samples_leaf")
    #
    # print(tree.max_depth) #10
    # print(tree.min_samples_leaf) #32

    max_depths = [10]
    min_samples_leaf = [32]

    tree.set_params("max_depth", max_depths)
    tree.set_params("min_samples_leaf", min_samples_leaf)
    tree.fit_decision_tree_classifier()
    tree.predict_decision_tree("train")
    tree.predict_decision_tree("test")
    tree.get_accuracy("train")
    tree.get_accuracy("test")
    print(tree.training_accuracy)
    print(tree.testing_accuracy)
    tree.display_first_three_levels()

    return

    filename = "/Users/ldubrosa/maddie-coding/homework/machine_learning/project_4/ps4_data/adult_train.txt"

    data = CensusDecisionTreeData()

    data.read_data(filename, "train")
    data.get_initial_data("train")
    data.get_mean_and_mode_values("train")
    data.fill_missing_values("train")
    data.convert_feature_vector("train")

    tree = CensusDecisionTree(data.total_training_data, data.initial_y_training_values)
    max_depths = [10]
    min_samples_leaf = [32]

    tree.set_params("max_depth", max_depths)
    tree.set_params("min_samples_leaf", min_samples_leaf)
    tree.fit_decision_tree_classifier()
    tree.predict_decision_tree("train")
    tree.get_accuracy("train")

    test_filename = "/Users/ldubrosa/maddie-coding/homework/machine_learning/project_4/ps4_data/adult_test.txt"
    data.read_data(test_filename, "test")
    data.get_initial_data("test")
    # print(data.initial_test_data[0])
    data.fill_missing_values("test")
    # print(data.complete_test_data[0])
    data.convert_feature_vector("test")
    # print(data.total_test_data[0])
    # print(data.initial_y_test_values[0])

    tree.set_testing_data(data.total_test_data, data.initial_y_test_values)

    tree.predict_decision_tree("test")
    tree.get_accuracy("test")
    # print(tree.y_test[0])
    # print(tree.y_test_predicted[0])
    print(tree.testing_accuracy)


main()


