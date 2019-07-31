#!/bin/python3

"""Sentiment Analysis

Usage: python3 prog3.py <training data> <testing data>
"""

import sys
import copy
import math


class Classify:
    """Runs the Naive Bayes algorithm.

    Methods:
    __init__(preprocessed)
    __str__()
    calculate_accuracy()
    classify_data()
    learn_parameters()
    pos_stat()
    run()
    set_stats(classif)
    """
    def __init__(self, preprocessed):
        """Constructor of the Classify class.

        Parameters:
        preprocessed -- a Preprocess object
        """
        self.training = preprocessed.training
        self.test = preprocessed.test

    def __str__(self):
        """Print the accuracy of the classifications."""
        result = []
        result.append("-------------------------------------")
        result.append("Training File: " + self.training.file_name)
        result.append("Testing File: " + self.test.file_name)
        result.append("Accuracy: " + str(self.accuracy))
        result.append("-------------------------------------")
        return "\n".join(result)

    def calculate_accuracy(self):
        """Calculate the accuracy of the classifications."""
        accuracy = 0
        for i, feature in enumerate(self.test.features):
            if feature.classlabel == self.test.classifications[i]:
                accuracy += 1
        accuracy /= len(self.test.classifications)
        self.accuracy = accuracy

    def classify_data(self):
        """Calculate the most likely classlabel and set it."""
        ln_prob_pos = math.log(self.prob_pos / self.prob_neg)
        ln_sum = 0
        for feature in self.test.features:
            ln_sum = 0
            for i, word in enumerate(feature):
                if word == 1:
                    ln_sum += math.log(self.prob_w_pos[i] / self.prob_w_neg[i])
            if ln_prob_pos + ln_sum > 0.0:
                feature.classify(1)
            else:
                feature.classify(0)

    def learn_parameters(self):
        """Find the general stats for all words."""
        self.pos_stat()
        self.prob_neg = 1.0 - self.prob_pos
        self.prob_w_pos = self.set_stats(1)
        self.prob_w_neg = self.set_stats(0)

    def pos_stat(self):
        """Find the percentage of positive reviews."""
        result = 0
        for feature in self.training.features:
            if feature.classlabel == 1:
                result += 1
        self.prob_pos = result / len(self.training.features)

    def run(self):
        """Go through the steps of classifying the data."""
        self.learn_parameters()
        self.classify_data()
        self.calculate_accuracy()

    def set_stats(self, classif):
        """Set the stats for each feature based on the classification.

        Parameters:
        classif -- the classification
        """
        num_words = len(self.training.features[0])
        result = [0 for i in range(num_words)]
        num_word_type = 0
        for feature in self.training.features:
            if feature.classlabel == classif:
                num_word_type += 1
                for word in range(num_words):
                    if feature[word] == 1:
                        result[word] += 1
        for word in range(num_words):
            # Uniform Dirichlet prior
            if result[word] == 0:
                result[word] = 1.0 / num_words
            else:
                result[word] = result[word] / num_word_type
        return result


class Data:
    """Review data stored in various ways.

    Methods:
    __init__(name)
    clean_data(temp)
    make_features(vocab)
    """
    def __init__(self, name):
        """Constructor for the Data class.

        Parameters:
        name -- the name of the file with the data
        """
        self.file_name = name
        with open(self.file_name) as f:
            temp = f.read()
        self.clean_data(temp)

    def clean_data(self, raw):
        """Strips excess characters.

        Parameters:
        raw -- raw data, straight from the file
        """
        raw = raw.lower()
        stringlist = raw.splitlines()
        self.data = []
        self.classifications = []
        for line in stringlist:
            word_list = line.split('\t')
            self.classifications.append(int(word_list[-1]))
        for line in [list(stringlist[i]) for i in range(len(stringlist))]:
            for val in line.copy():
                if not (val.isalpha() or val.isspace()):
                    line.remove(val)
            tempstring = "".join(line)
            self.data.append(tempstring.split())

    def make_features(self, vocab):
        """Convert data into feature objects

        Parameters:
        vocab -- a list of all words in the training data
        """
        self.features = []
        for line in self.data:
            new_feature = Feature(len(vocab))
            for word in line:
                new_feature.add(vocab, word)
            self.features.append(new_feature)


class Feature:
    """Vector of binary values referencing the vocabulary.

    Methods:
    __init(size)
    __len__()
    __getitem__(key)
    __setitem__(key, value)
    __str__()
    add(vocab, word)
    classify(pos)
    """
    def __init__(self, size):
        """Constructor for the Feature class.

        Parameters:
        size -- number of words in the vocabulary
        """
        self.length = size
        self.vector = [0 for i in range(size)]
        self.classlabel = -1

    def __len__(self):
        """Return the length of the vector."""
        return self.length

    def __getitem__(self, key):
        """Return an element of the vector."""
        return self.vector[key]

    def __setitem__(self, key, value):
        """Set an element of the vector."""
        self.vector[key] = value

    def __str__(self):
        """Printing method for a feature object."""
        return ",".join([",".join(map(str, self.vector)), str(self.classlabel)])

    def add(self, vocab, word):
        """Set the corresponding element of the vector if the word
        is in the vocabulary.

        Parameters:
        vocab -- a list of all words in the training data
        word -- the word to be added to the feature vector
        """
        if word in vocab:
            self.vector[vocab.index(word)] = 1

    def classify(self, pos):
        """Set the classification of the feature.

        Parameters:
        pos -- indicates if the classlabel is positive or negative
        """
        self.classlabel = pos


class Preprocess:
    """Preprocesses data for the Naive Bayes algorithm.

    Methods:
    __init__(training, test)
    make_vocab(data)
    output_features(data, file_name)
    vocab_str()
    run()
    """
    def __init__(self, training, test):
        """Constructor for the Preprocess class.

        Parameters:
        training -- file name for the training data
        test -- file name for the testing data
        """
        self.training = Data(training)
        self.test = Data(test)

    def make_vocab(self, data):
        """Make a list of vocabulary words.

        Parameters:
        data -- the sentences to be used
        """
        temp = set()
        for line in data:
            for word in line:
                temp.add(word)
        self.vocab = sorted(list(temp))

    def output_features(self, data, file_name):
        """Print feature information to a file.

        Parameters:
        data -- the sentences to be used
        file_name -- the name of the file to output to
        """
        with open(file_name, "w") as f:
            print(self.vocab_str(), file=f)
            for feature in data.features:
                print(feature, file=f)

    def vocab_str(self):
        """Return the vocabulary as a string of words."""
        return ",".join(self.vocab) + ",classlabel"

    def run(self):
        """Go through the steps of preprocessing the data."""
        self.make_vocab(self.training.data)
        self.training.make_features(self.vocab)
        for i, feature in enumerate(self.training.features):
            feature.classify(self.training.classifications[i])
        self.test.make_features(self.vocab)
        self.output_features(self.training, "preprocessed_train.txt")
        self.output_features(self.test, "preprocessed_test.txt")


def main(argv):
    """The main function of the program."""
    if len(argv) != 3:
        sys.exit("Usage: python3 prog3.py <training data> <testing data>")
    preprocessed = Preprocess(argv[1], argv[2])
    preprocessed.run()
    classified = Classify(preprocessed)
    classified.run()
    print(classified)


main(sys.argv)
