# STEP 1: rename this file to textclassify_model.py

# feel free to include more imports as needed here
# these are the ones that we used for the base model
import numpy as np
import sys
from collections import Counter
import numpy

"""
Your name and file comment here:
Karan Satwani
"""

"""
Cite your sources here:
List of pronouns found: 
https://www.thefreedictionary.com/List-of-pronouns.htm"""

"""
Implement your functions that are not methods of the TextClassify class here
"""


def generate_tuples_from_file(training_file_path):
    """
  Generates tuples from file formated like:
  id\ttext\tlabel
  Parameters:
    training_file_path - str path to file to read in
  Return:
    a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
  """
    f = open(training_file_path, "r", encoding="utf8")
    listOfExamples = []
    for review in f:
        if len(review.strip()) == 0:
            continue
        dataInReview = review.split("\t")
        for i in range(len(dataInReview)):
            # remove any extraneous whitespace
            dataInReview[i] = dataInReview[i].strip()
        t = tuple(dataInReview)
        listOfExamples.append(t)
    f.close()
    return listOfExamples


def precision(gold_labels, predicted_labels):
    """
  Calculates the precision for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double precision (a number from 0 to 1)
  """
    correct_positive_preds = 0
    total_positive_preds = 0

    for i in range(len(predicted_labels)):
        if predicted_labels[i] == '1':
            total_positive_preds += 1

            if gold_labels[i] == '1':
                correct_positive_preds += 1

    return correct_positive_preds / total_positive_preds


def recall(gold_labels, predicted_labels):
    """
  Calculates the recall for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double recall (a number from 0 to 1)
  """
    positive_preds = 0
    actual_positive_preds = 0

    for i in range(len(predicted_labels)):
        if gold_labels[i] == '1':
            actual_positive_preds += 1

            if predicted_labels[i] == '1':
                positive_preds += 1

    return positive_preds / actual_positive_preds


def f1(gold_labels, predicted_labels):
    """
  Calculates the f1 for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double f1 (a number from 0 to 1)
  """
    prec = precision(gold_labels, predicted_labels)
    rec = recall(gold_labels, predicted_labels)

    if prec == 0 and rec == 0:
        return 0

    f_score = (2 * prec * rec) / (prec + rec)

    return f_score


"""
Implement any other non-required functions here
"""


def load_lexicon(filename):
    """
  Load a file from Bing Liu's sentiment lexicon
  (https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html), containing
  English words in Latin-1 encoding.

  One file contains a list of positive words, and the other contains
  a list of negative words. The files contain comment lines starting
  with ';' and blank lines, which should be skipped.
  """
    lexicon = []
    with open(filename, encoding='latin-1') as infile:
        for line in infile:
            line = line.rstrip()
            if line and not line.startswith(';'):
                lexicon.append(line)
    return lexicon


def load_pronoun(filename):
    """
  Loading a txt file containing over 100 pronons
  list of pronouns taken from:
  https://www.thefreedictionary.com/List-of-pronouns.htm

  """
    pronouns = []
    with open(filename, encoding="utf8") as infile:
        for line in infile:
            pronouns.append(line.split("\n")[0])
    return pronouns


"""
implement your TextClassify class here
"""


class TextClassify:

    def __init__(self):
        # do whatever you need to do to set up your class here
        self.vocab_size = 0
        self.unique_vocab = []
        self.positive_reviews = []
        self.positive_dict = Counter
        self.negative_reviews = []
        self.negative_dict = Counter
        self.positive_count = 0
        self.negative_count = 0

    def train(self, examples):
        """
    Trains the classifier based on the given examples
    Parameters:
      examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    Return: None
    """
        words = []
        pos_word_list = []
        neg_word_list = []
        for i in range(len(examples)):
            temp_words = examples[i][1].split()
            if (examples[i][2] == '0'):
                self.negative_reviews.append(examples[i][1])
            else:
                self.positive_reviews.append(examples[i][1])
            for word in temp_words:
                words.append(word)
                if (examples[i][2] == '0'):  # negative reviews
                    neg_word_list.append(word)
                    self.negative_count += 1
                else:
                    pos_word_list.append(word)
                    self.positive_count += 1

        self.unique_vocab = Counter(words)
        self.vocab_size = len(self.unique_vocab)
        self.positive_dict = Counter(pos_word_list)
        self.negative_dict = Counter(neg_word_list)

        pass

    def score(self, data):
        """
    Score a given piece of text
    Parameters:
      data - str like "I loved the hotel"
    Return: dict of class: score mappings
    """
        data = data.split(" ")

        # how many sentences are +ve v/s -ve
        positive_review_multiplier = len(self.positive_reviews) / (
                len(self.positive_reviews) + len(self.negative_reviews))
        negative_review_multiplier = 1 - positive_review_multiplier
        positive_denominator = self.positive_count + self.vocab_size
        negative_denominator = self.negative_count + self.vocab_size
        print(self.positive_count)
        print(self.negative_count)
        positive_rating = positive_review_multiplier
        negative_rating = negative_review_multiplier

        for word in data:
            if word in self.positive_dict:
                positive_numerator = self.positive_dict.get(word) + 1
                positive_rating = positive_rating * (positive_numerator) / positive_denominator
                if word not in self.negative_dict:
                    negative_rating = negative_rating * (1) / negative_denominator
            if word in self.negative_dict:
                negative_numerator = self.negative_dict.get(word) + 1
                negative_rating = negative_rating * (negative_numerator) / negative_denominator
                # if word is in negative word but not in postive dict, we take its count as 1
                if word not in self.positive_dict:
                    positive_rating = positive_rating * (1) / positive_denominator

        dict = {'1': positive_rating, '0': negative_rating}
        print(dict)
        return dict

    def classify(self, data):
        """
    Label a given piece of text
    Parameters:
      data - str like "I loved the hotel"
    Return: string class label
    """
        classify = self.score(data)
        if classify['1'] > classify['0']:
            return "1"
        else:
            return "0"

    def featurize(self, data):
        """
    we use this format to make implementation of your TextClassifyImproved model more straightforward and to be 
    consistent with what you see in nltk
    Parameters:
      data - str like "I loved the hotel"
    Return: a list of tuples linking features to values
    for BoW, a list of tuples linking every word to True [("I", True), ("loved", True), ("it", True)]
    """
        features = []
        words = data.split()

        for word in words:
            if word in self.unique_vocab:
                features.append((word, True))
            else:
                features.append((word, False))

        return features

    def __str__(self):
        return "Naive Bayes - bag-of-words baseline"


class TextClassifyImproved:

    def __init__(self):
        self.vocab_size = 0
        self.unique_vocab = []
        self.positive_reviews = []
        self.negative_reviews = []
        self.positive_count = 0
        self.negative_count = 0

        # improved
        self.all_reviews = []
        self.all_labels = []
        self.BOG = []
        self.features = []
        self.weights = []
        self.bias = 0
        self.learning_rate = 0.8
        self.updated_weights = []
        self.epochs = 100
        # additonal features
        self.positive_words = Counter(load_lexicon("positive-words.txt"))
        self.negative_words = Counter(load_lexicon("negative-words.txt"))
        self.pronouns = Counter(load_pronoun("pronouns.txt"))
        self.review_lengths = []
        pass

    def sigmoid(self, x):
        """
      Calculates the sigmoid of a scalar or
      an numpy array-like set of values
      Parameters:
      x: input value(s)
      return
      Scalar or array corresponding to x passed through the sigmoid function
      """
        return 1 / (1 + np.e ** (-1 * x))

    def train(self, examples):
        """
    Trains the classifier based on the given examples
    Parameters:
      examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    Return: None
    """
        words = []
        pos_word_list = []
        neg_word_list = []
        # This loop goes through the input parameter, creates a list of positive and negative reviews
        # saves the length of each review and also creates a list of all words in reviews
        for i in range(len(examples)):
            temp_words = examples[i][1].split()
            self.all_reviews.append(examples[i][1])
            self.all_labels.append(int(examples[i][2]))
            # feature 3 - saving length of each review in a list to be added as a feature
            self.review_lengths.append(len(temp_words))
            if (examples[i][2] == '0'):
                self.negative_reviews.append(examples[i][1])
            else:
                self.positive_reviews.append(examples[i][1])
            for word in temp_words:
                words.append(word)

        # creating a counter for all the words, basically creating a dictionary of unique words
        self.unique_vocab = Counter(words)
        # taking all unique words in a list, this will be features for BOW
        self.features = list(self.unique_vocab.keys())
        print(self.features)
        # feature 3 - initialzing counter
        counter = -1

        # for token in self.features:
        for review in self.all_reviews:
            # Feature 2 - positive and negative count to be added as features
            pos_count = 0
            neg_count = 0
            # feature 3 - running for feature 3, to be used as index position
            counter += 1
            # feature 4 - initializing pronoun count for a review
            pronoun_count = 0

            temp_vector = []
            review_words = review.split()
            # making a dict of a single review to check if the feature is in that review
            temp_review_dict = Counter(review_words)
            for word in self.features:
                if word in temp_review_dict:
                    temp_vector.append(int(temp_review_dict.get(word)))
                else:
                    temp_vector.append(0)

                # Feature 2 - increasing count of positive and negative words
                if word in self.positive_words and word in temp_review_dict:
                    # print(f"positive {word}")
                    pos_count += 1
                elif word in self.negative_words and word in temp_review_dict:
                    # print(f"negative {word}")
                    neg_count += 1

                # Feature 4 - taking the pronount count
                if word in self.pronouns:
                    pronoun_count += 1
            # Feature 2 - adding postive and negative count of words in that review as features
            temp_vector.append(pos_count)
            temp_vector.append(neg_count)
            # Feature 3 - adding lenght of vector as feature
            temp_vector.append(self.review_lengths[counter])
            # Feature 4 - adding pronon as feature
            temp_vector.append(pronoun_count)
            self.BOG.append(temp_vector)

        self.vocab_size = len(self.features)
        # Feature 2 - adding 2 here for positive and negative count features
        self.vocab_size += 2
        # Feature 3 - adding 1 here for lenght of vector which will be added as feature
        self.vocab_size += 1
        # Feature 4 - adding 1 here for pronouns count
        self.vocab_size += 1

        self.weights = [0] * (self.vocab_size)

        # calculate P(y = 1) and getting updated weights, running it epochs times
        for i in range(self.epochs):
            for j in range(len(self.BOG)):
                sig = numpy.dot(self.BOG[j], self.weights) + self.bias
                y_1 = self.sigmoid(sig)
                temp_weights = (y_1 - self.all_labels[j]) * np.asarray(self.BOG[j])
                self.updated_weights = self.weights - (self.learning_rate * temp_weights)
                self.weights = self.updated_weights
        self.positive_dict = Counter(pos_word_list)
        self.negative_dict = Counter(neg_word_list)
        print(self.BOG[0])
        print(len(self.BOG))
        pass

    def score(self, data):
        """
    Score a given piece of text
    you’ll compute e ^ (log(p(c)) + sum(log(p(w_i | c))) here
    
    Parameters:
      data - str like "I loved the hotel"
    Return: dict of class: score mappings
    return a dictionary of the values of P(data | c)  for each class, 
    as in section 4.3 of the textbook e.g. {"0": 0.000061, "1": 0.000032}
    """
        data = data.split()
        example_count = Counter(data)
        temp_vector = []
        # Initializing variables for features
        pos_count = 0
        neg_count = 0
        pronoun_count = 0
        for word in self.features:
            if word in example_count:
                temp_vector.append(example_count.get(word))
            else:
                temp_vector.append(0)
            # Feature 2 - taking into account positive and negative words in the given review
            if word in self.positive_words and word in example_count:
                pos_count += 1
            elif word in self.negative_words and word in example_count:
                neg_count += 1
            # Feature 4 - updating if the the word is pronoun
            if word in self.pronouns:
                pronoun_count += 1

        # Feature 2 - adding the count for positive and negative as features
        temp_vector.append(pos_count)
        temp_vector.append(neg_count)
        # Feature 3 - adding lenght of data parameter as feature
        temp_vector.append(len(data))
        # Feature 4 -  adding pronoun count as feature
        temp_vector.append(pronoun_count)

        # calculate P(y = 1) and P(y = 0),
        sig = numpy.dot(temp_vector, self.weights) + self.bias
        y_1 = self.sigmoid(sig)
        y_0 = 1 - y_1
        dict = {'1': y_1, '0': y_0}
        return dict

    def classify(self, data):
        """
    Label a given piece of text
    Parameters:
      data - str like "I loved the hotel"
    Return: string class label
    """
        classify = self.score(data)
        if classify['1'] > classify['0']:
            return "1"
        else:
            return "0"

    def featurize(self, data):
        """
    we use this format to make implementation of this class more straightforward and to be 
    consistent with what you see in nltk
    Parameters:
      data - str like "I loved the hotel"
    Return: a list of tuples linking features to values
    for BoW, a list of tuples linking every word to True [("I", True), ("loved", True), ("it", True)]
    """
        features = []
        words = data.split()

        for word in words:
            if word in self.unique_vocab:
                features.append((word, True))
            else:
                features.append((word, False))

        return features

    def __str__(self):
        return "Improved Classifier"


def main():
    training = sys.argv[1]
    testing = sys.argv[2]

    classifier = TextClassify()
    # print(classifier)
    # # do the things that you need to with your base class
    samples = generate_tuples_from_file(training)
    classifier.train(samples)

    test_samples = generate_tuples_from_file(testing)

    gold_labels = []
    pred_labels = []
    for sample in test_samples:
        idx, text, label = sample
        gold_labels.append(label)
        pred_labels.append(classifier.classify(text))

    # report precision, recall, f1
    print(f"Precision: {precision(gold_labels, pred_labels)}\n")
    print(f"Recall: {recall(gold_labels, pred_labels)}\n")
    print(f"F1 Score: {f1(gold_labels, pred_labels)}")

    # printing improved class after calling functions
    classifier = TextClassifyImproved()
    samples = generate_tuples_from_file(training)
    classifier.train(samples)

    test_samples = generate_tuples_from_file(testing)

    gold_labels = []
    pred_labels = []
    for sample in test_samples:
        idx, text, label = sample
        gold_labels.append(label)
        pred_labels.append(classifier.classify(text))

    # report precision, recall, f1
    print("\nImproved Model")
    print(f"Precision: {precision(gold_labels, pred_labels)}\n")
    print(f"Recall: {recall(gold_labels, pred_labels)}\n")
    print(f"F1 Score: {f1(gold_labels, pred_labels)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:", "python textclassify_model.py training-file.txt testing-file.txt")
        sys.exit(1)
    main()
