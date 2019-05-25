from __future__ import division

from collections import defaultdict
from math import log
import csv


# create function to train
# it takes data data is a dict: (words) -> lang
# from tuple of language words -> to language
def train(data):
    # 2 dicts with 0 as default values
    # classes are languages, we classify texts by languages
    # counts how many programms written in each language
    # e.g. JAVA: 39123
    classes = defaultdict(lambda: 0)
    # frequenses is dict: (LANG, WORD) -> FREQ
    # from tuple of lang and a word from this lang -> to frequency of this word in lang
    # e.g. (C++, #include<iostream>): 302
    # means that #include<iostream> meets C++ 302 times
    frequences = defaultdict(lambda: 0)

    print('learning...')
    # features here are words from language
    # label is a language
    for features, label in data.items():
        # count each text in the language
        classes[label] += 1
        # for each word in the text
        for feature in features:
            # increase count for the word from the language
            # (C++, #include<iostream>): 302 -> (C++, #include<iostream>): 303
            frequences[label, feature] += 1

    # label is language here
    # feature is a word from this language
    for label, feature in frequences.keys():
        # divide each lang-word-counter by amount of programms in the language
        # we get the probability of meeting of the word in the text of the language
        frequences[label, feature] /= classes[label]

    # for each language
    for c in classes:
        # we get the probability of meeting of the language
        classes[c] /= len(data)

    print('learned.')

    return classes, frequences


def classify(classifier, features):
    # classes.keys is frequences of languages
    # prob is probability of meeting word in the language
    # prob type (LANGUAGE, WORD) -> PROBABILITY
    classes, prob = classifier
    # find minimum value in Languages by function which is math magic
    # key defines how we search minimum, how to compare values
    # how to map Languages to numbers to make them comparable
    return min(
            classes.keys(),
            key = lambda cl: -log(classes[cl]) + sum(-log(prob.get((cl,f), 10**(-7))) for f in features)
    )


# function to extract features modify it change
# how words are separated, which words are taken and so on
def get_features(text: str) -> tuple:
    # v2 addition: delete words with length < 2
    long_word = lambda s: len(s) > 1

    return tuple( filter(long_word, text.split()) )


# create dicts for train and tests
# their type is Dict<TupleOfStrings, String>
# where TupleOfStrings are words for language
# String is language name
test_data = {}
train_data = {}


def read_data():
    # just reading data to test_data and train_data
    # open file data.csv
    data_csv = open('data.csv', 'r')
    # create csv reader from the file
    data_reader = csv.reader(data_csv)
    # reading header to make it absent in data
    header = next(data_reader)

    print('reading train data')
    # reading 400 thousands of samples as training data
    i = 0
    for line in data_reader:
        # line is solutionId, program, language
        # e.g. 1234, "#include<iostream> using namespace std;...", 'C++'
        # we split program by space and get list ['#include<iostream>', 'using', 'namespace', 'std;']
        # we delete words from 1 symbol
        # we make tuple from result. tuple is immutable list
        words = get_features(line[1])
        train_data[words] = line[2]
        i += 1
        # we take only 400_000 programs for training, the others for test
#FIXME
#TODO replace 400 with 400_000 or number of programs to learn
# remaining will be used for test
# please note: original dataset has over 620 thousands of samples (link in README.md)
# data.csv in this repository has a thousand times less
        if i >= 400:
            break
    print('reading test data')
    # other 230_778 # idk how 400_000 + 230_778 = 620_536
    for line in data_reader:
        words = get_features(line[1])
        test_data[words] = line[2]
    # print(len(all_data)) # 620_536


read_data() # features
classifier = train(train_data)

# create dictionary with default value equal to (0, 0)
# it is a dict of type string: (int, int)
# where string is langueage name, and (int, int) is how
# predictions are correct of all programs amount
# if we have 300 programs in python, and we predict it to be python for 230 times
# it will be {'PYTH': (230, 300)}
analysis = defaultdict(lambda: (0, 0))

print('testing...')
size = len(test_data)
i = 0
for test_feat, lang in test_data.items():
    ans = classify(classifier, test_feat)
    # match - how many times we predicted the language
    # whole - how many programs in the language we met indeed
    match, whole = analysis[lang]
    # match is how many matches classifier got
    # whole is how many times classifier met the lang
    # if we predicted it right +1 otherwise do nothing
    # in both cases increase how many times we met programs in the language
    analysis[lang] = (match + 1 if ans == lang else match, whole + 1)
    if i % 1000 == 0:
        print(str(i) + ' of ' + str(size) + ' ready.')
    i += 1
print('ready.')

# pring results
# <LANG>: <we got it right> of <how many programs in the language actually> 
# e.g. JAVA: 1234 of 4312
for k, v in analysis.items():
    print(k + ': ' + str(v[0]) + ' of ' + str(v[1]))

