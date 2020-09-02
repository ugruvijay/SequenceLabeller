import os
import sys
import string
import pycrfsuite

import hw2_corpus_tool as reading_tool

punctuations = string.punctuation

def word2features(sent, i):
    word = sent[i][0]
    pos = sent[i][1]
    features = []
    features.extend([
        'TOKEN=' + word.lower(),
        'POS=' + pos,
        'IS_PUNCT=' + str(word in punctuations),
        'IS_TITLE=' + str(word.istitle())
    ])
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.extend([
            '-1:TOKEN_lower=' + word1.lower(),
            '-1:TOKEN_IS_PUNCT=' + str(word1 in punctuations),
            '-1:TOKEN_istitle=%s' % word1.istitle(),
            '-1:TOKEN_isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1
        ])
    else:
        features.append('BOS')

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.extend([
            '+1:TOKEN_lower=' + word1.lower(),
            '+1:TOKEN_IS_PUNCT=' + str(word1 in punctuations),
            '+1:TOKEN_istitle=%s' % word1.istitle(),
            '+1:TOKEN_isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1
        ])
    else:
        features.append('EOS')

    return features


def sent2features(sent, features):
    for i in range(len(sent)):
        features.extend(word2features(sent, i))
    return features


def process_dialog_utterance(pos, features):
    return sent2features(pos, features)


def get_Data_for_Tagger(file_names):
    X = []
    Y = []
    for each_dialog in file_names:
        count = 0

        act_tags = []
        feature_vector = []

        for each_utterance in each_dialog:
            act_tag = each_utterance[0]
            if count == 0:
                previous_speaker = "not-applicable"
                features = ['FIRST_UTTERANCE']
            else:
                previous_speaker = each_dialog[count - 1][1]
                features = []

            if previous_speaker == each_utterance[1]:
                features.append("SAME_SPEAKER")
            else:
                features.append("PREVIOUS_SPEAKER=" + previous_speaker)
                
            if count == len(each_dialog) - 1:
                features.append("LAST_UTTERANCE")

            count += 1
            if each_utterance[2] is not None:
                feature_vector.append(process_dialog_utterance(each_utterance[2], features))
            else:
                features.append("NON_VERBAL")
                feature_vector.append(features)

            if act_tag is None:
                act_tags.append("UNKNOWN")
            else:
                act_tags.append(act_tag)

        X.append(feature_vector)
        Y.append(act_tags)

    return [X, Y]


def predict_act_tag(x_test, y_test):
    output_file = open(sys.argv[3], "w")
    correct_predictions = 0
    predicted_tag = []
    for each_dialog in x_test:
        predicted_tag.append(tagger.tag(each_dialog))

    count = 0
    for prediction, true_value in list(zip(predicted_tag, y_test)):
        for i in range(len(prediction)):
            output_file.write(prediction[i] + "\n")
            if prediction[i] == true_value[i]:
                correct_predictions += 1
            count += 1
        output_file.write("\n")

    print("Accuracy = ", correct_predictions / count)


if __name__ == "__main__":
    training_set = list(reading_tool.get_data(sys.argv[1]))
    testing_set = list(reading_tool.get_data(sys.argv[2]))

    X_Train, Y_Train = get_Data_for_Tagger(training_set)
    X_Test, Y_Test = get_Data_for_Tagger(testing_set)


    trainer = pycrfsuite.Trainer(verbose=False)

    for i in range(len(X_Train)):
        trainer.append(X_Train[i], Y_Train[i])

    trainer.set_params({
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier
        'feature.possible_transitions': True
    })

    trainer.train('model.crfsuite')

    tagger = pycrfsuite.Tagger()
    tagger.open('model.crfsuite')

    predict_act_tag(X_Test, Y_Test)
