import os
import sys

import pycrfsuite

import hw2_corpus_tool as reading_tool


def word2features(sent, i):
    word = sent[i][0]
    pos = sent[i][1]
    features = []
    features.extend([
        'TOKEN=' + word,
        'POS=' + pos
    ])

    return features


def sent2features(sent, features):
    for i in range(len(sent)):
        features.extend(word2features(sent, i))
    return features
    # return [word2features(sent, i, features) for i in range(len(sent))]


def process_dialog_utterance(pos, features):
    return sent2features(pos, features)


def get_Data_for_Tagger(file_names):
    X = []
    Y = []
    for each_dialog in file_names:
        #dialog_utterances = (reading_tool.get_utterances_from_filename(file))
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

            count += 1
            if each_utterance[2] is not None:
                feature_vector.append(process_dialog_utterance(each_utterance[2], features))
            else:
                features.append("")
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
