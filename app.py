from flask import Flask, jsonify, make_response, request
import pandas as pd
import joblib
import threading

from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('bert-base-nli-mean-tokens')

# training data preprocessing
# - remove \n \t and convert to list of sentences


def preprocessing(passage):
    """ Input -> string
        Output -> processed list"""
    # print("passage: ", passage)
    passage_sent_list = passage.replace('\n', '').replace(
        '\t', '').replace(',', '').split('.')
    return passage_sent_list


def load_question_file(path_to_question_file):
    """ Input -> Path to Known Answers File
        Output -> list of processed sentences """
    with open(path_to_question_file, 'r', encoding="utf-8") as f:
        whole_text = f.read()
        sentence_list = preprocessing(whole_text)
    return sentence_list


def score(original_embedding, test_answer, model):
    if type(test_answer) == 'str':
        test_preprocessed = preprocessing(test_answer)
    else:
        test_preprocessed = test_answer
    test_embedding = model.encode(test_preprocessed)

    score = 0
    for i in test_embedding:
        sim_arr = cosine_similarity(
            [i],
            original_embedding[:]
        )
        # index = np.argmax(sim_arr)
        embedding_score = np.average(sim_arr)
        score += embedding_score

    return score/len(test_embedding)

# Load QUestions


questions = []

for i in range(51):
    file_name = 'q'+str(i+1) + '.txt'
    print('uplaoding file: ', file_name)
    questions.append(load_question_file(file_name))

# # model training / embeddings generation

embeddings = []

for j in range(51):
    file = questions[j]
    print(" Training on file: ", j + 1)
    embeddings.append(model.encode(file))


# # Declare a Flask app
app = Flask(__name__)

# # Main function here
# # ------------------


@app.route('/')
def Hello():
    return "I'm ALive"


@app.route('/predict', methods=['POST'])
def bestcv():
    id = int(request.get_json()['ques_id'])
    embed = embeddings[id]
    print("embeddings", embed)
    text = request.get_json()['text']
    print("text:", text)
    sentence = preprocessing(text)
    print("sentence :", sentence)
    Score = score(embed, sentence, model) * 200
    print("Score", Score)
    print("Score", Score)
    return jsonify({'result': Score})


# # Running the app
# if __name__ == '__main__':
#     app.run(debug=True)

threading.Thread(target=app.run, kwargs={
                 'host': '0.0.0.0', 'port': 5000}).start()
