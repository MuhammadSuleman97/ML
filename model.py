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
    print("passage: ", passage)
    passage_sent_list = passage.replace('\n', '').replace(
        '\t', '').replace(',', '').split('.')
    return passage_sent_list


def load_question_file(path_to_question_file):
    """ Input -> Path to Known Answers File
        Output -> list of processed sentences """
    with open(path_to_question_file, 'r') as f:
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


# load the questions
q1 = load_question_file('q1-about_yourself.txt')
q2 = load_question_file('q2-resume.txt')
q3 = load_question_file('q3.txt')

q4 = load_question_file('q4.txt')
q5 = load_question_file('q5.txt')
q6 = load_question_file('q6.txt')
q7 = load_question_file('q7.txt')
q8 = load_question_file('q8.txt')
q9 = load_question_file('q9.txt')
q10 = load_question_file('q10.txt')
q11 = load_question_file('q11.txt')
q12 = load_question_file('q12.txt')
q13 = load_question_file('q13.txt')
q14 = load_question_file('q14.txt')
q15 = load_question_file('q15.txt')
q16 = load_question_file('q16.txt')
q17 = load_question_file('q17.txt')
q18 = load_question_file('q18.txt')
q19 = load_question_file('q19.txt')
q20 = load_question_file('q20.txt')
q21 = load_question_file('q21.txt')
q22 = load_question_file('q22.txt')
q23 = load_question_file('q23.txt')
q24 = load_question_file('q24.txt')
q25 = load_question_file('q25.txt')
q26 = load_question_file('q26.txt')
q27 = load_question_file('q27.txt')
q28 = load_question_file('q28.txt')
q29 = load_question_file('q29.txt')
q30 = load_question_file('q30.txt')
q31 = load_question_file('q31.txt')
q32 = load_question_file('q32.txt')
q33 = load_question_file('q33.txt')
q34 = load_question_file('q34.txt')
q35 = load_question_file('q35.txt')
q36 = load_question_file('q36.txt')
q37 = load_question_file('q37.txt')
q38 = load_question_file('q38.txt')
q39 = load_question_file('q39.txt')
q40 = load_question_file('q40.txt')
q41 = load_question_file('q41.txt')
q42 = load_question_file('q42.txt')
q43 = load_question_file('q43.txt')
q44 = load_question_file('q44.txt')
q45 = load_question_file('q45.txt')
q46 = load_question_file('q46.txt')
q47 = load_question_file('q47.txt')
q48 = load_question_file('q48.txt')
q49 = load_question_file('q49.txt')
q50 = load_question_file('q50.txt')
q51 = load_question_file('q51.txt')


# model training / embeddings generation

q1_embeddings = model.encode(q1)
q2_embeddings = model.encode(q2)
q3_embeddings = model.encode(q3)
q4_embeddings = model.encode(q4)
q5_embeddings = model.encode(q5)
q6_embeddings = model.encode(q6)
q7_embeddings = model.encode(q7)
q8_embeddings = model.encode(q8)
q9_embeddings = model.encode(q9)
q10_embeddings = model.encode(q10)
q11_embeddings = model.encode(q11)
q12_embeddings = model.encode(q12)
q13_embeddings = model.encode(q13)
q14_embeddings = model.encode(q14)
q15_embeddings = model.encode(q15)
q16_embeddings = model.encode(q16)
q17_embeddings = model.encode(q17)
q18_embeddings = model.encode(q18)
q19_embeddings = model.encode(q19)
q20_embeddings = model.encode(q20)
q21_embeddings = model.encode(q21)
q22_embeddings = model.encode(q22)
q23_embeddings = model.encode(q23)
q24_embeddings = model.encode(q24)
q25_embeddings = model.encode(q25)
q26_embeddings = model.encode(q26)
q27_embeddings = model.encode(q27)
q28_embeddings = model.encode(q28)
q29_embeddings = model.encode(q29)
q30_embeddings = model.encode(q30)
q31_embeddings = model.encode(q31)
q32_embeddings = model.encode(q32)
q33_embeddings = model.encode(q33)
q34_embeddings = model.encode(q34)
q35_embeddings = model.encode(q35)
q36_embeddings = model.encode(q36)
q37_embeddings = model.encode(q37)
q38_embeddings = model.encode(q38)
q39_embeddings = model.encode(q39)
q40_embeddings = model.encode(q40)
q41_embeddings = model.encode(q41)
q42_embeddings = model.encode(q42)
q43_embeddings = model.encode(q43)
q44_embeddings = model.encode(q44)
q45_embeddings = model.encode(q45)
q46_embeddings = model.encode(q46)
q47_embeddings = model.encode(q47)
q48_embeddings = model.encode(q48)
q49_embeddings = model.encode(q49)
q50_embeddings = model.encode(q50)
q51_embeddings = model.encode(q51)
embeddings = [q1_embeddings, q2_embeddings, q3_embeddings, q4_embeddings, q5_embeddings, q6_embeddings, q7_embeddings, q8_embeddings, q9_embeddings, q10_embeddings, q11_embeddings, q12_embeddings, q13_embeddings, q14_embeddings, q15_embeddings, q16_embeddings, q17_embeddings, q18_embeddings, q19_embeddings, q20_embeddings, q21_embeddings, q22_embeddings, q23_embeddings, q24_embeddings, q25_embeddings,
              q26_embeddings, q27_embeddings, q28_embeddings, q29_embeddings, q30_embeddings, q31_embeddings, q32_embeddings, q33_embeddings, q34_embeddings, q35_embeddings, q36_embeddings, q37_embeddings, q38_embeddings, q39_embeddings, q40_embeddings, q41_embeddings, q42_embeddings, q43_embeddings, q44_embeddings, q45_embeddings, q46_embeddings, q47_embeddings, q48_embeddings, q49_embeddings, q50_embeddings, q51_embeddings]
a = embeddings[1]
print(a)
print(embeddings[1])
