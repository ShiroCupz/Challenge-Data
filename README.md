# Drug-related questions classification by Posos

## Dates

From Jan. 1, 2019 to Jan. 1, 2020

## Challenge context

More than 6,000 different drugs are commercialized in France. Not only patients but also healthcare professionals remain unable to use them correctly when they do not have immediate access to appropriate information they may seek. Medicines remain responsible for more than 144,000 hospitalizations every year in France while in the United States, 1.5 million people face drug misuse every year. Questions about drug prescription, dispensation or use should never remain unanswered. What is particularly interesting to understand about drug queries is what information people expect to get as an answer: for instance, associated side effects, ingredients or contra-indications. Millions of queries are asked every year about drugs. There is a limited number of query types but the same question could be asked in many ways. Therefore, understanding what information people expect to get when asking a question is a great challenge.

## Challenge goals

The goal of Posos challenge is to predict for each question the associated intent.

## Data description

The input data `input_train.csv` is a list of questions written in French. Each line is constituted with a unique ID followed by one question.

In the input file, the first line is the header. Columns are separated by `,` characters. They correspond to: ID: line number. It relates to the line number in the output file. question: the question whose intent has to be predicted.

Here below is an example of the input file content:

1) `Est-ce qu'il existe une forme adaptée aux enfants de 5ans du Micropakine ?`
2) `laroxyl à doses faibles pour le stress ?`
3) `mon psy me dit de prendre 50mg de sertraline le matin et 50 mg de sertraline le soir. Peut-on prendre 100mg soit le matin ou à midi?`

The output file `output_train.csv` contains the intent associated to each ID. The first line is the header; columns are always separated by `,` characters. They correspond to the line number and the intent identification number.

| ID | intention |
| - | - |
| 1 | 44 |
| 2 | 31 |
| 3 | 42 |

A list of 50 different intents has been predefined. We anonymized them for this challenge converting intents to identification numbers.

Intents are homogeneously distributed between training and test files.

From the `input_test.csv file`, competitors will provide an `output_test.csv` file with the same format as the `output_train.csv` file. They will predict for each line number, the associated intent number.

## Benchmark description

Our reference solution is a Convolutional Neural Network applied at word scale.

- The words of the sentence are first vectorized in dimension 128.
- Filters of size 3, 4 and 5 are then applied (128 filters per size). For each of these filters, only the maximum value during the sentence is kept (max pooling over the entire length of the sentence).
- A dropout is then applied, during training, with a probability 0.5.
- A last fully-connected layer is finally applied, each neuron corresponding to an intention.
- The likelihood of each intention is estimated by softmax.
- The optimized loss function is the cross entropy between this estimate and the target distribution (which is a dirac).

This system, trained from start to finish, vocabulary vectorization included, gives a score of 82% on a set of 10,000 examples. However, it is suggested to pre-train the vectorization of words on a more important corpus (like word2vec), or to pre-process questions to detect specific properties such as drug names.

## Public metric

accuracy_score from scikit-learn : scikit-learn metrics
