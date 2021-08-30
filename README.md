This internship is carried out as part of the engineering cycle training (second year) of the Polytechnic School of the Université Côte d'Azur.
***
# Active Learning for Deep Neural Networks

## Presentation
This internship was carried out with the Python language.

### Purpose:
The topic of the internship deals with active learning. It is a machine learning paradigm that relies on defining a strategy to train a model from as little annotated data as possible. A prediction model is first trained from a very small initial training set. Then an iterative strategy for selecting non-annotated data is defined so that the annotation of this data and then its addition to the training set leads to a better model once re-trained.
The challenge is to define a selection strategy of non-annotated data such as only a few new data is annotated at each iteration.

Previous researches had proposed several selection strategies that proved to greatly improve performance for deep neural networks. In particular, one of the strategies relied on the generation of **adversarial examples** using **adversarial attacks**.

The main objective of my study was to conduct several experiments where a different adversarial attack was used at each iteration in the process. I was also tasked to evaluate the performance of the active learning models and their robustness against adversarial attacks.
