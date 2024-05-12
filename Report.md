# Introduction:
This project is based upon the sentiment anaylysis in which we have trained an AI model which reads the datasets of textual messages which are from different social media platforms. The model tries to identify whether the tone of message is either positive or negative. For this project we used 5000 messages as our dataset for training.
# Preprocessing Used:
These are the methods used inorder to remove uneccesary parts from the messages like the mentions (@), tags(#), Emojis, numeric characters, punctuations, URLs, special characters and even non english text. Following are the most important preprocessing techniques involved:
* ## Removal of stopwords:
  Stopwords are the words such as "and", "the", "but" are removed as it becomes easy for the model to interpret and understand the important details related to that text
* ## Usage of stemming:
  Stemming basically refers to converting the words into their first form. For example in case of running, it becomes run. This is also done for making the model easily make the sentiments.
* ## Using chat words' full form:
  Basically the acronyms used on a daily basis in social media like "lol" and "idk" are replaced by thier full form for better understanding the text. For that we have used a dictionary of these chat words to replace them everytime when we encounter them in the preprocessing with replace_char_words function[1].
* ## Making the message in lower case:
  Making the message in lower case makes it easy to read the dataset for the model and produce some valueable results[1].
# Libararies used
* Pandas
* string
* Numpy
* nltk
* sklearn
* Tensorflow
* seaborn
* matplotlib
# Usage of data Visualization:
We have used data visualization inorder to improve the efficiency of the words in the dataset, identify the outliers in the data and to reveal the data patterns and relations of those specific words with the sentiments we have used. [2]
# Machine Learning algorithms used:
The input of the dataset will be given to CountVectorizer and then to TfIdfTransformer and then used in these algorithms used as follows:
* ## SVC algorithm:
  We used the SVC algorithm which is basically a machine learning algorithm which is  is a particular method used for classification tasks. It operates by identifying the most effective hyperplane to divide data points into distinct categories. SVC, as an implementation of SVM, transforms data points into a higher-dimensional space and then determines the most suitable hyperplane to classify the data. The value of the confusion matrix are as follows. The value of True positive values: 2386, the value of False Positive values: 428 , the value of False Negative values: 335 and the value of True Negative values: 2332. The precision for the positive messages is 88% and recall is 85%. On the other hand the precision for the negative messages are 84% and recall is 87%.
* ## KNN algorithm:
  The second algorithm which we used is KNN. This algorithm relies on measuring proximity to determine the similarity between a data point and a set of training data, which it then uses for making predictions. KNN is considered a non-parametric method and is commonly utilized for both classification and regression purposes, along with handling missing values. The values of the confusion matrix are are as follows. The value of True positive values: 2486, the value of False Positive values: 328 , the value of False Negative values: 669 and the value of True Negative values: 1998. The precision for the positive messages is 79% and recall is 88%. On the other hand the precision for the negative messages are 86% and recall is 75%.
* ## RFC algorithm:
  This is the algorithm  an ensemble learning method that constructs multiple decision trees by sampling from the original dataset and randomly selecting features at each node. It then aggregates the predictions of these trees to make the final prediction, offering robustness to overfitting, suitability for high-dimensional data, and feature importance estimation. The values of the confusion matrix are are as follows. The value of True positive values: 2400, the value of False Positive values: 414 , the value of False Negative values: 383 and the value of True Negative values: 2284. The precision for the positive messages is 86% and recall is 85%. On the other hand the precision for the negative messages are 85% and recall is 86%.
* ## Multinomial NB:
  This is a classification algorithm in machine learning which employs a probabilistic strategy to categorize data by analyzing its statistical characteristics. Widely utilized in Natural Language Processing (NLP), MNB is favored for predicting labels of textual data like emails or articles, offering a popular method for supervised learning tasks. The values of the confusion matrix are are as follows. The value of True positive values: 2508, the value of False Positive values: 306 , the value of False Negative values: 483 and the value of True Negative values: 2184. The precision for the positive messages is 84% and recall is 89%. On the other hand the precision for the negative messages are 88% and recall is 82%.
# Model used:
We built a Bidirectional LSTM Model which is "sequencial 2" on which we trained the messages inorder to get their sentiments[1]. LSTM is a recurrent neural network variant, addresses the shortcomings of traditional RNNs by employing gated cells to retain long-term dependencies, making it effective in tasks like language modeling and time series prediction. Each of the values are split on how much wants to be trained and tested. We defined the Model, added an Embedding Layer, added a Bidirectional GRU Layer, added Batch Normalization Layer, added Dropout Regularization, added a Dense Layer with ReLU Activation, added nnother Dropout Layer and finally added the Output Layer. The detailed implementation for this is done in our python code for this.
# Citations:
* https://www.kaggle.com/code/samanfatima7/crushing-it-bilstm-rnn-delivers-94-accuracy [1]
* https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset/code [2]
# Accuracy graphs:
! [Accuracy bar charts](Accuracy_bar_chart.jpg)
! [Accuray line graph](Accuracy_line_graph.jpg)
