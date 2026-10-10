# Machine learning concepts used in this project

## Train and test split
The data is split into a training set, used to fit the model, and a test set kept aside to measure how well the model works on data it has never seen. Classification tasks use a stratified split, which keeps the class proportions the same in both sets.

## Cross-validation and model selection
Cross-validation repeats the train and validate cycle on different folds of the training set. This project cross-validates every candidate model on the training data only, picks the best one, and only then measures it on the test set. Choosing a model using the test set would leak information from it and make the final score look better than it really is.

## Data leakage
Leakage happens when information from outside the training data influences the model. A typical example is fitting a scaler or a TF-IDF vectoriser on the whole dataset before splitting. Putting preprocessing inside a scikit-learn Pipeline means it is fit on the training folds only.

## Classification metrics
Accuracy is the fraction of correct predictions. Precision is the fraction of positive predictions that are correct. Recall is the fraction of real positives that are found. F1 is the harmonic mean of precision and recall. A confusion matrix shows how often each actual class was predicted as each class, which reveals which classes get mixed up.

## Regression metrics
Mean Squared Error penalises large errors heavily. RMSE has the units of the target. Mean Absolute Error is the average absolute error and is easier to interpret. R squared is the share of variance explained.

## Retrieval augmented generation
RAG lets a language model answer using documents it has not been trained on. The documents are split into chunks and converted into embedding vectors. For each question the most similar chunks are retrieved and added to the prompt, so the answer is grounded in the retrieved text and can cite its sources.

## Tool calling
Tool calling lets a language model request that the application run a function, such as a prediction. The model outputs the function name and arguments, the application runs it and returns the result, and the model then writes the final answer using that result. The model never computes the prediction itself.
