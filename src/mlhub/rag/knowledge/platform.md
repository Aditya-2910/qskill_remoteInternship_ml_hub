# About the ML Hub platform

## Architecture
ML Hub has four layers. Training scripts fit and evaluate three scikit-learn pipelines and save them to the models folder together with metadata and plots. A FastAPI backend loads the saved pipelines and exposes prediction endpoints under /v1/predict for iris, spam and house. A chat assistant, powered by a local language model served by Ollama, can call those prediction endpoints as tools and retrieve explanations from the knowledge base. A Streamlit web app talks to the FastAPI backend over HTTP.

## The three models
The iris model classifies a flower into setosa, versicolor or virginica from four measurements. The spam model classifies a text message as spam or ham using TF-IDF features. The house model predicts the median house value of a California census block group from eight features.

## Experiment tracking
Every training run is logged to MLflow with parameters, cross-validation scores for each candidate model, test metrics, plots and the fitted pipeline, which is also added to the MLflow model registry. Run the MLflow UI to compare runs.

## Monitoring
Each prediction is logged with its latency. For the tabular models the inputs are logged, and for spam only the message length and word count are logged, never the text. A drift check compares recent inputs with a reference sample saved at training time using a Kolmogorov-Smirnov test, and flags features whose distribution has shifted. Drift is a signal that the model may need to be retrained on newer data.

## Retraining
Run the training command again to train all models on fresh data. New model files are written to the models folder and the API picks them up when it is restarted or when the reload endpoint is called.
