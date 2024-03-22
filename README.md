# Fake-News-Detector

This Streamlit application is designed to verify whether a given news text is true or false. Let's break down the functionality and implementation details:

### Libraries Used
- **Streamlit**: For building the web application interface.
- **NumPy, Pandas**: For data manipulation.
- **TensorFlow and Keras**: For building and training the deep learning model.
- **scikit-learn**: For preprocessing tasks.
- **Gensim**: For working with word embeddings.

### Data Preparation
- The code reads a dataset from a CSV file ("news.csv") containing news articles along with their labels.
- It preprocesses the data by dropping unnecessary columns and encoding the labels.

### Model Building and Training
- The app prompts the user to input text for verification.
- If the input text is provided, the code preprocesses it and builds a deep learning model for text classification.
- The model architecture consists of an embedding layer, followed by dropout, convolutional, max pooling, LSTM, and dense layers.
- Pre-trained GloVe word embeddings are used for the embedding layer.
- The model is trained using a portion of the provided dataset.
- The training history is stored for later analysis.

### Text Verification
- After training, the user's input text is tokenized and padded to match the model input shape.
- The trained model then predicts whether the input news is true or false based on the provided text.
- The result is displayed in the Streamlit interface.

### Functionality
- The application provides a simple interface for users to input news text and receive a prediction regarding its authenticity.
- It leverages a deep learning model trained on a labeled dataset to make these predictions.
- Users can use this tool to quickly assess the credibility of news articles they encounter online.
