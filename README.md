# Artificial-Intelligence-based-customer-sentiment-analysis
Build a deep learning model for customer sentiment analysis. This project involves analyzing customer behavior based on key performance indicators (KPIs), such as:

Products Bought: Data on what customers purchased.
Last Engagement: Timing and nature of the last interaction with our company.
Customer Reviews: Textual feedback and ratings from customers.
Other Behavioral/Transactional Data: Additional relevant datasets.
The goal is to create a sentiment analysis model that delivers actionable insights, enabling us to enhance customer engagement and satisfaction.
=============
To build a deep learning model for customer sentiment analysis based on the provided key performance indicators (KPIs) such as products bought, last engagement, customer reviews, and other behavioral/transactional data, we can follow a structured approach. Here’s how to implement a deep learning model in Python using TensorFlow/Keras, and also preprocessing the necessary data for sentiment analysis.
Steps:

    Data Collection & Preprocessing:
        Collect and clean data on products bought, engagement timings, customer reviews, and other behavioral data.
        Use embeddings for textual data (reviews).
        Normalize numerical features like purchase amounts, last engagement date, etc.

    Model Architecture:
        A deep learning model combining both text-based features (via embeddings) and structured data (e.g., product details, engagement history).

    Sentiment Analysis:
        Use TensorFlow and Keras to define the architecture for sentiment classification.
        Use TextVectorization for processing text and embeddings.
        Use Sequential models for building layers that process both numerical and text data.

Python Code:

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Concatenate, Input, Dropout, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

# Example Dataframe (Assumed structure)
data = {
    'products_bought': ['Laptop', 'Smartphone', 'Headphones', 'Tablet', 'Smartwatch'],
    'last_engagement': ['2024-01-01', '2024-01-05', '2024-01-08', '2024-02-01', '2024-02-05'],
    'customer_reviews': ['Great product!', 'Very good phone, fast.', 'Good sound quality.', 'Nice tablet for work.', 'Amazing smartwatch!'],
    'amount_spent': [1000, 800, 150, 300, 250],
    'sentiment': [1, 1, 0, 1, 1]  # Sentiment: 1 for Positive, 0 for Negative
}

df = pd.DataFrame(data)

# Preprocess text data for reviews (Tokenization and padding)
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['customer_reviews'])
sequences = tokenizer.texts_to_sequences(df['customer_reviews'])
max_sequence_length = max([len(seq) for seq in sequences])
X_text = pad_sequences(sequences, maxlen=max_sequence_length)

# Preprocess numerical data (Products Bought Amount, Engagement Data)
scaler = StandardScaler()
X_numerical = scaler.fit_transform(df[['amount_spent']])

# Date feature transformation (Convert last engagement to days since last interaction)
current_date = pd.to_datetime('2024-02-10')
df['last_engagement'] = pd.to_datetime(df['last_engagement'])
df['days_since_last_engagement'] = (current_date - df['last_engagement']).dt.days
X_numerical = np.concatenate([X_numerical, df[['days_since_last_engagement']].values], axis=1)

# Split dataset
X_text_train, X_text_test, X_numerical_train, X_numerical_test, y_train, y_test = train_test_split(
    X_text, X_numerical, df['sentiment'], test_size=0.2, random_state=42
)

# Define the deep learning model

# Text Input Model (LSTM for customer reviews)
text_input = Input(shape=(max_sequence_length,), name='text_input')
embedding_layer = Embedding(input_dim=5000, output_dim=128)(text_input)
lstm_layer = LSTM(128, return_sequences=False)(embedding_layer)
text_output = Dense(64, activation='relu')(lstm_layer)

# Numerical Input Model (Amount spent and Days since last engagement)
numerical_input = Input(shape=(X_numerical_train.shape[1],), name='numerical_input')
numerical_output = Dense(64, activation='relu')(numerical_input)
numerical_output = Dropout(0.2)(numerical_output)

# Combine both models
combined = Concatenate()([text_output, numerical_output])
dense_layer = Dense(128, activation='relu')(combined)
dropout_layer = Dropout(0.5)(dense_layer)
final_output = Dense(1, activation='sigmoid')(dropout_layer)  # Binary sentiment output (0/1)

# Compile the model
model = Model(inputs=[text_input, numerical_input], outputs=final_output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([X_text_train, X_numerical_train], y_train, epochs=10, batch_size=32, validation_data=([X_text_test, X_numerical_test], y_test))

# Evaluate the model
loss, accuracy = model.evaluate([X_text_test, X_numerical_test], y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Make predictions
predictions = model.predict([X_text_test, X_numerical_test])

# Display predictions for review
print("Predictions (Sentiment):", predictions)

Explanation of the Code:

    Data Preprocessing:
        Text data (customer reviews): We preprocess the reviews using Tokenizer and pad_sequences to convert textual data into numerical format that the deep learning model can understand.
        Numerical data (amount spent, engagement data): We scale numerical data using StandardScaler and convert the last engagement date into a numeric feature representing days since the last interaction.

    Model Architecture:
        Text Input Model: The customer reviews are processed through an embedding layer followed by an LSTM (Long Short-Term Memory) layer. This allows the model to capture semantic and syntactic features from the text data.
        Numerical Input Model: The numerical features like amount spent and days since last engagement are processed using a Dense layer.
        Combined Model: The outputs of the text and numerical models are concatenated, followed by a dense layer and dropout for regularization, and the final output layer is a sigmoid function to predict sentiment (positive/negative).

    Model Training & Evaluation:
        The model is compiled using the Adam optimizer and binary cross-entropy loss (since this is a binary classification problem).
        The model is trained for 10 epochs, and we evaluate the performance using accuracy.

    Prediction:
        After training, the model is used to predict sentiment on the test set, with results returned as probabilities.

Key Notes:

    Data Representation: The model works with both text (customer reviews) and numerical data (behavioral metrics like purchase amounts and engagement).
    Text Feature Processing: We use an embedding layer and LSTM to extract semantic information from the reviews, making the model suitable for analyzing customer sentiment from natural language.
    Model Flexibility: You can adapt this model to include additional KPIs or use more advanced features, such as incorporating additional transactional data or customer interaction metrics.

Future Enhancements:

    Textual Sentiment Analysis: Incorporate a pre-trained language model like BERT for better understanding of complex sentiment in reviews.
    Feature Engineering: Add more features based on customer demographics or interaction history.
    Model Hyperparameter Tuning: Use grid search or random search to optimize model parameters such as the learning rate, number of layers, and units in LSTM.

This model can help gain actionable insights into customer behavior, ultimately enabling improved customer engagement and satisfaction.
