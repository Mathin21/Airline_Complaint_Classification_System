# Install Required Libraries
# pip install tensorflow pandas numpy matplotlib seaborn
# pip install scikit-learn nltk imbalanced-learn
# pip install nlpaug focal-loss

# Import Necessary Libraries
import re
import random
import numpy as np
import joblib

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, Bidirectional, GRU, SpatialDropout1D, Input, GlobalMaxPooling1D

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from focal_loss import SparseCategoricalFocalLoss

# NLTK
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nlpaug.augmenter.word as naw

# synonym augmenter
augmenter = naw.SynonymAug(aug_src='wordnet')


# Load the dataset
data = pd.read_csv("Cleaned_Indigo_Format.csv", encoding="latin1")
print("Original Shape:", data.shape)


# Remove NULL values and Duplicates
data = data.dropna()
data = data.drop_duplicates(subset=["Complaint Text", "Category"])
data = data.drop_duplicates(subset=["Complaint Text"])
data = data.reset_index(drop=True)
print("Cleaned Shape:", data.shape)


# Text Preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()

    # remove urls
    text = re.sub(r"http\S+|www\S+", "", text)

    # remove mentions
    text = re.sub(r"@\w+", "", text)

    # remove special chars
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()

    # remove stopwords + lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 2
    ]
    return " ".join(words)

data["clean_text"] = data["Complaint Text"].apply(clean_text)


# Data Augmentation for minority classes only
class_counts = data["Category"].value_counts()
print("\nClass Distribution:")
print(class_counts)

max_samples = class_counts.max()
augmented_rows = []
for category in class_counts.index:
    category_df = data[data["Category"] == category]
    current_count = len(category_df)
    needed = max_samples - current_count
    print(f"\n{category}")
    print(f"Current: {current_count}")
    print(f"Need Augmentation: {needed}")

    if needed > 0:
        samples = category_df.sample(n=needed, replace=True, random_state=42)
        for _, row in samples.iterrows():
            text = row["clean_text"]
            try:
                augmented_text = augmenter.augment(text)
                if isinstance(augmented_text, list):
                    augmented_text = augmented_text[0]
                augmented_rows.append({
                    "clean_text": augmented_text,
                    "Category": category
                })
            except:
                pass

# create augmented dataframe
aug_df = pd.DataFrame(augmented_rows)

# merge original + augmented
final_data = pd.concat([data[["clean_text", "Category"]], aug_df], ignore_index=True)
print("\nFinal Dataset Shape:", final_data.shape)
print("\nFinal Class Distribution:")
print(final_data["Category"].value_counts())


# Split the data
X = final_data["clean_text"]
y = final_data["Category"]
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# Label Encoding
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_text)
y_test = label_encoder.transform(y_test_text)
num_classes = len(label_encoder.classes_)
print("\nNumber of Classes:", num_classes)


# Tokenization
VOCAB_SIZE = 20000
MAX_LEN = 180
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)
X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)
X_train = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
X_test = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")


# Load the Glove Embeddings
embedding_dim = 100
embedding_index = {}
with open("glove.6B.100d.txt", encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embedding_index[word] = coefs
print("\nLoaded GloVe Words:", len(embedding_index))


# Create the embedding matrix
word_index = tokenizer.word_index
vocab_size = min(VOCAB_SIZE, len(word_index) + 1)
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    if i < vocab_size:
        vector = embedding_index.get(word)
        if vector is not None:
            embedding_matrix[i] = vector
print("Embedding Matrix Shape:", embedding_matrix.shape)


# Class Weights
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))
print("\nClass Weights:")
print(class_weights)


# Build the model
model = Sequential([
    Input(shape=(MAX_LEN,)),
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=True),
    SpatialDropout1D(0.2),
    Bidirectional(GRU(128, return_sequences=True, dropout=0.2)),
    GlobalMaxPooling1D(),
    Dense(128,activation="relu"),
    Dropout(0.3),
    Dense(64,activation="relu"),
    Dropout(0.2),
    Dense(num_classes, activation="softmax")
])

# Complile the model
optimizer = Adam(
    learning_rate=0.0003
)
model.compile(
    optimizer=optimizer,
    loss=SparseCategoricalFocalLoss(gamma=2),
    metrics=["accuracy"]
)
model.summary()


# CallBacks
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    verbose=1
)


# Train the model
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=25,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)


# Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print("\n===================================")
print("Test Accuracy:", round(accuracy, 4))
print("===================================")


# Predictions
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)


# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()


# Training and Validation Loss
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


# Test Custom Predictions
test_examples = [
    "My flight was delayed for 5 hours",
    "Customer support ignored my refund request",
    "My baggage was lost at the airport",
    "The flight attendant was rude",
    "Long queue at check in counter",
    "Seat booking failed and payment deducted",
    "Luggage got damaged during travel"
]

print("\n===================================")
print("Custom Predictions")
print("===================================")

for text in test_examples:
    clean = clean_text(text)
    seq = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    pred_prob = model.predict(padded, verbose=0)
    pred_id = np.argmax(pred_prob)
    confidence = pred_prob[0][pred_id]
    label = label_encoder.inverse_transform([pred_id])[0]

    print("\n-----------------------------------")
    print("Complaint :", text)
    print("Predicted :", label)
    print("Confidence:", round(float(confidence), 4))

# Save the model
model.save("airline_complaint_classifier.keras")
print("\nModel Saved Successfully!")

joblib.dump(tokenizer, "tokenizer.pkl")
print("\nTokenizer Saved Sucessfully!")

joblib.dump(label_encoder, "label_encoder.pkl")
print("\nLabel Encoder Saved Successfully!")

model.save_weights("airline_complaint_classifier.weights.h5")
print("\nModel Weights Saved Successfully!")