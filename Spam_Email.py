# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# Load dataset (assume 'emails.csv' has 'text' and 'label' columns)
df = pd.read_csv('emails.csv')

# Separate data into features (X) and labels (y)
X = df['text']
y = df['label']

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize vectorizers for Bag-of-Words (Count) and TF-IDF
vectorizer_bow = CountVectorizer()  # Bag-of-Words model
vectorizer_tfidf = TfidfVectorizer()  # TF-IDF model

# Transform the training and test data using the vectorizers
X_train_bow = vectorizer_bow.fit_transform(X_train)
X_test_bow = vectorizer_bow.transform(X_test)

X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
X_test_tfidf = vectorizer_tfidf.transform(X_test)

# Handle class imbalance by calculating class weights (important for SVM)
class_weights = compute_class_weight('balanced', classes=y.unique(), y=y)
class_weight_dict = dict(zip(y.unique(), class_weights))

# Initialize the models
nb_bow = MultinomialNB()  # Naive Bayes with Bag-of-Words
nb_tfidf = MultinomialNB()  # Naive Bayes with TF-IDF
svm_bow = SVC(class_weight=class_weight_dict)  # SVM with Bag-of-Words
svm_tfidf = SVC(class_weight=class_weight_dict)  # SVM with TF-IDF

# Train the models
nb_bow.fit(X_train_bow, y_train)  # Naive Bayes with Bag-of-Words
nb_tfidf.fit(X_train_tfidf, y_train)  # Naive Bayes with TF-IDF
svm_bow.fit(X_train_bow, y_train)  # SVM with Bag-of-Words
svm_tfidf.fit(X_train_tfidf, y_train)  # SVM with TF-IDF

# Make predictions on the test set
y_pred_nb_bow = nb_bow.predict(X_test_bow)
y_pred_nb_tfidf = nb_tfidf.predict(X_test_tfidf)
y_pred_svm_bow = svm_bow.predict(X_test_bow)
y_pred_svm_tfidf = svm_tfidf.predict(X_test_tfidf)

# Evaluate the models and print the accuracy
print("Naive Bayes (Bag-of-Words) Accuracy: ", accuracy_score(y_test, y_pred_nb_bow))
print("Naive Bayes (TF-IDF) Accuracy: ", accuracy_score(y_test, y_pred_nb_tfidf))
print("SVM (Bag-of-Words) Accuracy: ", accuracy_score(y_test, y_pred_svm_bow))
print("SVM (TF-IDF) Accuracy: ", accuracy_score(y_test, y_pred_svm_tfidf))

# Print detailed classification reports for each model
print("\nNaive Bayes (Bag-of-Words) Classification Report:\n", classification_report(y_test, y_pred_nb_bow))
print("\nNaive Bayes (TF-IDF) Classification Report:\n", classification_report(y_test, y_pred_nb_tfidf))
print("\nSVM (Bag-of-Words) Classification Report:\n", classification_report(y_test, y_pred_svm_bow))
print("\nSVM (TF-IDF) Classification Report:\n", classification_report(y_test, y_pred_svm_tfidf))
