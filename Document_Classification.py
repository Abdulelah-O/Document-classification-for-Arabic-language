import os
import nltk
import numpy as np
from nltk.corpus import stopwords
from camel_tools.utils.dediac import dediac_ar
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.normalize import normalize_unicode
from camel_tools.utils.normalize import normalize_alef_maksura_ar, normalize_alef_ar, normalize_teh_marbuta_ar
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse import hstack
from sklearn.feature_selection import SelectKBest, chi2
import unicodedata

# Load Arabic stopwords from nltk
arabic_stopwords = set(stopwords.words('arabic'))

# Define a function to clean, normalize and remove stop words from Arabic text
def clean_and_normalize(text):
    tokens = simple_word_tokenize(text)
    text = ' '.join(tokens)
    
    # Normalize Arabic letters using camel-tools normalization functions 
    text = normalize_alef_ar(text)  # Normalize Alef (أ, إ, آ to ا)
    text = normalize_alef_maksura_ar(text)  # Normalize Alef Maksura (ى to ي)
    text = normalize_teh_marbuta_ar(text)  # Normalize Teh Marbuta (ة to ه)
    text = dediac_ar(text)  # Remove diacritics
    text = normalize_unicode(text)  # Normalize Unicode
    text = unicodedata.normalize('NFKD', text)  # Additional normalization

    # Remove punctuation manually
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])

    # Tokenize the text after cleaning
    tokens = simple_word_tokenize(text)

    # Remove stop words from the tokens
    tokens = [word for word in tokens if word not in arabic_stopwords]

    # Rejoin the tokens back into a cleaned text string
    return ' '.join(tokens)

# Define a function to extract two hand-crafted features from the text
def extract_handcrafted_features(text):
    # Text length (number of characters)
    text_length = len(text)

    # Number of words in the text
    word_count = len(text.split())

    # Hand-crafted features combined in a list (only two features)
    features = [
        text_length,
        word_count
    ]

    return np.array(features)

# Define the path to your text files directory
input_directory = "C:/Users/abode/NLP/NLP"  # Change to your directory path

# Lists to store data for features and labels
text_data = []
labels = []
handcrafted_features_list = []

# Define the class labels
class_labels = ['sports', 'religion', 'economy']

# Loop through the text files in the directory and process each file
for filename in os.listdir(input_directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_directory, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text_content = file.read().strip()

            cleaned_text = clean_and_normalize(text_content)

            if 'sports' in filename.lower():
                label = 'sports'
            elif 'religion' in filename.lower():
                label = 'religion'
            elif 'economy' in filename.lower():
                label = 'economy'
            else:
                print(f"Skipping file with unknown label: {filename}")
                continue

            if cleaned_text.strip() == '':
                print(f"Skipping empty file: {filename}")
                continue

            text_data.append(cleaned_text)
            labels.append(label)

            # Extract hand-crafted features for each text
            handcrafted_features = extract_handcrafted_features(cleaned_text)
            handcrafted_features_list.append(handcrafted_features)

# Convert the list of handcrafted features to a NumPy array
handcrafted_features_array = np.array(handcrafted_features_list)

# Use TfidfVectorizer to convert text data into numerical features
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(text_data)

# Combine TF-IDF features and hand-crafted features into a single feature matrix
X_combined = hstack([X_tfidf, handcrafted_features_array])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, labels, test_size=0.2, random_state=42)

# Apply SelectKBest to select the top 20 features
selector = SelectKBest(chi2, k=20)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Create classifiers
svm_classifier = svm.LinearSVC()
nb_classifier = MultinomialNB()
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Train the classifiers
svm_classifier.fit(X_train_selected, y_train)
nb_classifier.fit(X_train_selected, y_train)
knn_classifier.fit(X_train_selected, y_train)

# Make predictions on the test set
y_pred_svm = svm_classifier.predict(X_test_selected)
y_pred_nb = nb_classifier.predict(X_test_selected)
y_pred_knn = knn_classifier.predict(X_test_selected)

# Evaluate the model's performance
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print(f"SVM Model Accuracy: {accuracy_svm:.4f}")
print(f"Naive Bayes Model Accuracy: {accuracy_nb:.4f}")
print(f"KNN Model Accuracy: {accuracy_knn:.4f}")

# Display classification reports
print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm))
print("Naive Bayes Classification Report:\n", classification_report(y_test, y_pred_nb))
print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn))

# Function to test a new paragraph of text
def test_new_paragraph(paragraph):
    cleaned_paragraph = clean_and_normalize(paragraph)
    handcrafted_features = extract_handcrafted_features(cleaned_paragraph)
    
    # Convert the cleaned paragraph to TF-IDF format using the same vectorizer
    paragraph_tfidf = tfidf_vectorizer.transform([cleaned_paragraph])
    
    # Combine TF-IDF features and hand-crafted features
    paragraph_combined = hstack([paragraph_tfidf, handcrafted_features.reshape(1, -1)])
    
    # Select top 20 features from the paragraph's feature set
    paragraph_combined_selected = selector.transform(paragraph_combined)
    
    # Predict the label using the trained models
    prediction_svm = svm_classifier.predict(paragraph_combined_selected)
    prediction_nb = nb_classifier.predict(paragraph_combined_selected)
    prediction_knn = knn_classifier.predict(paragraph_combined_selected)
    
    print(f"SVM Prediction: {prediction_svm[0]}")
    print(f"Naive Bayes Prediction: {prediction_nb[0]}")
    print(f"KNN Prediction: {prediction_knn[0]}")

# Test with Three external examples
test_paragraph_1 = """
يشهد الاقتصاد العالمي تغيرات كبيرة نتيجة التحولات التكنولوجية المتسارعة والتوترات التجارية بين الدول الكبرى. أدت هذه العوامل إلى تذبذب في الأسواق المالية.
"""

test_paragraph_2 = """
تشهد الرياضة العالمية اهتمامًا متزايدًا بفضل الإنجازات الرياضية الكبرى التي تحققها الفرق والرياضيون في مختلف الألعاب الرياضية.
"""
test_paragraph_3 =  """
    تلعب الأديان دورًا رئيسيًا في تشكيل الثقافة والمجتمع. يشهد العالم تنوعًا دينيًا واسعًا، حيث يعتنق الناس معتقدات مختلفة.
    """
# Test paragraphs on all classifiers
print("\nTest Paragraph 1 (Economy):")
test_new_paragraph(test_paragraph_1)

print("\nTest Paragraph 2 (Sports):")
test_new_paragraph(test_paragraph_2)

print("\nTest Paragraph 3 (Religion):")
test_new_paragraph(test_paragraph_3)