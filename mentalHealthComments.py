#!/usr/bin/env python3
"""
NAIVE BAYES CLASSIFIER FOR REDDIT MENTAL HEALTH DATA

This script implements multiple Naive Bayes classifiers for:
1. Sentiment Analysis (positive/negative/neutral)
2. Topic Classification (anxiety, depression, therapy, etc.)
3. Support-Seeking Detection (seeking help vs providing support)
4. Crisis Detection (crisis vs normal)

Usage:
    python3 naive_bayes_classifier.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import re
import pickle


class RedditMentalHealthClassifier:
    """Naive Bayes classifier for Reddit mental health data"""

    def __init__(self, vectorizer_type='tfidf'):
        """
        Initialize classifier

        Args:
            vectorizer_type: 'tfidf' or 'count' (default: tfidf)
        """
        self.vectorizer_type = vectorizer_type

        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                min_df=2,
                max_df=0.8,
                ngram_range=(1, 2),
                stop_words='english'
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=5000,
                min_df=2,
                max_df=0.8,
                ngram_range=(1, 2),
                stop_words='english'
            )

        self.model = MultinomialNB()
        self.label_encoder = LabelEncoder()

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text) or text == '':
            return ''

        # Convert to lowercase
        text = str(text).lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def train(self, X, y):
        """
        Train the Naive Bayes classifier

        Args:
            X: List or array of text documents
            y: List or array of labels
        """
        print("\n" + "="*60)
        print("TRAINING NAIVE BAYES CLASSIFIER")
        print("="*60)

        # Preprocess text
        print("\nPreprocessing text...")
        X_clean = [self.preprocess_text(text) for text in X]

        # Encode labels
        print("Encoding labels...")
        y_encoded = self.label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        print(f"\nDataset split:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Testing: {len(X_test)} samples")

        # Vectorize
        print(f"\nVectorizing text ({self.vectorizer_type})...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        print(f"  Feature dimension: {X_train_vec.shape[1]}")

        # Train model
        print("\nTraining Naive Bayes model...")
        self.model.fit(X_train_vec, y_train)

        # Evaluate
        print("\nEvaluating model...")
        y_pred = self.model.predict(X_test_vec)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n✓ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        # Classification report
        print("\nClassification Report:")
        print("="*60)
        report = classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_,
            digits=4
        )
        print(report)

        # Confusion matrix
        print("\nConfusion Matrix:")
        print("="*60)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        # Store for later use
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred

        return accuracy

    def predict(self, texts):
        """
        Predict labels for new texts

        Args:
            texts: List of text documents

        Returns:
            List of predicted labels
        """
        # Preprocess
        texts_clean = [self.preprocess_text(text) for text in texts]

        # Vectorize
        texts_vec = self.vectorizer.transform(texts_clean)

        # Predict
        predictions = self.model.predict(texts_vec)

        # Decode labels
        predictions_decoded = self.label_encoder.inverse_transform(predictions)

        return predictions_decoded

    def predict_proba(self, texts):
        """
        Get prediction probabilities

        Args:
            texts: List of text documents

        Returns:
            DataFrame with probabilities for each class
        """
        # Preprocess
        texts_clean = [self.preprocess_text(text) for text in texts]

        # Vectorize
        texts_vec = self.vectorizer.transform(texts_clean)

        # Get probabilities
        probas = self.model.predict_proba(texts_vec)

        # Create DataFrame
        df_proba = pd.DataFrame(
            probas,
            columns=self.label_encoder.classes_
        )

        return df_proba

    def save_model(self, filename):
        """Save trained model to file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'model': self.model,
                'label_encoder': self.label_encoder
            }, f)
        print(f"\n✓ Model saved to {filename}")

    def load_model(self, filename):
        """Load trained model from file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.model = data['model']
            self.label_encoder = data['label_encoder']
        print(f"\n✓ Model loaded from {filename}")


def create_sentiment_labels(df):
    """
    Create sentiment labels based on score and keywords

    Args:
        df: DataFrame with 'content' and 'score' columns

    Returns:
        List of sentiment labels
    """
    labels = []

    for idx, row in df.iterrows():
        text = str(row.get('content', '')).lower()
        score = int(row.get('score', 0))

        # Positive keywords
        positive_words = ['thank', 'helped', 'better', 'improve', 'great', 'good',
                         'happy', 'grateful', 'appreciate', 'success', 'recover']

        # Negative keywords
        negative_words = ['depress', 'anxious', 'panic', 'suicid', 'hopeless',
                         'worse', 'afraid', 'scared', 'crisis', 'breakdown']

        # Count keywords
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)

        # Determine sentiment
        if pos_count > neg_count and score >= 1:
            labels.append('positive')
        elif neg_count > pos_count or score < 0:
            labels.append('negative')
        else:
            labels.append('neutral')

    return labels


def create_topic_labels(df):
    """
    Create topic labels based on keywords

    Args:
        df: DataFrame with 'content' column

    Returns:
        List of topic labels
    """
    labels = []

    for idx, row in df.iterrows():
        text = str(row.get('content', '')).lower()

        # Topic keywords
        if any(word in text for word in ['anxiety', 'anxious', 'panic', 'worry', 'nervous']):
            labels.append('anxiety')
        elif any(word in text for word in ['depress', 'sad', 'hopeless', 'suicide']):
            labels.append('depression')
        elif any(word in text for word in ['therapy', 'therapist', 'counseling', 'counselor', 'treatment']):
            labels.append('therapy')
        elif any(word in text for word in ['medication', 'med', 'drug', 'prescription', 'ssri']):
            labels.append('medication')
        elif any(word in text for word in ['support', 'help', 'advice', 'tips']):
            labels.append('support_seeking')
        else:
            labels.append('general')

    return labels


def create_support_labels(df):
    """
    Detect if post is seeking support vs providing support

    Args:
        df: DataFrame with 'content' column

    Returns:
        List of support labels
    """
    labels = []

    for idx, row in df.iterrows():
        text = str(row.get('content', '')).lower()

        # Seeking support indicators
        seeking = ['help', 'advice', 'what should i', 'anyone else', 'please',
                  'struggling', 'need', 'how do i', 'what can i']

        # Providing support indicators
        providing = ['try', 'suggest', 'recommend', 'helped me', 'you should',
                    'what worked', 'consider', 'maybe']

        # Count indicators
        seek_count = sum(1 for phrase in seeking if phrase in text)
        provide_count = sum(1 for phrase in providing if phrase in text)

        if seek_count > provide_count:
            labels.append('seeking_support')
        elif provide_count > 0:
            labels.append('providing_support')
        else:
            labels.append('general_discussion')

    return labels


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print(" "*15 + "NAIVE BAYES CLASSIFIER FOR MENTAL HEALTH DATA")
    print("="*80)

    # Load data
    print("\n[STEP 1] Loading dataset...")
    try:
        df = pd.read_csv('reddit_combined_expanded.csv')
        print(f"✓ Loaded {len(df)} entries")
    except FileNotFoundError:
        print("❌ Error: reddit_combined_expanded.csv not found!")
        print("\nPlease run extract_all_reddit_data.py first to create the dataset.")
        return

    # Filter only posts and comments with content
    df = df[df['content'].notna() & (df['content'] != '')]
    print(f"✓ Filtered to {len(df)} entries with content")

    if len(df) < 20:
        print("\n⚠ WARNING: Dataset too small for training!")
        print("You need at least 50-100 entries for meaningful results.")
        print("\nTo get more data:")
        print("  1. Save more Reddit HTML files")
        print("  2. Or use reddit_api_large_scraper.py to get thousands automatically")
        return

    # ========================================================================
    # TASK 1: SENTIMENT ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("TASK 1: SENTIMENT ANALYSIS (positive/negative/neutral)")
    print("="*80)

    sentiment_labels = create_sentiment_labels(df)
    df['sentiment'] = sentiment_labels

    # Check label distribution
    print("\nSentiment distribution:")
    print(df['sentiment'].value_counts())

    # Train sentiment classifier
    if df['sentiment'].nunique() >= 2:
        sentiment_clf = RedditMentalHealthClassifier(vectorizer_type='tfidf')
        sentiment_clf.train(df['content'].values, df['sentiment'].values)

        # Save model
        sentiment_clf.save_model('sentiment_model.pkl')

        # Test predictions
        print("\n" + "="*60)
        print("Example Predictions:")
        print("="*60)
        test_texts = [
            "I'm feeling much better now, therapy really helped!",
            "I'm so anxious and depressed, I don't know what to do",
            "Just wanted to share my experience with this medication"
        ]
        predictions = sentiment_clf.predict(test_texts)
        probas = sentiment_clf.predict_proba(test_texts)

        for i, text in enumerate(test_texts):
            print(f"\nText: {text}")
            print(f"Predicted: {predictions[i]}")
            print(f"Probabilities: {dict(probas.iloc[i])}")
    else:
        print("\n⚠ Not enough sentiment variety for classification")

    # ========================================================================
    # TASK 2: TOPIC CLASSIFICATION
    # ========================================================================
    print("\n" + "="*80)
    print("TASK 2: TOPIC CLASSIFICATION")
    print("="*80)

    topic_labels = create_topic_labels(df)
    df['topic'] = topic_labels

    # Check label distribution
    print("\nTopic distribution:")
    print(df['topic'].value_counts())

    # Train topic classifier
    if df['topic'].nunique() >= 2:
        topic_clf = RedditMentalHealthClassifier(vectorizer_type='tfidf')
        topic_clf.train(df['content'].values, df['topic'].values)

        # Save model
        topic_clf.save_model('topic_model.pkl')

        # Test predictions
        print("\n" + "="*60)
        print("Example Predictions:")
        print("="*60)
        test_texts = [
            "I've been having panic attacks lately and feel very anxious",
            "My therapist recommended cognitive behavioral therapy",
            "Started taking SSRIs last week, hoping they help"
        ]
        predictions = topic_clf.predict(test_texts)

        for i, text in enumerate(test_texts):
            print(f"\nText: {text}")
            print(f"Predicted Topic: {predictions[i]}")
    else:
        print("\n⚠ Not enough topic variety for classification")

    # ========================================================================
    # TASK 3: SUPPORT DETECTION
    # ========================================================================
    print("\n" + "="*80)
    print("TASK 3: SUPPORT SEEKING vs PROVIDING")
    print("="*80)

    support_labels = create_support_labels(df)
    df['support_type'] = support_labels

    # Check label distribution
    print("\nSupport type distribution:")
    print(df['support_type'].value_counts())

    # Train support classifier
    if df['support_type'].nunique() >= 2:
        support_clf = RedditMentalHealthClassifier(vectorizer_type='tfidf')
        support_clf.train(df['content'].values, df['support_type'].values)

        # Save model
        support_clf.save_model('support_model.pkl')
    else:
        print("\n⚠ Not enough support type variety for classification")

    # ========================================================================
    # SAVE LABELED DATASET
    # ========================================================================
    print("\n" + "="*80)
    print("SAVING LABELED DATASET")
    print("="*80)

    df.to_csv('reddit_data_with_labels.csv', index=False)
    print("\n✓ Saved labeled dataset to: reddit_data_with_labels.csv")
    print(f"  Columns: {', '.join(df.columns)}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\n✅ Completed Naive Bayes Classification!")
    print("\n📊 Models trained:")
    print("  1. Sentiment Analysis (positive/negative/neutral)")
    print("  2. Topic Classification (anxiety/depression/therapy/etc)")
    print("  3. Support Detection (seeking vs providing)")

    print("\n📁 Generated files:")
    print("  - sentiment_model.pkl (sentiment classifier)")
    print("  - topic_model.pkl (topic classifier)")
    print("  - support_model.pkl (support detector)")
    print("  - reddit_data_with_labels.csv (labeled dataset)")

    print("\n💡 To use the models:")
    print("  from naive_bayes_classifier import RedditMentalHealthClassifier")
    print("  clf = RedditMentalHealthClassifier()")
    print("  clf.load_model('sentiment_model.pkl')")
    print("  predictions = clf.predict(['your text here'])")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
