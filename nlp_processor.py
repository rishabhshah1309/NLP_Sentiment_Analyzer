"""
Fixed NLP Processor for Sports Sponsorship Sentiment Analysis
ISBA 2411 - Rishabh Shah / hermantacobear

Fixes the "empty vocabulary" error with better text preprocessing
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple
import logging

# NLP libraries
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedSponsorshipNLPProcessor:
    """Fixed NLP processor that handles text preprocessing better"""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize NLP tools
        self.sia = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Minimal stop words - keep sponsorship-relevant terms
        basic_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'through',
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
            'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'shall'
        }
        self.stop_words = basic_stopwords
        
        # Keep these important words that are often in stopword lists
        self.keep_words = {
            'sponsor', 'sponsorship', 'partnership', 'deal', 'agreement', 'contract',
            'nike', 'adidas', 'emirates', 'coca', 'cola', 'pepsi', 'jersey', 'kit',
            'stadium', 'arena', 'naming', 'rights', 'official', 'partner', 'brand',
            'team', 'player', 'club', 'football', 'soccer', 'basketball', 'baseball',
            'good', 'bad', 'great', 'terrible', 'love', 'hate', 'like', 'dislike',
            'positive', 'negative', 'amazing', 'awful', 'excellent', 'horrible'
        }
        
        # Custom sentiment keywords
        self.positive_keywords = [
            'excited', 'great', 'awesome', 'love', 'perfect', 'amazing', 'excellent',
            'quality', 'professional', 'clean', 'good deal', 'fair price', 'smart',
            'brilliant', 'fantastic', 'outstanding', 'superb', 'wonderful'
        ]
        
        self.negative_keywords = [
            'hate', 'disgusting', 'terrible', 'awful', 'ugly', 'expensive', 'horrible',
            'sellout', 'commercial', 'ruins', 'greedy', 'waste', 'worst', 'pathetic',
            'ridiculous', 'stupid', 'disappointing', 'trash', 'garbage'
        ]
        
        # Sponsorship classification keywords
        self.sponsorship_types = {
            'jersey_partnership': ['jersey', 'shirt', 'kit', 'uniform', 'front of shirt', 'sleeve'],
            'naming_rights': ['stadium', 'arena', 'naming rights', 'venue', 'ballpark', 'field'],
            'brand_partnership': ['official partner', 'global partner', 'strategic partner', 'exclusive'],
            'title_sponsorship': ['title sponsor', 'presenting sponsor', 'league sponsor', 'tournament']
        }
        
        self.vectorizers = {}
        self.models = {}
        self.label_encoder = LabelEncoder()
    
    def preprocess_text_conservative(self, text):
        """Conservative text preprocessing that preserves more content"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string and basic cleaning
        text = str(text).lower()
        
        # Remove URLs but keep everything else
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'/u/\w+|/r/\w+', '', text)  # Reddit specific
        
        # Keep alphanumeric, spaces, and basic punctuation
        text = re.sub(r'[^\w\s!?.,\'-]', ' ', text)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Only do minimal tokenization and filtering
        try:
            tokens = word_tokenize(text)
            
            # Only remove very basic stop words, keep content words
            filtered_tokens = []
            for token in tokens:
                if (len(token) > 1 and  # Keep words longer than 1 character
                    token.isalpha() and  # Only alphabetic tokens
                    (token not in self.stop_words or token in self.keep_words)):  # Remove basic stopwords but keep important ones
                    filtered_tokens.append(token)
            
            result = ' '.join(filtered_tokens)
            
            # Ensure we return something meaningful
            if len(result.strip()) < 3:
                # If over-filtered, return original cleaned text
                return re.sub(r'[^\w\s]', ' ', str(text).lower()).strip()
            
            return result
            
        except Exception as e:
            # If tokenization fails, return basic cleaned text
            return re.sub(r'[^\w\s]', ' ', str(text).lower()).strip()
    
    def get_vader_sentiment(self, text):
        """Get VADER sentiment analysis"""
        if pd.isna(text) or text == '':
            return {'compound': 0, 'pos': 0, 'neu': 1, 'neg': 0}
        
        return self.sia.polarity_scores(str(text))
    
    def get_textblob_sentiment(self, text):
        """Get TextBlob sentiment analysis"""
        if pd.isna(text) or text == '':
            return {'polarity': 0, 'subjectivity': 0}
        
        try:
            blob = TextBlob(str(text))
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except:
            return {'polarity': 0, 'subjectivity': 0}
    
    def get_custom_sentiment(self, text):
        """Calculate custom sentiment based on keyword matching"""
        if pd.isna(text) or text == '':
            return 0
        
        text_lower = str(text).lower()
        
        positive_score = sum(1 for kw in self.positive_keywords if kw in text_lower)
        negative_score = sum(1 for kw in self.negative_keywords if kw in text_lower)
        
        if positive_score == 0 and negative_score == 0:
            return 0
        
        # Normalize to [-1, 1] range
        total_score = positive_score - negative_score
        max_possible = max(len(self.positive_keywords), len(self.negative_keywords))
        
        return max(-1, min(1, total_score / max_possible))
    
    def classify_sponsorship_type(self, text):
        """Classify the type of sponsorship"""
        if pd.isna(text):
            return 'unknown'
        
        text_lower = str(text).lower()
        
        # Check if it's sponsorship-related
        sponsorship_indicators = ['sponsor', 'sponsorship', 'deal', 'partnership', 'agreement']
        if not any(indicator in text_lower for indicator in sponsorship_indicators):
            return 'not_sponsorship'
        
        # Classify specific type
        for category, keywords in self.sponsorship_types.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return 'general_sponsorship'
    
    def ensemble_sentiment_classification(self, vader_score, textblob_score, custom_score):
        """Combine multiple sentiment scores"""
        # Weighted ensemble
        ensemble_score = (0.4 * vader_score + 0.3 * textblob_score + 0.3 * custom_score)
        
        if ensemble_score >= 0.1:
            return 'positive'
        elif ensemble_score <= -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def create_features(self, df):
        """Create comprehensive feature set with better error handling"""
        logger.info("Creating NLP features...")
        
        # Preprocess text conservatively
        logger.info("Preprocessing text...")
        df['processed_text'] = df['text'].apply(self.preprocess_text_conservative)
        
        # Check if we have meaningful text after preprocessing
        meaningful_text = df[df['processed_text'].str.len() > 5]
        logger.info(f"Meaningful text after preprocessing: {len(meaningful_text)}/{len(df)}")
        
        if len(meaningful_text) == 0:
            logger.warning("No meaningful text after preprocessing, using original text")
            df['processed_text'] = df['text'].str.lower().str.replace(r'[^\w\s]', ' ', regex=True)
        
        # Sentiment analysis on original text (not processed)
        logger.info("Analyzing sentiment...")
        vader_results = df['text'].apply(self.get_vader_sentiment)
        textblob_results = df['text'].apply(self.get_textblob_sentiment)
        
        # Extract sentiment scores
        df['vader_compound'] = [r['compound'] for r in vader_results]
        df['vader_pos'] = [r['pos'] for r in vader_results]
        df['vader_neu'] = [r['neu'] for r in vader_results]
        df['vader_neg'] = [r['neg'] for r in vader_results]
        
        df['textblob_polarity'] = [r['polarity'] for r in textblob_results]
        df['textblob_subjectivity'] = [r['subjectivity'] for r in textblob_results]
        
        df['custom_sentiment'] = df['text'].apply(self.get_custom_sentiment)
        
        # Ensemble sentiment
        df['ensemble_sentiment'] = df.apply(
            lambda row: self.ensemble_sentiment_classification(
                row['vader_compound'], row['textblob_polarity'], row['custom_sentiment']
            ), axis=1
        )
        
        # Sponsorship classification
        df['sponsorship_type'] = df['text'].apply(self.classify_sponsorship_type)
        
        # Additional features
        df['word_count'] = df['processed_text'].str.split().str.len()
        df['exclamation_count'] = df['text'].str.count('!')
        df['question_count'] = df['text'].str.count('\\?')
        df['caps_ratio'] = df['text'].str.count(r'[A-Z]') / df['text'].str.len().replace(0, 1)
        
        # Sentiment intensity
        df['sentiment_intensity'] = abs(df['vader_compound'])
        df['is_subjective'] = df['textblob_subjectivity'] > 0.5
        
        logger.info(f"Features created for {len(df)} posts")
        return df
    
    def train_ml_models_safe(self, df):
        """Train ML models with better error handling"""
        logger.info("Training ML models...")
        
        # Filter out non-sponsorship content for training
        sponsorship_df = df[df['sponsorship_type'] != 'not_sponsorship'].copy()
        
        if len(sponsorship_df) < 10:
            logger.warning("Not enough sponsorship data for ML training")
            return {}
        
        # Prepare features
        X = sponsorship_df['processed_text']
        y = sponsorship_df['ensemble_sentiment']
        
        # Check if we have text and multiple classes
        valid_texts = X[X.str.len() > 0]
        if len(valid_texts) < 5:
            logger.warning("Not enough valid text for ML training")
            return {}
        
        if y.nunique() < 2:
            logger.warning("Need multiple sentiment classes for training")
            return {}
        
        # Filter to only valid texts
        valid_mask = X.str.len() > 0
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        try:
            # Create TF-IDF features with more lenient settings
            tfidf_vectorizer = TfidfVectorizer(
                max_features=500,  # Reduced from 1000
                ngram_range=(1, 1),  # Only unigrams to start
                min_df=1,  # More lenient - include words that appear at least once
                max_df=0.95,
                stop_words=None,  # Don't use sklearn's stop words
                lowercase=True,
                token_pattern=r'\\b\\w{2,}\\b'  # Words with at least 2 characters
            )
            
            logger.info(f"Training on {len(X_valid)} texts...")
            X_tfidf = tfidf_vectorizer.fit_transform(X_valid)
            
            logger.info(f"TF-IDF vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
            
            if len(tfidf_vectorizer.vocabulary_) == 0:
                logger.error("Empty vocabulary after TF-IDF - using simpler approach")
                return {}
            
            self.vectorizers['tfidf'] = tfidf_vectorizer
            
            # Split data
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_tfidf, y_valid, test_size=0.2, random_state=42, stratify=y_valid
                )
            except ValueError:
                # If stratification fails, use regular split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_tfidf, y_valid, test_size=0.2, random_state=42
                )
            
            # Train models
            models = {
                'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
                'random_forest': RandomForestClassifier(n_estimators=50, random_state=42)  # Reduced trees
            }
            
            results = {}
            for name, model in models.items():
                try:
                    logger.info(f"Training {name}...")
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    results[name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'vectorizer': tfidf_vectorizer
                    }
                    
                    logger.info(f"{name} accuracy: {accuracy:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error training {name}: {e}")
            
            self.models = results
            return results
            
        except ValueError as e:
            if "empty vocabulary" in str(e):
                logger.error("Empty vocabulary error - text preprocessing removed too much content")
                logger.info("Suggestion: Check your text data and preprocessing")
                return {}
            else:
                raise e
        
        except Exception as e:
            logger.error(f"ML training error: {e}")
            return {}
    
    def simple_gpt2_analysis(self, df):
        """Simple rule-based analysis"""
        logger.info("Running simplified sponsorship classification...")
        
        def classify_sentiment(text):
            if pd.isna(text):
                return {'sentiment': 'neutral', 'confidence': 0.0}
            
            text_lower = str(text).lower()
            
            # Simple rule-based classification
            positive_words = ['great', 'awesome', 'love', 'excellent', 'amazing', 'good', 'smart', 'brilliant']
            negative_words = ['hate', 'terrible', 'awful', 'bad', 'worst', 'horrible', 'disgusting', 'stupid']
            
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                return {'sentiment': 'positive', 'confidence': 0.7}
            elif neg_count > pos_count:
                return {'sentiment': 'negative', 'confidence': 0.7}
            else:
                return {'sentiment': 'neutral', 'confidence': 0.5}
        
        gpt_results = df['text'].apply(classify_sentiment)
        df['gpt2_sentiment'] = [r['sentiment'] for r in gpt_results]
        df['gpt2_confidence'] = [r['confidence'] for r in gpt_results]
        
        return df
    
    def process_data(self, df):
        """Main processing function with better error handling"""
        logger.info("Starting NLP processing...")
        
        if len(df) == 0:
            logger.warning("No data to process")
            return df
        
        try:
            # Create features
            df_processed = self.create_features(df)
            
            # Train ML models with error handling
            try:
                model_results = self.train_ml_models_safe(df_processed)
                if model_results:
                    logger.info(f"Successfully trained {len(model_results)} ML models")
                else:
                    logger.info("ML training skipped - using rule-based analysis only")
            except Exception as e:
                logger.warning(f"ML training failed: {e}")
                logger.info("Continuing with rule-based analysis...")
            
            # Add simple GPT-style analysis
            df_processed = self.simple_gpt2_analysis(df_processed)
            
            logger.info("NLP processing completed successfully")
            return df_processed
            
        except Exception as e:
            logger.error(f"NLP processing error: {e}")
            logger.info("Returning basic sentiment analysis...")
            
            # Fallback: just do basic sentiment analysis
            df['vader_scores'] = df['text'].apply(self.get_vader_sentiment)
            df['vader_compound'] = [s['compound'] for s in df['vader_scores']]
            df['ensemble_sentiment'] = df['vader_compound'].apply(
                lambda x: 'positive' if x > 0.1 else 'negative' if x < -0.1 else 'neutral'
            )
            df['sponsorship_type'] = df['text'].apply(self.classify_sponsorship_type)
            
            return df

def process_sponsorship_data(df, config):
    """Main function for NLP processing"""
    processor = FixedSponsorshipNLPProcessor(config)
    return processor.process_data(df)

print("🔧 FIXED NLP PROCESSOR READY!")
print("="*50)
print("✅ Fixes:")
print("  • Conservative text preprocessing")
print("  • Better stop word handling")
print("  • Robust error handling")
print("  • Fallback methods")
print("  • Empty vocabulary protection")
print("\\n💡 This should resolve the 'empty vocabulary' error!")