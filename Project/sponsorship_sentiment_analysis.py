import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries for Sentiment Analysis
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# ML Libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Text Processing
from textblob import TextBlob
from wordcloud import WordCloud

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class SponsorshipSentimentAnalyzer:
    """
    Comprehensive sentiment analysis for sports sponsorship discussions
    """
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add sports-specific stopwords but keep sentiment-relevant terms
        sports_stopwords = {
            'team', 'player', 'game', 'season', 'year', 'time', 'would', 'could', 'said'
        }
        self.stop_words.update(sports_stopwords)
        
        # Define sentiment-specific keywords for sponsorships
        self.sentiment_keywords = {
            'positive': {
                'financial': ['profitable', 'revenue', 'good deal', 'worth it', 'fair price', 'investment'],
                'aesthetic': ['looks good', 'nice design', 'clean', 'subtle', 'professional', 'sharp'],
                'brand': ['quality brand', 'respected', 'premium', 'trusted', 'established'],
                'general': ['excited', 'great', 'awesome', 'perfect', 'love', 'fantastic', 'brilliant']
            },
            'negative': {
                'financial': ['expensive', 'overpriced', 'waste', 'ripoff', 'greedy', 'money grab'],
                'aesthetic': ['ugly', 'hideous', 'cluttered', 'tacky', 'ruins', 'destroys', 'awful'],
                'brand': ['cheap', 'sketchy', 'unknown', 'questionable', 'terrible'],
                'general': ['hate', 'disgusting', 'horrible', 'worst', 'pathetic', 'ridiculous']
            }
        }
        
        self.vectorizers = {}
        self.models = {}
        self.label_encoder = LabelEncoder()
    
    def load_sponsorship_data(self, filename='distilgpt2_sponsorship_analysis.csv'):
        """
        Load the sponsorship data from previous analysis
        """
        try:
            df = pd.read_csv(filename)
            print(f"Loaded {len(df)} records from {filename}")
            
            # Show data overview
            if 'gpt2_predicted_category' in df.columns:
                print("GPT2 Predictions available:")
                print(df['gpt2_predicted_category'].value_counts())
            
            return df
        except FileNotFoundError:
            print(f"File {filename} not found. Please run the DistilGPT2 analysis first.")
            return None
    
    def preprocess_text_for_sentiment(self, text):
        """
        Preprocess text specifically for sentiment analysis
        """
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text).lower()
        
        # Remove URLs and Reddit formatting but keep emoticons and punctuation
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'/u/\w+|/r/\w+', '', text)
        
        # Keep important punctuation for sentiment (!, ?, ...)
        text = re.sub(r'[^\w\s!?.,]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_comprehensive_sentiment_scores(self, text):
        """
        Get sentiment scores using multiple methods
        """
        if pd.isna(text) or text == '':
            return {
                'vader_compound': 0, 'vader_pos': 0, 'vader_neu': 1, 'vader_neg': 0,
                'textblob_polarity': 0, 'textblob_subjectivity': 0,
                'custom_score': 0, 'final_sentiment': 'neutral'
            }
        
        processed_text = self.preprocess_text_for_sentiment(text)
        
        # VADER Sentiment (good for social media)
        vader_scores = self.sia.polarity_scores(text)
        
        # TextBlob Sentiment
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # Custom keyword-based sentiment
        custom_score = self.calculate_custom_sentiment(processed_text)
        
        # Ensemble final sentiment
        final_sentiment = self.ensemble_sentiment_classification(
            vader_scores['compound'], textblob_polarity, custom_score
        )
        
        return {
            'vader_compound': vader_scores['compound'],
            'vader_pos': vader_scores['pos'],
            'vader_neu': vader_scores['neu'],
            'vader_neg': vader_scores['neg'],
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'custom_score': custom_score,
            'final_sentiment': final_sentiment
        }
    
    def calculate_custom_sentiment(self, text):
        """
        Calculate custom sentiment score based on sponsorship-specific keywords
        """
        if not text:
            return 0
        
        positive_score = 0
        negative_score = 0
        
        # Count positive keywords
        for category, keywords in self.sentiment_keywords['positive'].items():
            for keyword in keywords:
                if keyword in text:
                    weight = 2 if category == 'general' else 1.5  # General terms weighted higher
                    positive_score += weight
        
        # Count negative keywords
        for category, keywords in self.sentiment_keywords['negative'].items():
            for keyword in keywords:
                if keyword in text:
                    weight = 2 if category == 'general' else 1.5
                    negative_score += weight
        
        # Normalize to [-1, 1] range
        total_score = positive_score - negative_score
        if total_score == 0:
            return 0
        
        # Simple normalization
        max_possible = 10  # Rough estimate of max keywords in a text
        normalized_score = max(-1, min(1, total_score / max_possible))
        
        return normalized_score
    
    def ensemble_sentiment_classification(self, vader_compound, textblob_polarity, custom_score):
        """
        Combine multiple sentiment scores for final classification
        """
        # Weighted ensemble
        ensemble_score = (
            0.4 * vader_compound +
            0.3 * textblob_polarity +
            0.3 * custom_score
        )
        
        # Classification thresholds
        if ensemble_score >= 0.1:
            return 'positive'
        elif ensemble_score <= -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_aspect_based_sentiment(self, text):
        """
        Analyze sentiment for specific aspects of sponsorship
        """
        if pd.isna(text):
            return {
                'financial_sentiment': 'neutral',
                'aesthetic_sentiment': 'neutral', 
                'brand_sentiment': 'neutral',
                'overall_sentiment': 'neutral'
            }
        
        text_lower = str(text).lower()
        
        aspects = {
            'financial': {
                'positive': self.sentiment_keywords['positive']['financial'],
                'negative': self.sentiment_keywords['negative']['financial']
            },
            'aesthetic': {
                'positive': self.sentiment_keywords['positive']['aesthetic'],
                'negative': self.sentiment_keywords['negative']['aesthetic']
            },
            'brand': {
                'positive': self.sentiment_keywords['positive']['brand'],
                'negative': self.sentiment_keywords['negative']['brand']
            }
        }
        
        # Get overall sentiment
        overall_scores = self.get_comprehensive_sentiment_scores(text)
        overall_sentiment = overall_scores['final_sentiment']
        
        # Analyze each aspect
        aspect_sentiments = {'overall_sentiment': overall_sentiment}
        
        for aspect, keywords in aspects.items():
            pos_count = sum(1 for kw in keywords['positive'] if kw in text_lower)
            neg_count = sum(1 for kw in keywords['negative'] if kw in text_lower)
            
            if pos_count > neg_count:
                aspect_sentiments[f'{aspect}_sentiment'] = 'positive'
            elif neg_count > pos_count:
                aspect_sentiments[f'{aspect}_sentiment'] = 'negative'
            else:
                # Fall back to overall sentiment if no specific aspect keywords
                aspect_sentiments[f'{aspect}_sentiment'] = overall_sentiment
        
        return aspect_sentiments
    
    def create_sentiment_features(self, df):
        """
        Create comprehensive sentiment features for the dataset
        """
        print("Creating comprehensive sentiment features...")
        
        # Basic sentiment analysis
        sentiment_data = df['text'].apply(self.get_comprehensive_sentiment_scores)
        sentiment_df = pd.json_normalize(sentiment_data)
        df = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)
        
        # Aspect-based sentiment
        aspect_data = df['text'].apply(self.analyze_aspect_based_sentiment)
        aspect_df = pd.json_normalize(aspect_data)
        df = pd.concat([df.reset_index(drop=True), aspect_df], axis=1)
        
        # Additional sentiment features
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        df['exclamation_count'] = df['text'].str.count('!')
        df['question_count'] = df['text'].str.count('\?')
        df['caps_ratio'] = df['text'].str.count(r'[A-Z]') / df['text'].str.len().replace(0, 1)
        
        # Sentiment intensity features
        df['sentiment_intensity'] = abs(df['vader_compound'])
        df['is_subjective'] = df['textblob_subjectivity'] > 0.5
        df['is_strong_sentiment'] = df['sentiment_intensity'] > 0.6
        
        print(f"Sentiment features created for {len(df)} records")
        return df
    
    def train_sentiment_classifiers(self, df):
        """
        Train machine learning models for sentiment classification
        """
        print("Training sentiment classification models...")
        
        # Prepare text data
        df['processed_text'] = df['text'].apply(self.preprocess_text_for_sentiment)
        
        # Filter out empty texts
        df_clean = df[df['processed_text'] != ''].copy()
        
        if len(df_clean) < 10:
            print("Not enough data for training ML models.")
            return {}
        
        # Prepare features and labels
        X_text = df_clean['processed_text']
        y = df_clean['final_sentiment']
        
        # Check label distribution
        label_counts = y.value_counts()
        print(f"Label distribution: {label_counts.to_dict()}")
        
        # Only proceed if we have multiple classes with sufficient examples
        if len(label_counts) < 2 or label_counts.min() < 2:
            print("Insufficient label diversity for ML training.")
            return {}
        
        # Create feature vectors
        # TF-IDF Features
        tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        X_tfidf = tfidf_vectorizer.fit_transform(X_text)
        
        # Count Features  
        count_vectorizer = CountVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        X_count = count_vectorizer.fit_transform(X_text)
        
        # Store vectorizers
        self.vectorizers['tfidf'] = tfidf_vectorizer
        self.vectorizers['count'] = count_vectorizer
        
        # Split data
        try:
            X_tfidf_train, X_tfidf_test, y_train, y_test = train_test_split(
                X_tfidf, y, test_size=0.2, random_state=42, stratify=y
            )
            X_count_train, X_count_test, _, _ = train_test_split(
                X_count, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            # Fall back to simple split if stratification fails
            X_tfidf_train, X_tfidf_test, y_train, y_test = train_test_split(
                X_tfidf, y, test_size=0.2, random_state=42
            )
            X_count_train, X_count_test, _, _ = train_test_split(
                X_count, y, test_size=0.2, random_state=42
            )
        
        # Train models
        models = {
            'logistic_tfidf': LogisticRegression(random_state=42, max_iter=1000),
            'logistic_count': LogisticRegression(random_state=42, max_iter=1000),
            'rf_tfidf': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm_tfidf': SVC(kernel='linear', random_state=42)
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            
            try:
                if 'tfidf' in model_name:
                    model.fit(X_tfidf_train, y_train)
                    y_pred = model.predict(X_tfidf_test)
                else:
                    model.fit(X_count_train, y_train)
                    y_pred = model.predict(X_count_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                
                results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'true_labels': y_test
                }
                
                print(f"{model_name} accuracy: {accuracy:.3f}")
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
        
        self.models = results
        return results
    
    def generate_sentiment_insights_report(self, df):
        """
        Generate comprehensive sentiment analysis report
        """
        print("\n" + "="*70)
        print("SPORTS SPONSORSHIP SENTIMENT ANALYSIS REPORT")
        print("="*70)
        
        # Overall sentiment distribution
        print("\n1. OVERALL SENTIMENT DISTRIBUTION")
        sentiment_dist = df['final_sentiment'].value_counts(normalize=True) * 100
        for sentiment, pct in sentiment_dist.items():
            print(f"   {sentiment.title()}: {pct:.1f}%")
        
        # Sentiment by sponsorship type (if available)
        if 'gpt2_predicted_category' in df.columns:
            print("\n2. SENTIMENT BY SPONSORSHIP TYPE")
            sponsorship_df = df[df['gpt2_predicted_category'] != 'not_sponsorship']
            
            if len(sponsorship_df) > 0:
                type_sentiment = sponsorship_df.groupby(['gpt2_predicted_category', 'final_sentiment']).size().unstack(fill_value=0)
                type_sentiment_pct = type_sentiment.div(type_sentiment.sum(axis=1), axis=0) * 100
                print(type_sentiment_pct.round(1))
        
        # Aspect-based sentiment analysis
        print("\n3. ASPECT-BASED SENTIMENT ANALYSIS")
        aspects = ['financial_sentiment', 'aesthetic_sentiment', 'brand_sentiment']
        
        for aspect in aspects:
            if aspect in df.columns:
                aspect_name = aspect.replace('_sentiment', '').title()
                aspect_dist = df[aspect].value_counts(normalize=True) * 100
                print(f"\n   {aspect_name} Sentiment:")
                for sentiment, pct in aspect_dist.items():
                    print(f"     {sentiment.title()}: {pct:.1f}%")
        
        # Sentiment intensity analysis
        print("\n4. SENTIMENT INTENSITY ANALYSIS")
        print(f"   Average sentiment intensity: {df['sentiment_intensity'].mean():.3f}")
        print(f"   Posts with strong sentiment (>0.6): {(df['is_strong_sentiment']).sum()} ({(df['is_strong_sentiment']).mean()*100:.1f}%)")
        print(f"   Subjective posts: {(df['is_subjective']).sum()} ({(df['is_subjective']).mean()*100:.1f}%)")
        
        # Sentiment by subreddit (if available)
        if 'subreddit' in df.columns:
            print("\n5. SENTIMENT BY SUBREDDIT")
            subreddit_sentiment = df.groupby(['subreddit', 'final_sentiment']).size().unstack(fill_value=0)
            if not subreddit_sentiment.empty:
                subreddit_sentiment_pct = subreddit_sentiment.div(subreddit_sentiment.sum(axis=1), axis=0) * 100
                print(subreddit_sentiment_pct.round(1))
        
        # Most positive and negative examples
        print("\n6. SAMPLE SENTIMENT EXAMPLES")
        
        positive_samples = df[df['final_sentiment'] == 'positive'].nlargest(3, 'vader_compound')
        if len(positive_samples) > 0:
            print("\n   Most Positive Comments:")
            for i, (_, row) in enumerate(positive_samples.iterrows(), 1):
                print(f"   {i}. Score: {row['vader_compound']:.3f}")
                print(f"      Text: {row['text'][:100]}...")
        
        negative_samples = df[df['final_sentiment'] == 'negative'].nsmallest(3, 'vader_compound')
        if len(negative_samples) > 0:
            print("\n   Most Negative Comments:")
            for i, (_, row) in enumerate(negative_samples.iterrows(), 1):
                print(f"   {i}. Score: {row['vader_compound']:.3f}")
                print(f"      Text: {row['text'][:100]}...")
        
        # Key insights and recommendations
        print("\n7. KEY INSIGHTS & STRATEGIC RECOMMENDATIONS")
        
        positive_pct = (df['final_sentiment'] == 'positive').mean() * 100
        negative_pct = (df['final_sentiment'] == 'negative').mean() * 100
        avg_sentiment = df['vader_compound'].mean()
        
        print(f"   • Overall sentiment score: {avg_sentiment:.3f}")
        print(f"   • {positive_pct:.1f}% positive, {negative_pct:.1f}% negative sentiment")
        
        if positive_pct > 50:
            print("   ✓ Majority positive sentiment - favorable environment for sponsorships")
        elif negative_pct > 40:
            print("   ⚠ High negative sentiment - address fan concerns")
        
        # Aspect-specific insights
        if 'financial_sentiment' in df.columns:
            financial_negative = (df['financial_sentiment'] == 'negative').mean() * 100
            if financial_negative > 30:
                print("   ⚠ Financial concerns - emphasize value and fan benefits")
        
        if 'aesthetic_sentiment' in df.columns:
            aesthetic_negative = (df['aesthetic_sentiment'] == 'negative').mean() * 100
            if aesthetic_negative > 30:
                print("   ⚠ Design concerns - focus on subtle, integrated sponsorship placement")
    
    def create_sentiment_visualizations(self, df):
        """
        Create visualizations for sentiment analysis
        """
        print("Creating sentiment visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Sports Sponsorship Sentiment Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Overall sentiment distribution
        sentiment_counts = df['final_sentiment'].value_counts()
        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Overall Sentiment Distribution')
        
        # 2. Sentiment scores distribution
        axes[0, 1].hist(df['vader_compound'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('VADER Sentiment Score Distribution')
        axes[0, 1].set_xlabel('Sentiment Score (-1 to 1)')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Sentiment by subreddit (if available)
        if 'subreddit' in df.columns and df['subreddit'].nunique() > 1:
            subreddit_sentiment = df.groupby(['subreddit', 'final_sentiment']).size().unstack(fill_value=0)
            subreddit_sentiment.plot(kind='bar', ax=axes[0, 2], stacked=True)
            axes[0, 2].set_title('Sentiment by Subreddit')
            axes[0, 2].tick_params(axis='x', rotation=45)
        else:
            axes[0, 2].text(0.5, 0.5, 'Insufficient\nSubreddit Data', ha='center', va='center', fontsize=12)
            axes[0, 2].set_title('Sentiment by Subreddit')
        
        # 4. Aspect-based sentiment comparison
        aspects = ['financial_sentiment', 'aesthetic_sentiment', 'brand_sentiment']
        aspect_data = []
        
        for aspect in aspects:
            if aspect in df.columns:
                aspect_name = aspect.replace('_sentiment', '').title()
                positive_pct = (df[aspect] == 'positive').mean() * 100
                aspect_data.append((aspect_name, positive_pct))
        
        if aspect_data:
            aspect_names, positive_pcts = zip(*aspect_data)
            axes[1, 0].bar(aspect_names, positive_pcts, color=['green', 'blue', 'orange'])
            axes[1, 0].set_title('Positive Sentiment by Aspect')
            axes[1, 0].set_ylabel('Positive Sentiment %')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Sentiment intensity vs subjectivity
        if 'textblob_subjectivity' in df.columns:
            scatter = axes[1, 1].scatter(df['textblob_subjectivity'], df['sentiment_intensity'], 
                                       c=df['vader_compound'], cmap='RdYlGn', alpha=0.6)
            axes[1, 1].set_xlabel('Subjectivity')
            axes[1, 1].set_ylabel('Sentiment Intensity')
            axes[1, 1].set_title('Sentiment Intensity vs Subjectivity')
            plt.colorbar(scatter, ax=axes[1, 1], label='VADER Score')
        
        # 6. Word cloud of positive and negative sentiments
        try:
            positive_text = ' '.join(df[df['final_sentiment'] == 'positive']['text'].astype(str))
            negative_text = ' '.join(df[df['final_sentiment'] == 'negative']['text'].astype(str))
            
            if positive_text and negative_text:
                # Create a simple text summary instead of wordcloud if wordcloud fails
                axes[1, 2].text(0.5, 0.7, f"Positive Texts: {len(df[df['final_sentiment'] == 'positive'])}", 
                               ha='center', va='center', fontsize=12, color='green')
                axes[1, 2].text(0.5, 0.5, f"Negative Texts: {len(df[df['final_sentiment'] == 'negative'])}", 
                               ha='center', va='center', fontsize=12, color='red')
                axes[1, 2].text(0.5, 0.3, f"Neutral Texts: {len(df[df['final_sentiment'] == 'neutral'])}", 
                               ha='center', va='center', fontsize=12, color='gray')
            axes[1, 2].set_title('Sentiment Summary')
            axes[1, 2].axis('off')
        except:
            axes[1, 2].text(0.5, 0.5, 'Visualization\nNot Available', ha='center', va='center')
            axes[1, 2].set_title('Word Cloud')
        
        plt.tight_layout()
        plt.savefig('sponsorship_sentiment_analysis.png', dpi=300, bbox_inches='tight')
        print("Visualizations saved as 'sponsorship_sentiment_analysis.png'")
        plt.show()
    
    def save_sentiment_results(self, df, filename_prefix='sponsorship_sentiment_analysis'):
        """
        Save comprehensive sentiment analysis results
        """
        # Save detailed dataset
        df.to_csv(f'{filename_prefix}_detailed.csv', index=False)
        
        # Create summary statistics
        summary_stats = {
            'analysis_date': datetime.now().isoformat(),
            'total_posts_analyzed': len(df),
            'sentiment_distribution': df['final_sentiment'].value_counts().to_dict(),
            'average_sentiment_scores': {
                'vader_compound': df['vader_compound'].mean(),
                'textblob_polarity': df['textblob_polarity'].mean(),
                'custom_score': df['custom_score'].mean()
            },
            'sentiment_intensity': {
                'average': df['sentiment_intensity'].mean(),
                'strong_sentiment_posts': (df['is_strong_sentiment']).sum(),
                'subjective_posts': (df['is_subjective']).sum()
            },
            'aspect_sentiment': {}
        }
        
        # Add aspect sentiment if available
        aspects = ['financial_sentiment', 'aesthetic_sentiment', 'brand_sentiment']
        for aspect in aspects:
            if aspect in df.columns:
                summary_stats['aspect_sentiment'][aspect] = df[aspect].value_counts().to_dict()
        
        # Save summary
        import json
        with open(f'{filename_prefix}_summary.json', 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        print(f"Results saved to {filename_prefix}_detailed.csv and {filename_prefix}_summary.json")

def run_sponsorship_sentiment_analysis(data_file='distilgpt2_sponsorship_analysis.csv'):
    """
    Run comprehensive sentiment analysis on sponsorship data
    """
    # Initialize analyzer
    analyzer = SponsorshipSentimentAnalyzer()
    
    # Load data
    df = analyzer.load_sponsorship_data(data_file)
    if df is None:
        return None
    
    # Create sentiment features
    df_with_sentiment = analyzer.create_sentiment_features(df)
    
    # Train ML models (if enough data)
    model_results = analyzer.train_sentiment_classifiers(df_with_sentiment)
    
    # Generate insights report
    analyzer.generate_sentiment_insights_report(df_with_sentiment)
    
    # Create visualizations
    analyzer.create_sentiment_visualizations(df_with_sentiment)
    
    # Save results
    analyzer.save_sentiment_results(df_with_sentiment)
    
    return df_with_sentiment, analyzer, model_results

if __name__ == "__main__":
    # Run the comprehensive sentiment analysis
    results, analyzer, models = run_sponsorship_sentiment_analysis()
    
    print("\n🎯 Sentiment Analysis Complete!")
    print("📊 Check the generated files:")
    print("   - sponsorship_sentiment_analysis_detailed.csv")
    print("   - sponsorship_sentiment_analysis_summary.json") 
    print("   - sponsorship_sentiment_analysis.png")