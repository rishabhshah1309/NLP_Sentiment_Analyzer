"""
Advanced Keyword Sentiment Scorer and Visualizer
ISBA 2411 - Rishabh Shah

Creates detailed sentiment scores for individual words and phrases with visualization
Shows how much each keyword contributes to positive/negative sentiment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import re
from typing import Dict, List, Tuple, Optional
import logging
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# NLP libraries
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from textblob import TextBlob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KeywordSentimentScorer:
    """Advanced keyword sentiment analysis with detailed scoring and visualization"""
    
    def __init__(self, df, config=None):
        self.df = df
        self.config = config
        
        # Initialize sentiment analyzer
        self.sia = SentimentIntensityAnalyzer()
        
        # Stop words
        self.stop_words = set(stopwords.words('english'))
        self.sports_stopwords = {'team', 'teams', 'game', 'games', 'player', 'players'}
        
        # Keep important sentiment and domain words
        self.keep_words = {
            'sponsor', 'sponsorship', 'partnership', 'deal', 'nike', 'adidas', 'emirates',
            'good', 'bad', 'great', 'terrible', 'love', 'hate', 'amazing', 'awful',
            'excited', 'disappointed', 'brilliant', 'disgusting', 'perfect', 'horrible'
        }
        
        # Enhanced sports sponsorship lexicon with detailed scoring
        self.detailed_lexicon = self._create_detailed_lexicon()
        
        # Results storage
        self.keyword_scores = {}
        self.phrase_scores = {}
        self.context_scores = {}
        
    def _create_detailed_lexicon(self):
        """Create detailed sentiment lexicon with explanations"""
        return {
            # Very Positive (+2.0 to +1.5)
            'amazing': {'score': 2.0, 'category': 'very_positive', 'context': 'enthusiasm'},
            'incredible': {'score': 2.0, 'category': 'very_positive', 'context': 'amazement'},
            'fantastic': {'score': 1.8, 'category': 'very_positive', 'context': 'enthusiasm'},
            'brilliant': {'score': 1.8, 'category': 'very_positive', 'context': 'admiration'},
            'perfect': {'score': 1.7, 'category': 'very_positive', 'context': 'satisfaction'},
            'outstanding': {'score': 1.7, 'category': 'very_positive', 'context': 'excellence'},
            'phenomenal': {'score': 1.9, 'category': 'very_positive', 'context': 'amazement'},
            'superb': {'score': 1.6, 'category': 'very_positive', 'context': 'quality'},
            
            # Positive (+1.4 to +0.5)
            'excellent': {'score': 1.4, 'category': 'positive', 'context': 'quality'},
            'great': {'score': 1.2, 'category': 'positive', 'context': 'general_approval'},
            'good': {'score': 1.0, 'category': 'positive', 'context': 'general_approval'},
            'excited': {'score': 1.3, 'category': 'positive', 'context': 'anticipation'},
            'happy': {'score': 1.1, 'category': 'positive', 'context': 'emotion'},
            'pleased': {'score': 1.0, 'category': 'positive', 'context': 'satisfaction'},
            'smart': {'score': 0.8, 'category': 'positive', 'context': 'intelligence'},
            'strategic': {'score': 0.7, 'category': 'positive', 'context': 'business'},
            'valuable': {'score': 0.9, 'category': 'positive', 'context': 'worth'},
            'beneficial': {'score': 0.8, 'category': 'positive', 'context': 'advantage'},
            'impressive': {'score': 1.2, 'category': 'positive', 'context': 'admiration'},
            'professional': {'score': 0.6, 'category': 'positive', 'context': 'quality'},
            'successful': {'score': 1.0, 'category': 'positive', 'context': 'achievement'},
            'strong': {'score': 0.7, 'category': 'positive', 'context': 'power'},
            'solid': {'score': 0.6, 'category': 'positive', 'context': 'reliability'},
            'love': {'score': 1.4, 'category': 'positive', 'context': 'emotion'},
            'like': {'score': 0.8, 'category': 'positive', 'context': 'preference'},
            
            # Sports-specific positive
            'loyal': {'score': 0.8, 'category': 'positive', 'context': 'sports_values'},
            'prestigious': {'score': 1.0, 'category': 'positive', 'context': 'status'},
            'iconic': {'score': 1.1, 'category': 'positive', 'context': 'recognition'},
            'legendary': {'score': 1.3, 'category': 'positive', 'context': 'legacy'},
            'champion': {'score': 1.2, 'category': 'positive', 'context': 'achievement'},
            'winning': {'score': 1.0, 'category': 'positive', 'context': 'success'},
            'trophy': {'score': 0.9, 'category': 'positive', 'context': 'achievement'},
            'glory': {'score': 1.1, 'category': 'positive', 'context': 'honor'},
            'pride': {'score': 1.0, 'category': 'positive', 'context': 'emotion'},
            
            # Sponsorship-specific positive
            'partnership': {'score': 0.5, 'category': 'positive', 'context': 'collaboration'},
            'collaboration': {'score': 0.6, 'category': 'positive', 'context': 'cooperation'},
            'investment': {'score': 0.7, 'category': 'positive', 'context': 'commitment'},
            'support': {'score': 0.8, 'category': 'positive', 'context': 'backing'},
            'commitment': {'score': 0.7, 'category': 'positive', 'context': 'dedication'},
            'backing': {'score': 0.6, 'category': 'positive', 'context': 'support'},
            'endorsement': {'score': 0.5, 'category': 'positive', 'context': 'approval'},
            
            # Neutral (0.4 to -0.4)
            'announced': {'score': 0.0, 'category': 'neutral', 'context': 'factual'},
            'confirmed': {'score': 0.0, 'category': 'neutral', 'context': 'factual'},
            'signed': {'score': 0.0, 'category': 'neutral', 'context': 'factual'},
            'agreed': {'score': 0.0, 'category': 'neutral', 'context': 'factual'},
            'extended': {'score': 0.0, 'category': 'neutral', 'context': 'factual'},
            'renewed': {'score': 0.0, 'category': 'neutral', 'context': 'factual'},
            'contract': {'score': 0.0, 'category': 'neutral', 'context': 'business'},
            'deal': {'score': 0.0, 'category': 'neutral', 'context': 'business'},
            'sponsor': {'score': 0.0, 'category': 'neutral', 'context': 'sponsorship'},
            'million': {'score': 0.0, 'category': 'neutral', 'context': 'financial'},
            'billion': {'score': 0.0, 'category': 'neutral', 'context': 'financial'},
            
            # Negative (-0.5 to -1.4)
            'bad': {'score': -1.0, 'category': 'negative', 'context': 'general_disapproval'},
            'terrible': {'score': -1.3, 'category': 'negative', 'context': 'strong_disapproval'},
            'awful': {'score': -1.4, 'category': 'negative', 'context': 'strong_disapproval'},
            'disappointing': {'score': -1.1, 'category': 'negative', 'context': 'unmet_expectations'},
            'poor': {'score': -0.9, 'category': 'negative', 'context': 'quality'},
            'weak': {'score': -0.7, 'category': 'negative', 'context': 'strength'},
            'concerning': {'score': -0.8, 'category': 'negative', 'context': 'worry'},
            'problematic': {'score': -1.0, 'category': 'negative', 'context': 'issues'},
            'questionable': {'score': -0.8, 'category': 'negative', 'context': 'doubt'},
            'overpriced': {'score': -0.9, 'category': 'negative', 'context': 'value'},
            'expensive': {'score': -0.6, 'category': 'negative', 'context': 'cost'},
            'waste': {'score': -1.2, 'category': 'negative', 'context': 'inefficiency'},
            'unnecessary': {'score': -0.7, 'category': 'negative', 'context': 'redundancy'},
            'hate': {'score': -1.4, 'category': 'negative', 'context': 'emotion'},
            'dislike': {'score': -0.8, 'category': 'negative', 'context': 'preference'},
            'disappointed': {'score': -1.0, 'category': 'negative', 'context': 'emotion'},
            
            # Sports-specific negative
            'sellout': {'score': -1.3, 'category': 'negative', 'context': 'betrayal'},
            'commercial': {'score': -0.6, 'category': 'negative', 'context': 'commercialization'},
            'corporate': {'score': -0.7, 'category': 'negative', 'context': 'impersonal'},
            'greedy': {'score': -1.2, 'category': 'negative', 'context': 'greed'},
            'tradition': {'score': -0.5, 'category': 'negative', 'context': 'loss_of_heritage'},
            'history': {'score': -0.4, 'category': 'negative', 'context': 'loss_of_heritage'},
            'soul': {'score': -0.8, 'category': 'negative', 'context': 'loss_of_identity'},
            'authentic': {'score': -0.6, 'category': 'negative', 'context': 'loss_of_authenticity'},
            
            # Very Negative (-1.5 to -2.0)
            'disgusting': {'score': -2.0, 'category': 'very_negative', 'context': 'revulsion'},
            'horrible': {'score': -1.8, 'category': 'very_negative', 'context': 'extreme_disapproval'},
            'pathetic': {'score': -1.7, 'category': 'very_negative', 'context': 'contempt'},
            'ridiculous': {'score': -1.5, 'category': 'very_negative', 'context': 'absurdity'},
            'outrageous': {'score': -1.8, 'category': 'very_negative', 'context': 'anger'},
            'insulting': {'score': -1.6, 'category': 'very_negative', 'context': 'offense'},
            'betrayal': {'score': -1.9, 'category': 'very_negative', 'context': 'trust_violation'},
            'disgraceful': {'score': -1.7, 'category': 'very_negative', 'context': 'shame'},
            'shameful': {'score': -1.6, 'category': 'very_negative', 'context': 'dishonor'},
            'unacceptable': {'score': -1.5, 'category': 'very_negative', 'context': 'rejection'}
        }
    
    def analyze_keyword_sentiments(self):
        """Comprehensive keyword sentiment analysis"""
        logger.info("🔍 Analyzing keyword-level sentiments...")
        
        all_keywords = {}
        phrase_sentiments = {}
        
        for idx, row in self.df.iterrows():
            text = str(row.get('text', ''))
            overall_sentiment = row.get('ensemble_sentiment', 'neutral')
            sentiment_score = row.get('ensemble_score', 0)
            
            # Analyze individual words
            word_sentiments = self._analyze_words(text, overall_sentiment, sentiment_score)
            for word, data in word_sentiments.items():
                if word not in all_keywords:
                    all_keywords[word] = {
                        'total_score': 0,
                        'frequency': 0,
                        'contexts': [],
                        'lexicon_score': data.get('lexicon_score', 0),
                        'vader_score': data.get('vader_score', 0),
                        'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0}
                    }
                
                all_keywords[word]['total_score'] += data['contribution']
                all_keywords[word]['frequency'] += 1
                all_keywords[word]['contexts'].append(data['context'])
                all_keywords[word]['sentiment_distribution'][overall_sentiment] += 1
            
            # Analyze phrases (bigrams and trigrams)
            phrase_sentiments.update(self._analyze_phrases(text, sentiment_score))
        
        # Calculate average scores and confidence
        for word, data in all_keywords.items():
            data['average_score'] = data['total_score'] / data['frequency']
            data['confidence'] = min(data['frequency'] / 10, 1.0)  # Higher confidence with more occurrences
            data['dominant_sentiment'] = max(data['sentiment_distribution'], 
                                           key=data['sentiment_distribution'].get)
        
        self.keyword_scores = all_keywords
        self.phrase_scores = phrase_sentiments
        
        return all_keywords, phrase_sentiments
    
    def _analyze_words(self, text, overall_sentiment, sentiment_score):
        """Analyze individual words in text"""
        word_sentiments = {}
        
        # Tokenize and clean
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha() and len(w) > 2]
        
        for word in words:
            if word in self.stop_words and word not in self.keep_words:
                continue
            
            word_data = {
                'word': word,
                'lexicon_score': 0,
                'vader_score': 0,
                'context_score': 0,
                'contribution': 0,
                'context': self._get_word_context(text, word)
            }
            
            # Get lexicon score
            if word in self.detailed_lexicon:
                word_data['lexicon_score'] = self.detailed_lexicon[word]['score']
            
            # Get VADER score for individual word
            if word in self.sia.lexicon:
                word_data['vader_score'] = self.sia.lexicon[word]
            
            # Calculate context-based contribution
            word_context = word_data['context']
            if word_context:
                context_sentiment = TextBlob(word_context).sentiment.polarity
                word_data['context_score'] = context_sentiment
            
            # Calculate overall contribution to sentence sentiment
            # Weight: 40% lexicon, 30% VADER, 30% context
            word_contribution = (
                0.4 * word_data['lexicon_score'] +
                0.3 * word_data['vader_score'] +
                0.3 * word_data['context_score']
            )
            
            word_data['contribution'] = word_contribution
            word_sentiments[word] = word_data
        
        return word_sentiments
    
    def _analyze_phrases(self, text, sentiment_score):
        """Analyze sentiment of phrases (bigrams and trigrams)"""
        phrase_sentiments = {}
        
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha()]
        
        # Analyze bigrams
        for bigram in ngrams(words, 2):
            phrase = ' '.join(bigram)
            if any(word in self.stop_words for word in bigram):
                continue
            
            phrase_sentiment = TextBlob(phrase).sentiment.polarity
            phrase_sentiments[phrase] = {
                'score': phrase_sentiment,
                'type': 'bigram',
                'frequency': 1
            }
        
        # Analyze trigrams
        for trigram in ngrams(words, 3):
            phrase = ' '.join(trigram)
            if len([w for w in trigram if w in self.stop_words]) > 1:
                continue
            
            phrase_sentiment = TextBlob(phrase).sentiment.polarity
            phrase_sentiments[phrase] = {
                'score': phrase_sentiment,
                'type': 'trigram',
                'frequency': 1
            }
        
        return phrase_sentiments
    
    def _get_word_context(self, text, word, window=3):
        """Get context window around a word"""
        words = text.lower().split()
        try:
            word_indices = [i for i, w in enumerate(words) if word in w]
            if word_indices:
                idx = word_indices[0]
                start = max(0, idx - window)
                end = min(len(words), idx + window + 1)
                return ' '.join(words[start:end])
        except:
            pass
        return ''
    
    def create_keyword_sentiment_visualization(self, output_dir='results'):
        """Create comprehensive keyword sentiment visualizations"""
        logger.info("📊 Creating keyword sentiment visualizations...")
        
        if not self.keyword_scores:
            self.analyze_keyword_sentiments()
        
        # Prepare data for visualization
        viz_data = []
        for word, data in self.keyword_scores.items():
            if data['frequency'] >= 2:  # Only words that appear multiple times
                viz_data.append({
                    'word': word,
                    'average_score': data['average_score'],
                    'frequency': data['frequency'],
                    'confidence': data['confidence'],
                    'lexicon_score': data['lexicon_score'],
                    'vader_score': data['vader_score'],
                    'dominant_sentiment': data['dominant_sentiment'],
                    'total_score': data['total_score']
                })
        
        viz_df = pd.DataFrame(viz_data)
        
        if len(viz_df) == 0:
            logger.warning("No sufficient keyword data for visualization")
            return
        
        # Sort by absolute average score
        viz_df = viz_df.reindex(viz_df['average_score'].abs().sort_values(ascending=False).index)
        
        # Create multiple visualizations
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Keyword Sentiment Scores', 'Frequency vs Sentiment', 
                          'Top Positive Keywords', 'Top Negative Keywords'),
            specs=[[{"secondary_y": True}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. Keyword sentiment scores with confidence
        top_words = viz_df.head(20)
        
        fig.add_trace(
            go.Bar(
                x=top_words['word'],
                y=top_words['average_score'],
                name='Sentiment Score',
                marker=dict(
                    color=top_words['average_score'],
                    colorscale='RdYlGn',
                    colorbar=dict(title="Sentiment Score")
                ),
                hovertemplate='<b>%{x}</b><br>Score: %{y:.3f}<br>Frequency: %{customdata}<extra></extra>',
                customdata=top_words['frequency']
            ),
            row=1, col=1
        )
        
        # Add confidence line
        fig.add_trace(
            go.Scatter(
                x=top_words['word'],
                y=top_words['confidence'],
                mode='lines+markers',
                name='Confidence',
                yaxis='y2',
                line=dict(color='orange', width=2)
            ),
            row=1, col=1, secondary_y=True
        )
        
        # 2. Frequency vs Sentiment scatter
        fig.add_trace(
            go.Scatter(
                x=viz_df['frequency'],
                y=viz_df['average_score'],
                mode='markers',
                text=viz_df['word'],
                name='Keywords',
                marker=dict(
                    size=viz_df['confidence']*20,
                    color=viz_df['average_score'],
                    colorscale='RdYlGn',
                    showscale=True
                ),
                hovertemplate='<b>%{text}</b><br>Frequency: %{x}<br>Score: %{y:.3f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Top positive keywords
        positive_words = viz_df[viz_df['average_score'] > 0].head(10)
        fig.add_trace(
            go.Bar(
                x=positive_words['average_score'],
                y=positive_words['word'],
                orientation='h',
                name='Positive Keywords',
                marker=dict(color='green', opacity=0.7),
                hovertemplate='<b>%{y}</b><br>Score: %{x:.3f}<br>Frequency: %{customdata}<extra></extra>',
                customdata=positive_words['frequency']
            ),
            row=2, col=1
        )
        
        # 4. Top negative keywords
        negative_words = viz_df[viz_df['average_score'] < 0].head(10)
        fig.add_trace(
            go.Bar(
                x=negative_words['average_score'],
                y=negative_words['word'],
                orientation='h',
                name='Negative Keywords',
                marker=dict(color='red', opacity=0.7),
                hovertemplate='<b>%{y}</b><br>Score: %{x:.3f}<br>Frequency: %{customdata}<extra></extra>',
                customdata=negative_words['frequency']
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Keyword Sentiment Analysis Dashboard",
            title_font_size=16,
            showlegend=True
        )
        
        # Update x-axis labels for readability
        fig.update_xaxes(tickangle=45, row=1, col=1)
        
        # Save interactive plot
        import os
        os.makedirs(output_dir, exist_ok=True)
        fig.write_html(f'{output_dir}/keyword_sentiment_dashboard.html')
        
        # Create static matplotlib version for backup
        self._create_static_keyword_plot(viz_df, output_dir)
        
        logger.info(f"Keyword sentiment visualizations saved to {output_dir}/")
    
    def _create_static_keyword_plot(self, viz_df, output_dir):
        """Create static matplotlib version of keyword sentiment plot"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Keyword Sentiment Analysis', fontsize=16, fontweight='bold')
        
        # 1. Top sentiment words
        top_words = viz_df.head(15)
        colors = ['green' if score > 0 else 'red' if score < 0 else 'gray' 
                 for score in top_words['average_score']]
        
        axes[0, 0].barh(range(len(top_words)), top_words['average_score'], color=colors, alpha=0.7)
        axes[0, 0].set_yticks(range(len(top_words)))
        axes[0, 0].set_yticklabels(top_words['word'])
        axes[0, 0].set_xlabel('Average Sentiment Score')
        axes[0, 0].set_title('Top Keywords by Sentiment Impact')
        axes[0, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # 2. Frequency vs sentiment
        scatter = axes[0, 1].scatter(viz_df['frequency'], viz_df['average_score'], 
                                   c=viz_df['average_score'], cmap='RdYlGn', 
                                   s=viz_df['confidence']*100, alpha=0.7)
        axes[0, 1].set_xlabel('Word Frequency')
        axes[0, 1].set_ylabel('Average Sentiment Score')
        axes[0, 1].set_title('Frequency vs Sentiment (size=confidence)')
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.colorbar(scatter, ax=axes[0, 1], label='Sentiment Score')
        
        # 3. Positive word cloud
        positive_words = viz_df[viz_df['average_score'] > 0]
        if len(positive_words) > 0:
            pos_text = ' '.join([f"{row['word']} " * int(row['frequency']) 
                               for _, row in positive_words.iterrows()])
            try:
                wordcloud_pos = WordCloud(width=400, height=300, background_color='white',
                                        colormap='Greens').generate(pos_text)
                axes[1, 0].imshow(wordcloud_pos, interpolation='bilinear')
                axes[1, 0].set_title('Positive Keywords Cloud')
                axes[1, 0].axis('off')
            except:
                axes[1, 0].text(0.5, 0.5, 'Positive Keywords\n' + ', '.join(positive_words['word'].head(10)), 
                              ha='center', va='center', wrap=True)
                axes[1, 0].set_title('Positive Keywords')
        
        # 4. Negative word cloud  
        negative_words = viz_df[viz_df['average_score'] < 0]
        if len(negative_words) > 0:
            neg_text = ' '.join([f"{row['word']} " * int(row['frequency']) 
                               for _, row in negative_words.iterrows()])
            try:
                wordcloud_neg = WordCloud(width=400, height=300, background_color='white',
                                        colormap='Reds').generate(neg_text)
                axes[1, 1].imshow(wordcloud_neg, interpolation='bilinear')
                axes[1, 1].set_title('Negative Keywords Cloud')
                axes[1, 1].axis('off')
            except:
                axes[1, 1].text(0.5, 0.5, 'Negative Keywords\n' + ', '.join(negative_words['word'].head(10)), 
                              ha='center', va='center', wrap=True)
                axes[1, 1].set_title('Negative Keywords')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/keyword_sentiment_static.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_keyword_sentiment_report(self, output_dir='results'):
        """Generate comprehensive keyword sentiment report"""
        logger.info("📄 Generating keyword sentiment report...")
        
        if not self.keyword_scores:
            self.analyze_keyword_sentiments()
        
        # Prepare summary statistics
        total_keywords = len(self.keyword_scores)
        positive_keywords = len([k for k, v in self.keyword_scores.items() if v['average_score'] > 0])
        negative_keywords = len([k for k, v in self.keyword_scores.items() if v['average_score'] < 0])
        neutral_keywords = total_keywords - positive_keywords - negative_keywords
        
        # Get top keywords by category
        sorted_keywords = sorted(self.keyword_scores.items(), 
                               key=lambda x: abs(x[1]['average_score']), reverse=True)
        
        top_positive = [(k, v) for k, v in sorted_keywords if v['average_score'] > 0][:10]
        top_negative = [(k, v) for k, v in sorted_keywords if v['average_score'] < 0][:10]
        most_frequent = sorted(self.keyword_scores.items(), 
                             key=lambda x: x[1]['frequency'], reverse=True)[:15]
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Keyword Sentiment Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 10px; text-align: center; }}
                .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
                .metric-card {{ background: #e8f4f8; padding: 15px; border-radius: 8px; text-align: center; }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
                .keyword-section {{ margin: 20px 0; padding: 15px; border-radius: 8px; }}
                .positive {{ background-color: #d4edda; border-left: 4px solid #28a745; }}
                .negative {{ background-color: #f8d7da; border-left: 4px solid #dc3545; }}
                .neutral {{ background-color: #e2e3e5; border-left: 4px solid #6c757d; }}
                .keyword-table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                .keyword-table th, .keyword-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .keyword-table th {{ background-color: #f2f2f2; }}
                .score-positive {{ color: #28a745; font-weight: bold; }}
                .score-negative {{ color: #dc3545; font-weight: bold; }}
                .score-neutral {{ color: #6c757d; font-weight: bold; }}
                .confidence-high {{ background-color: #d4edda; }}
                .confidence-medium {{ background-color: #fff3cd; }}
                .confidence-low {{ background-color: #f8d7da; }}
                .keyword-detail {{ background-color: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 Keyword Sentiment Analysis Report</h1>
                <h2>Individual Word and Phrase Sentiment Scoring</h2>
                <p>Understanding which words drive positive and negative sentiment in sports sponsorship discussions</p>
            </div>
            
            <div class="summary">
                <div class="metric-card">
                    <div class="metric-value">{total_keywords}</div>
                    <div>Total Keywords Analyzed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color: #28a745;">{positive_keywords}</div>
                    <div>Positive Keywords</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color: #dc3545;">{negative_keywords}</div>
                    <div>Negative Keywords</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color: #6c757d;">{neutral_keywords}</div>
                    <div>Neutral Keywords</div>
                </div>
            </div>
        """
        
        # Add top positive keywords section
        html_content += """
            <div class="keyword-section positive">
                <h2>🟢 Top Positive Sentiment Keywords</h2>
                <p>Words that most strongly contribute to positive sentiment in sponsorship discussions:</p>
                <table class="keyword-table">
                    <tr>
                        <th>Keyword</th>
                        <th>Sentiment Score</th>
                        <th>Frequency</th>
                        <th>Confidence</th>
                        <th>Context</th>
                    </tr>
        """
        
        for word, data in top_positive:
            confidence_class = ('confidence-high' if data['confidence'] > 0.7 else 
                              'confidence-medium' if data['confidence'] > 0.4 else 'confidence-low')
            context = self.detailed_lexicon.get(word, {}).get('context', 'general')
            
            html_content += f"""
                    <tr class="{confidence_class}">
                        <td><strong>{word}</strong></td>
                        <td class="score-positive">+{data['average_score']:.3f}</td>
                        <td>{data['frequency']}</td>
                        <td>{data['confidence']:.2f}</td>
                        <td>{context.replace('_', ' ').title()}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="keyword-section negative">
                <h2>🔴 Top Negative Sentiment Keywords</h2>
                <p>Words that most strongly contribute to negative sentiment in sponsorship discussions:</p>
                <table class="keyword-table">
                    <tr>
                        <th>Keyword</th>
                        <th>Sentiment Score</th>
                        <th>Frequency</th>
                        <th>Confidence</th>
                        <th>Context</th>
                    </tr>
        """
        
        for word, data in top_negative:
            confidence_class = ('confidence-high' if data['confidence'] > 0.7 else 
                              'confidence-medium' if data['confidence'] > 0.4 else 'confidence-low')
            context = self.detailed_lexicon.get(word, {}).get('context', 'general')
            
            html_content += f"""
                    <tr class="{confidence_class}">
                        <td><strong>{word}</strong></td>
                        <td class="score-negative">{data['average_score']:.3f}</td>
                        <td>{data['frequency']}</td>
                        <td>{data['confidence']:.2f}</td>
                        <td>{context.replace('_', ' ').title()}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="keyword-section neutral">
                <h2>📊 Most Frequently Used Keywords</h2>
                <p>Words that appear most often in sponsorship discussions:</p>
                <table class="keyword-table">
                    <tr>
                        <th>Keyword</th>
                        <th>Frequency</th>
                        <th>Sentiment Score</th>
                        <th>Dominant Sentiment</th>
                        <th>Distribution</th>
                    </tr>
        """
        
        for word, data in most_frequent:
            sentiment_class = ('score-positive' if data['average_score'] > 0 else 
                             'score-negative' if data['average_score'] < 0 else 'score-neutral')
            
            pos_pct = data['sentiment_distribution']['positive'] / data['frequency'] * 100
            neg_pct = data['sentiment_distribution']['negative'] / data['frequency'] * 100
            neu_pct = data['sentiment_distribution']['neutral'] / data['frequency'] * 100
            
            html_content += f"""
                    <tr>
                        <td><strong>{word}</strong></td>
                        <td>{data['frequency']}</td>
                        <td class="{sentiment_class}">{data['average_score']:.3f}</td>
                        <td>{data['dominant_sentiment'].title()}</td>
                        <td>
                            <small>
                                Pos: {pos_pct:.0f}% | 
                                Neg: {neg_pct:.0f}% | 
                                Neu: {neu_pct:.0f}%
                            </small>
                        </td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="keyword-section">
                <h2>🎯 Keyword Scoring Methodology</h2>
                <div class="keyword-detail">
                    <h3>Multi-Method Scoring Approach</h3>
                    <p>Each keyword receives a sentiment score based on multiple analysis methods:</p>
                    <ul>
                        <li><strong>Custom Sports Lexicon (40%):</strong> Domain-specific sentiment scores for sports/sponsorship terms</li>
                        <li><strong>VADER Lexicon (30%):</strong> Social media optimized sentiment scores</li>
                        <li><strong>Context Analysis (30%):</strong> Sentiment of surrounding words and phrases</li>
                    </ul>
                </div>
                
                <div class="keyword-detail">
                    <h3>Confidence Scoring</h3>
                    <p>Confidence levels indicate reliability of sentiment scores:</p>
                    <ul>
                        <li><strong class="score-positive">High Confidence (0.7+):</strong> Word appears 7+ times, consistent sentiment pattern</li>
                        <li><strong style="color: #ffc107;">Medium Confidence (0.4-0.7):</strong> Word appears 4-6 times, mostly consistent</li>
                        <li><strong class="score-negative">Low Confidence (<0.4):</strong> Word appears 2-3 times, may have mixed usage</li>
                    </ul>
                </div>
                
                <div class="keyword-detail">
                    <h3>Score Interpretation</h3>
                    <ul>
                        <li><strong class="score-positive">+1.5 to +2.0:</strong> Very strong positive sentiment (amazing, incredible, perfect)</li>
                        <li><strong class="score-positive">+0.5 to +1.4:</strong> Positive sentiment (good, great, excited, smart)</li>
                        <li><strong class="score-neutral">-0.4 to +0.4:</strong> Neutral/factual (announced, deal, contract, million)</li>
                        <li><strong class="score-negative">-1.4 to -0.5:</strong> Negative sentiment (bad, disappointing, waste)</li>
                        <li><strong class="score-negative">-2.0 to -1.5:</strong> Very strong negative sentiment (disgusting, betrayal, outrageous)</li>
                    </ul>
                </div>
            </div>
            
            <div class="keyword-section">
                <h2>📈 Key Insights for Sponsorship Communications</h2>
                <div class="keyword-detail">
                    <h3>🟢 Leverage Positive Language</h3>
                    <p>Use these high-impact positive words in sponsorship announcements:</p>
                    <ul>
        """
        
        # Add top positive words as recommendations
        for word, data in top_positive[:5]:
            html_content += f"<li><strong>{word}</strong> (Score: +{data['average_score']:.2f}) - Use in context of {self.detailed_lexicon.get(word, {}).get('context', 'general').replace('_', ' ')}</li>"
        
        html_content += """
                    </ul>
                </div>
                
                <div class="keyword-detail">
                    <h3>🔴 Avoid Negative Triggers</h3>
                    <p>Be careful with these words that trigger negative sentiment:</p>
                    <ul>
        """
        
        # Add top negative words as warnings
        for word, data in top_negative[:5]:
            html_content += f"<li><strong>{word}</strong> (Score: {data['average_score']:.2f}) - Often associated with {self.detailed_lexicon.get(word, {}).get('context', 'general').replace('_', ' ')}</li>"
        
        html_content += """
                    </ul>
                </div>
                
                <div class="keyword-detail">
                    <h3>💡 Strategic Recommendations</h3>
                    <ul>
                        <li><strong>Frame partnerships positively:</strong> Use words like "partnership," "collaboration," "investment" rather than just "deal" or "contract"</li>
                        <li><strong>Emphasize mutual benefits:</strong> Highlight "support," "commitment," and "shared values"</li>
                        <li><strong>Address tradition concerns:</strong> When discussing naming rights, acknowledge "history" while emphasizing "future" and "growth"</li>
                        <li><strong>Avoid commercial language:</strong> Minimize words like "commercial," "corporate," or "money-grab" in communications</li>
                        <li><strong>Use emotional positive terms:</strong> Words like "excited," "proud," and "amazing" generate strong positive responses</li>
                    </ul>
                </div>
            </div>
            
            <div class="keyword-section">
                <h2>📊 Interactive Visualizations</h2>
                <p>For detailed interactive analysis, see:</p>
                <ul>
                    <li><a href="keyword_sentiment_dashboard.html">Interactive Keyword Dashboard</a> - Explore all keywords with filtering and drill-down</li>
                    <li><a href="keyword_sentiment_static.png">Static Visualization</a> - Summary charts and word clouds</li>
                </ul>
            </div>
            
        </body>
        </html>
        """
        
        # Save report
        import os
        os.makedirs(output_dir, exist_ok=True)
        with open(f'{output_dir}/keyword_sentiment_report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Save keyword data as CSV for further analysis
        self._save_keyword_data_csv(output_dir)
        
        logger.info(f"Keyword sentiment report saved to {output_dir}/keyword_sentiment_report.html")
        
        return {
            'total_keywords': total_keywords,
            'positive_keywords': positive_keywords,
            'negative_keywords': negative_keywords,
            'top_positive': top_positive,
            'top_negative': top_negative,
            'most_frequent': most_frequent
        }
    
    def _save_keyword_data_csv(self, output_dir):
        """Save detailed keyword data as CSV files"""
        
        # Main keyword data
        keyword_data = []
        for word, data in self.keyword_scores.items():
            keyword_data.append({
                'keyword': word,
                'average_sentiment_score': data['average_score'],
                'frequency': data['frequency'],
                'confidence': data['confidence'],
                'lexicon_score': data['lexicon_score'],
                'vader_score': data['vader_score'],
                'dominant_sentiment': data['dominant_sentiment'],
                'positive_contexts': data['sentiment_distribution']['positive'],
                'negative_contexts': data['sentiment_distribution']['negative'],
                'neutral_contexts': data['sentiment_distribution']['neutral'],
                'total_score_contribution': data['total_score'],
                'sentiment_category': self.detailed_lexicon.get(word, {}).get('category', 'unknown'),
                'context_type': self.detailed_lexicon.get(word, {}).get('context', 'general')
            })
        
        keyword_df = pd.DataFrame(keyword_data)
        keyword_df = keyword_df.sort_values('average_sentiment_score', key=abs, ascending=False)
        keyword_df.to_csv(f'{output_dir}/detailed_keyword_sentiments.csv', index=False)
        
        # Phrase data if available
        if self.phrase_scores:
            phrase_data = []
            for phrase, data in self.phrase_scores.items():
                phrase_data.append({
                    'phrase': phrase,
                    'sentiment_score': data['score'],
                    'phrase_type': data['type'],
                    'frequency': data['frequency']
                })
            
            phrase_df = pd.DataFrame(phrase_data)
            phrase_df = phrase_df.sort_values('sentiment_score', key=abs, ascending=False)
            phrase_df.to_csv(f'{output_dir}/phrase_sentiments.csv', index=False)
        
        # Summary statistics
        summary_stats = {
            'total_keywords_analyzed': len(self.keyword_scores),
            'positive_keywords_count': len([k for k, v in self.keyword_scores.items() if v['average_score'] > 0]),
            'negative_keywords_count': len([k for k, v in self.keyword_scores.items() if v['average_score'] < 0]),
            'neutral_keywords_count': len([k for k, v in self.keyword_scores.items() if abs(v['average_score']) <= 0.1]),
            'high_confidence_keywords': len([k for k, v in self.keyword_scores.items() if v['confidence'] > 0.7]),
            'average_sentiment_score': np.mean([v['average_score'] for v in self.keyword_scores.values()]),
            'most_positive_keyword': max(self.keyword_scores.items(), key=lambda x: x[1]['average_score'])[0],
            'most_negative_keyword': min(self.keyword_scores.items(), key=lambda x: x[1]['average_score'])[0],
            'most_frequent_keyword': max(self.keyword_scores.items(), key=lambda x: x[1]['frequency'])[0]
        }
        
        pd.DataFrame([summary_stats]).to_csv(f'{output_dir}/keyword_analysis_summary.csv', index=False)
        
        logger.info(f"Keyword data saved to CSV files in {output_dir}/")


def analyze_keyword_sentiments_comprehensive(df, config=None, output_dir='results'):
    """
    Main function to perform comprehensive keyword sentiment analysis
    
    Args:
        df: DataFrame with sentiment analysis results
        config: Configuration object (optional)
        output_dir: Directory to save outputs
    
    Returns:
        Dictionary with analysis results and file paths
    """
    
    logger.info("🚀 Starting Comprehensive Keyword Sentiment Analysis...")
    logger.info("="*60)
    
    # Initialize analyzer
    analyzer = KeywordSentimentScorer(df, config)
    
    # Perform analysis
    keyword_scores, phrase_scores = analyzer.analyze_keyword_sentiments()
    
    # Create visualizations
    analyzer.create_keyword_sentiment_visualization(output_dir)
    
    # Generate comprehensive report
    report_results = analyzer.generate_keyword_sentiment_report(output_dir)
    
    logger.info("="*60)
    logger.info("✅ Comprehensive Keyword Sentiment Analysis Complete!")
    logger.info(f"📊 Analyzed {len(keyword_scores)} unique keywords")
    logger.info(f"🟢 Positive keywords: {report_results['positive_keywords']}")
    logger.info(f"🔴 Negative keywords: {report_results['negative_keywords']}")
    
    # Safe access to top keywords with error handling
    if report_results['top_positive']:
        top_pos_word = report_results['top_positive'][0][0]
        top_pos_score = report_results['top_positive'][0][1]['average_score']
        logger.info(f"📈 Most positive: '{top_pos_word}' (+{top_pos_score:.3f})")
    else:
        logger.info("📈 Most positive: No positive keywords found")
    
    if report_results['top_negative']:
        top_neg_word = report_results['top_negative'][0][0]
        top_neg_score = report_results['top_negative'][0][1]['average_score']
        logger.info(f"📉 Most negative: '{top_neg_word}' ({top_neg_score:.3f})")
    else:
        logger.info("📉 Most negative: No negative keywords found")
    
    logger.info("="*60)
    logger.info(f"📁 Reports generated:")
    logger.info(f"   🌐 Interactive: {output_dir}/keyword_sentiment_dashboard.html")
    logger.info(f"   📄 Detailed Report: {output_dir}/keyword_sentiment_report.html")
    logger.info(f"   📊 Static Charts: {output_dir}/keyword_sentiment_static.png")
    logger.info(f"   📋 Data Files: {output_dir}/detailed_keyword_sentiments.csv")
    logger.info("="*60)
    
    return {
        'keyword_scores': keyword_scores,
        'phrase_scores': phrase_scores,
        'report_summary': report_results,
        'output_files': {
            'interactive_dashboard': f'{output_dir}/keyword_sentiment_dashboard.html',
            'detailed_report': f'{output_dir}/keyword_sentiment_report.html',
            'static_visualization': f'{output_dir}/keyword_sentiment_static.png',
            'keyword_data_csv': f'{output_dir}/detailed_keyword_sentiments.csv',
            'phrase_data_csv': f'{output_dir}/phrase_sentiments.csv',
            'summary_csv': f'{output_dir}/keyword_analysis_summary.csv'
        }
    }


# Usage example and feature summary
print("📊 ADVANCED KEYWORD SENTIMENT SCORER READY!")
print("="*60)
print("✨ Advanced Features:")
print("   🎯 Individual keyword sentiment scoring")
print("   📈 Multi-method scoring (Custom + VADER + Context)")
print("   🔍 Confidence levels based on frequency")
print("   📊 Interactive visualizations with Plotly")
print("   📄 Comprehensive HTML reports")
print("   💾 Detailed CSV exports for further analysis")
print("\\n🔬 Scoring Methods:")
print("   • Custom Sports Sponsorship Lexicon (40% weight)")
print("   • VADER Social Media Lexicon (30% weight)")
print("   • Contextual Analysis (30% weight)")
print("\\n📈 Outputs:")
print("   • Word-level sentiment scores (-2.0 to +2.0)")
print("   • Confidence ratings (0.0 to 1.0)")
print("   • Frequency analysis")
print("   • Context categorization")
print("   • Strategic recommendations")
print("\\n💡 Use: analyze_keyword_sentiments_comprehensive(df)")
print("="*60)