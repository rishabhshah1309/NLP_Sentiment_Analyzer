import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging

# Set up logging
logger = logging.getLogger(__name__)

class SentimentKeywordAnalyzer:
    """
    Analyzes keywords and posts that drive sentiment classifications
    """
    
    def __init__(self, df, text_column='processed_text', sentiment_column='ensemble_sentiment'):
        self.df = df
        self.text_column = text_column
        self.sentiment_column = sentiment_column
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        # Add sports-specific stopwords
        self.stop_words.update(['team', 'game', 'player', 'sport', 'fan', 'fans'])
        
    def extract_sentiment_keywords(self, max_features=1000, ngram_range=(1, 2)):
        """
        Extract keywords most associated with each sentiment using TF-IDF and Chi-square
        """
        logger.info("Extracting sentiment-specific keywords...")
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        
        # Fit on all text
        X = vectorizer.fit_transform(self.df[self.text_column].fillna(''))
        feature_names = vectorizer.get_feature_names_out()
        
        # Get sentiment labels and encode them
        y = self.df[self.sentiment_column]
        unique_sentiments = list(y.unique())
        
        # If we have less than 2 classes, we can't do chi-square
        if len(unique_sentiments) < 2:
            logger.warning("Less than 2 sentiment classes found. Using frequency-based analysis instead.")
            return self._fallback_keyword_extraction(unique_sentiments[0])
        
        # Encode labels as numbers for chi2
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Calculate chi-square scores
        chi2_scores, p_values = chi2(X, y_encoded)
        
        # Create results dictionary
        sentiment_keywords = {}
        
        for i, sentiment in enumerate(unique_sentiments):
            # Handle different shapes of chi2_scores
            if len(unique_sentiments) == 2:
                # For binary classification, chi2 returns 1D array
                scores = chi2_scores
                p_vals = p_values
            else:
                # For multi-class, chi2 returns 2D array
                scores = chi2_scores[:, i] if chi2_scores.ndim > 1 else chi2_scores
                p_vals = p_values[:, i] if p_values.ndim > 1 else p_values
            
            # Get top keywords for this sentiment
            if len(scores) > 0:
                top_indices = np.argsort(scores)[-min(50, len(scores)):][::-1]
                
                sentiment_keywords[sentiment] = {
                    'keywords': [feature_names[idx] for idx in top_indices],
                    'scores': [float(scores[idx]) for idx in top_indices],
                    'p_values': [float(p_vals[idx]) for idx in top_indices]
                }
            else:
                sentiment_keywords[sentiment] = {
                    'keywords': [],
                    'scores': [],
                    'p_values': []
                }
        
        return sentiment_keywords
    
    def _fallback_keyword_extraction(self, sentiment):
        """Fallback method when chi-square isn't applicable"""
        logger.info("Using TF-IDF scores as fallback...")
        
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        
        X = vectorizer.fit_transform(self.df[self.text_column].fillna(''))
        feature_names = vectorizer.get_feature_names_out()
        
        # Get mean TF-IDF scores
        mean_scores = np.array(X.mean(axis=0)).flatten()
        top_indices = np.argsort(mean_scores)[-50:][::-1]
        
        return {
            sentiment: {
                'keywords': [feature_names[i] for i in top_indices],
                'scores': [float(mean_scores[i]) for i in top_indices],
                'p_values': [0.0] * len(top_indices)  # Not applicable for TF-IDF
            }
        }
    
    def analyze_keyword_frequency(self):
        """
        Analyze frequency of keywords within each sentiment class
        """
        logger.info("Analyzing keyword frequencies by sentiment...")
        
        keyword_freq = {}
        
        for sentiment in self.df[self.sentiment_column].unique():
            # Get text for this sentiment
            sentiment_texts = self.df[self.df[self.sentiment_column] == sentiment][self.text_column]
            
            # Combine all text
            combined_text = ' '.join(sentiment_texts.fillna('').astype(str))
            
            # Tokenize and count
            tokens = word_tokenize(combined_text.lower())
            tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
            
            # Count frequencies
            freq_dist = Counter(tokens)
            keyword_freq[sentiment] = freq_dist.most_common(30)
        
        return keyword_freq
    
    def get_representative_posts(self, n_posts=5):
        """
        Get most representative posts for each sentiment category
        """
        logger.info("Extracting representative posts for each sentiment...")
        
        representative_posts = {}
        
        for sentiment in self.df[self.sentiment_column].unique():
            sentiment_df = self.df[self.df[self.sentiment_column] == sentiment]
            
            # Sort by confidence/intensity (using VADER compound score as proxy)
            if 'vader_compound' in sentiment_df.columns:
                if sentiment == 'positive':
                    top_posts = sentiment_df.nlargest(n_posts, 'vader_compound')
                elif sentiment == 'negative':
                    top_posts = sentiment_df.nsmallest(n_posts, 'vader_compound')
                else:  # neutral
                    # Get posts closest to 0
                    top_posts = sentiment_df.loc[sentiment_df['vader_compound'].abs().nsmallest(n_posts).index]
            else:
                # Random sample if no confidence scores
                top_posts = sentiment_df.sample(min(n_posts, len(sentiment_df)))
            
            representative_posts[sentiment] = top_posts[['text', 'vader_compound', 'platform']].to_dict('records')
        
        return representative_posts
    
    def create_sentiment_wordclouds(self, output_dir='results'):
        """
        Create word clouds for each sentiment
        """
        logger.info("Creating sentiment-specific word clouds...")
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        sentiments = ['positive', 'negative', 'neutral']
        colormaps = ['Greens', 'Reds', 'Greys']
        
        for i, sentiment in enumerate(sentiments):
            if sentiment in self.df[self.sentiment_column].values:
                # Get text for this sentiment
                sentiment_texts = self.df[self.df[self.sentiment_column] == sentiment][self.text_column]
                combined_text = ' '.join(sentiment_texts.fillna('').astype(str))
                
                # Create word cloud
                wordcloud = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    colormap=colormaps[i],
                    max_words=100,
                    stopwords=self.stop_words
                ).generate(combined_text)
                
                axes[i].imshow(wordcloud, interpolation='bilinear')
                axes[i].set_title(f'{sentiment.title()} Sentiment Keywords', fontsize=14, fontweight='bold')
                axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f'No {sentiment} posts', ha='center', va='center')
                axes[i].set_title(f'{sentiment.title()} Sentiment Keywords', fontsize=14)
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sentiment_wordclouds.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Word clouds saved to {output_dir}/sentiment_wordclouds.png")
    
    def analyze_sponsorship_keywords(self):
        """
        Analyze keywords specific to sponsorship-related posts
        """
        logger.info("Analyzing sponsorship-specific keywords...")
        
        if 'sponsorship_type' not in self.df.columns:
            logger.warning("No sponsorship_type column found")
            return {}
        
        sponsorship_keywords = {}
        
        # Get only sponsorship posts (not 'not_sponsorship')
        sponsorship_df = self.df[self.df['sponsorship_type'] != 'not_sponsorship']
        
        for stype in sponsorship_df['sponsorship_type'].unique():
            type_df = sponsorship_df[sponsorship_df['sponsorship_type'] == stype]
            
            # Analyze sentiment within this sponsorship type
            sentiment_analysis = {}
            for sentiment in type_df[self.sentiment_column].unique():
                sentiment_texts = type_df[type_df[self.sentiment_column] == sentiment][self.text_column]
                combined_text = ' '.join(sentiment_texts.fillna('').astype(str))
                
                # Extract keywords
                tokens = word_tokenize(combined_text.lower())
                tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
                
                sentiment_analysis[sentiment] = Counter(tokens).most_common(15)
            
            sponsorship_keywords[stype] = sentiment_analysis
        
        return sponsorship_keywords
    
    def generate_keyword_report(self, output_dir='results'):
        """
        Generate comprehensive keyword analysis report
        """
        logger.info("Generating keyword analysis report...")
        
        # Get all analyses
        sentiment_keywords = self.extract_sentiment_keywords()
        keyword_freq = self.analyze_keyword_frequency()
        representative_posts = self.get_representative_posts()
        sponsorship_keywords = self.analyze_sponsorship_keywords()
        
        # Create detailed HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Keyword and Post Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 10px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border-radius: 8px; }}
                .positive {{ background-color: #d4edda; border-left: 4px solid #28a745; }}
                .negative {{ background-color: #f8d7da; border-left: 4px solid #dc3545; }}
                .neutral {{ background-color: #e2e3e5; border-left: 4px solid #6c757d; }}
                .keyword {{ display: inline-block; background-color: #007bff; color: white; padding: 3px 8px; margin: 2px; border-radius: 12px; font-size: 0.9em; }}
                .post {{ background-color: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; font-style: italic; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .score {{ font-weight: bold; color: #007bff; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 Keyword & Post Analysis Report</h1>
                <h2>What Drives Fan Sentiment in Sports Sponsorships</h2>
                <p>Understanding the language and content behind sentiment classifications</p>
            </div>
        """
        
        # Add sentiment keyword analysis
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in sentiment_keywords:
                html_content += f"""
                <div class="section {sentiment}">
                    <h2>📊 {sentiment.title()} Sentiment Keywords</h2>
                    <h3>Top Discriminative Keywords:</h3>
                    <div>
                """
                
                # Add top keywords as tags
                for keyword in sentiment_keywords[sentiment]['keywords'][:20]:
                    html_content += f'<span class="keyword">{keyword}</span>'
                
                html_content += f"""
                    </div>
                    <h3>Most Representative Posts:</h3>
                """
                
                # Add representative posts
                if sentiment in representative_posts:
                    for post in representative_posts[sentiment]:
                        html_content += f"""
                        <div class="post">
                            "{post['text'][:300]}{'...' if len(post['text']) > 300 else ''}"
                            <br><small><strong>Platform:</strong> {post['platform']} | <strong>Sentiment Score:</strong> {post['vader_compound']:.3f}</small>
                        </div>
                        """
                
                html_content += "</div>"
        
        # Add sponsorship-specific analysis
        if sponsorship_keywords:
            html_content += """
            <div class="section">
                <h2>🎪 Sponsorship Type Keyword Analysis</h2>
            """
            
            for stype, analysis in sponsorship_keywords.items():
                html_content += f"""
                <h3>{stype.replace('_', ' ').title()} Sponsorships</h3>
                <table>
                    <tr><th>Sentiment</th><th>Top Keywords</th><th>Frequency</th></tr>
                """
                
                for sentiment, keywords in analysis.items():
                    keyword_list = ', '.join([f"{word} ({count})" for word, count in keywords[:10]])
                    html_content += f"""
                    <tr>
                        <td><strong>{sentiment.title()}</strong></td>
                        <td>{keyword_list}</td>
                        <td>{sum([count for word, count in keywords])}</td>
                    </tr>
                    """
                
                html_content += "</table>"
            
            html_content += "</div>"
        
        # Add insights section
        html_content += """
        <div class="section">
            <h2>💡 Key Insights</h2>
            <ul>
                <li><strong>Positive Drivers:</strong> Look for keywords like partnership, support, excited, great, love</li>
                <li><strong>Negative Drivers:</strong> Watch for sellout, money, corporate, disappointing, hate</li>
                <li><strong>Neutral Indicators:</strong> Factual terms like announced, deal, contract, agreement</li>
                <li><strong>Platform Differences:</strong> Twitter tends to be more emotional, Reddit more analytical</li>
                <li><strong>Timing Matters:</strong> Initial reactions often more extreme than long-term sentiment</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>🎯 Recommendations</h2>
            <ul>
                <li><strong>Monitor Negative Keywords:</strong> Set up alerts for terms that predict negative sentiment</li>
                <li><strong>Emphasize Positive Themes:</strong> Use language that historically generates positive responses</li>
                <li><strong>Address Concerns Proactively:</strong> Respond to common negative themes before they spread</li>
                <li><strong>Platform-Specific Messaging:</strong> Tailor announcements to each platform's audience</li>
                <li><strong>Timing Strategy:</strong> Consider when and how to announce sponsorships for maximum positive impact</li>
            </ul>
        </div>
        
        </body>
        </html>
        """
        
        # Save report
        import os
        os.makedirs(output_dir, exist_ok=True)
        with open(f'{output_dir}/keyword_analysis_report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Also save raw data as CSV for further analysis
        self._save_keyword_data(sentiment_keywords, keyword_freq, output_dir)
        
        logger.info(f"Keyword analysis report saved to {output_dir}/keyword_analysis_report.html")
        
        return {
            'sentiment_keywords': sentiment_keywords,
            'keyword_freq': keyword_freq,
            'representative_posts': representative_posts,
            'sponsorship_keywords': sponsorship_keywords
        }
    
    def _save_keyword_data(self, sentiment_keywords, keyword_freq, output_dir):
        """Save keyword data as CSV files for further analysis"""
        
        # Save sentiment keywords
        sentiment_kw_data = []
        for sentiment, data in sentiment_keywords.items():
            for i, keyword in enumerate(data['keywords']):
                sentiment_kw_data.append({
                    'sentiment': sentiment,
                    'keyword': keyword,
                    'chi2_score': data['scores'][i],
                    'p_value': data['p_values'][i],
                    'rank': i + 1
                })
        
        pd.DataFrame(sentiment_kw_data).to_csv(f'{output_dir}/sentiment_keywords.csv', index=False)
        
        # Save frequency data
        freq_data = []
        for sentiment, keywords in keyword_freq.items():
            for keyword, freq in keywords:
                freq_data.append({
                    'sentiment': sentiment,
                    'keyword': keyword,
                    'frequency': freq
                })
        
        pd.DataFrame(freq_data).to_csv(f'{output_dir}/keyword_frequencies.csv', index=False)
        
        logger.info(f"Keyword data saved to {output_dir}/")

# Example usage function to integrate with your main script
def analyze_sentiment_drivers(df, output_dir='results'):
    """
    Main function to analyze what drives sentiment in your sponsorship data
    
    Args:
        df: DataFrame with sentiment analysis results
        output_dir: Directory to save outputs
    
    Returns:
        Dictionary with all analysis results
    """
    
    # Initialize analyzer
    analyzer = SentimentKeywordAnalyzer(df)
    
    # Create word clouds
    analyzer.create_sentiment_wordclouds(output_dir)
    
    # Generate comprehensive report
    results = analyzer.generate_keyword_report(output_dir)
    
    print("\\n🔍 KEYWORD & POST ANALYSIS COMPLETE")
    print("="*50)
    print(f"📊 Reports saved to {output_dir}/")
    print(f"📈 Word clouds: sentiment_wordclouds.png")
    print(f"📄 Detailed report: keyword_analysis_report.html")
    print(f"📋 Raw data: sentiment_keywords.csv, keyword_frequencies.csv")
    
    return results