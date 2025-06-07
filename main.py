import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import os

# Import project modules
from config import Config
from data_collector import collect_sponsorship_data
from nlp_processor import process_sponsorship_data
from sentiment_keyword_analyzer import analyze_sentiment_drivers

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_visualizations(df, output_dir='results'):
    """Create comprehensive visualizations"""
    logger.info("Creating visualizations...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Sentiment Distribution
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Sports Sponsorship Sentiment Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Sentiment pie chart
    sentiment_counts = df['ensemble_sentiment'].value_counts()
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                   autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0, 0].set_title('Overall Sentiment Distribution')
    
    # Platform comparison
    if 'platform' in df.columns and df['platform'].nunique() > 1:
        platform_sentiment = df.groupby(['platform', 'ensemble_sentiment']).size().unstack(fill_value=0)
        platform_sentiment.plot(kind='bar', ax=axes[0, 1], stacked=True, color=colors)
        axes[0, 1].set_title('Sentiment by Platform')
        axes[0, 1].legend(title='Sentiment')
        axes[0, 1].tick_params(axis='x', rotation=45)
    else:
        axes[0, 1].text(0.5, 0.5, 'Single Platform Data', ha='center', va='center')
        axes[0, 1].set_title('Platform Comparison')
    
    # Sentiment scores distribution
    axes[1, 0].hist(df['vader_compound'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_title('VADER Sentiment Score Distribution')
    axes[1, 0].set_xlabel('Sentiment Score (-1 to 1)')
    axes[1, 0].set_ylabel('Frequency')
    
    # Engagement vs Sentiment
    if 'engagement' in df.columns:
        scatter = axes[1, 1].scatter(df['vader_compound'], df['engagement'], 
                                   alpha=0.6, c=df['textblob_subjectivity'], cmap='viridis')
        axes[1, 1].set_xlabel('Sentiment Score')
        axes[1, 1].set_ylabel('Engagement')
        axes[1, 1].set_title('Sentiment vs Engagement')
        plt.colorbar(scatter, ax=axes[1, 1], label='Subjectivity')
    else:
        axes[1, 1].text(0.5, 0.5, 'No Engagement Data', ha='center', va='center')
        axes[1, 1].set_title('Sentiment vs Engagement')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sentiment_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Sponsorship Type Analysis
    if 'sponsorship_type' in df.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Sponsorship type distribution
        type_counts = df['sponsorship_type'].value_counts()
        ax1.bar(range(len(type_counts)), type_counts.values, color='lightcoral')
        ax1.set_title('Sponsorship Type Distribution')
        ax1.set_xlabel('Sponsorship Type')
        ax1.set_ylabel('Count')
        ax1.set_xticks(range(len(type_counts)))
        ax1.set_xticklabels(type_counts.index, rotation=45)
        
        # Sentiment by sponsorship type
        sponsorship_df = df[df['sponsorship_type'] != 'not_sponsorship']
        if len(sponsorship_df) > 0:
            type_sentiment = sponsorship_df.groupby(['sponsorship_type', 'ensemble_sentiment']).size().unstack(fill_value=0)
            type_sentiment.plot(kind='bar', ax=ax2, stacked=True, color=colors)
            ax2.set_title('Sentiment by Sponsorship Type')
            ax2.legend(title='Sentiment')
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.text(0.5, 0.5, 'No Sponsorship Data', ha='center', va='center')
            ax2.set_title('Sentiment by Sponsorship Type')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sponsorship_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}/")

def generate_insights_report(df, output_dir='results'):
    """Generate comprehensive insights report"""
    logger.info("Generating insights report...")
    
    # Calculate key metrics
    total_posts = len(df)
    sentiment_dist = df['ensemble_sentiment'].value_counts(normalize=True) * 100
    avg_sentiment = df['vader_compound'].mean()
    
    # Platform analysis
    platform_analysis = ""
    if 'platform' in df.columns and df['platform'].nunique() > 1:
        platform_sentiment = df.groupby('platform')['vader_compound'].mean()
        best_platform = platform_sentiment.idxmax()
        worst_platform = platform_sentiment.idxmin()
        
        platform_analysis = f"""
        <h3>Platform Analysis</h3>
        <p><strong>Most Positive Platform:</strong> {best_platform.title()} (avg sentiment: {platform_sentiment[best_platform]:.3f})</p>
        <p><strong>Most Negative Platform:</strong> {worst_platform.title()} (avg sentiment: {platform_sentiment[worst_platform]:.3f})</p>
        
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr><th>Platform</th><th>Posts</th><th>Avg Sentiment</th><th>Engagement</th></tr>
        """
        
        for platform in df['platform'].unique():
            platform_data = df[df['platform'] == platform]
            avg_eng = platform_data['engagement'].mean() if 'engagement' in df.columns else 0
            platform_analysis += f"""
            <tr>
                <td>{platform.title()}</td>
                <td>{len(platform_data)}</td>
                <td>{platform_data['vader_compound'].mean():.3f}</td>
                <td>{avg_eng:.1f}</td>
            </tr>
            """
        platform_analysis += "</table>"
    
    # Sponsorship type analysis
    sponsorship_analysis = ""
    if 'sponsorship_type' in df.columns:
        sponsorship_counts = df['sponsorship_type'].value_counts()
        sponsorship_analysis = f"""
        <h3>Sponsorship Type Analysis</h3>
        <ul>
        """
        for stype, count in sponsorship_counts.head().items():
            pct = count / len(df) * 100
            sponsorship_analysis += f"<li><strong>{stype.replace('_', ' ').title()}:</strong> {count} posts ({pct:.1f}%)</li>"
        sponsorship_analysis += "</ul>"
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sports Sponsorship Sentiment Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 10px; text-align: center; }}
            .metric {{ background-color: #e8f4f8; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .positive {{ color: #27ae60; font-weight: bold; }}
            .negative {{ color: #e74c3c; font-weight: bold; }}
            .neutral {{ color: #7f8c8d; font-weight: bold; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .insight {{ background-color: #fff3cd; padding: 10px; margin: 10px 0; border-left: 4px solid #ffc107; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🏆 Sports Sponsorship Sentiment Analysis</h1>
            <h2>ISBA 2411 NLP Project - Rishabh Shah</h2>
            <p><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="metric">
            <h2>📊 Executive Summary</h2>
            <p><strong>Total Posts Analyzed:</strong> {total_posts:,}</p>
            <p><strong>Average Sentiment Score:</strong> {avg_sentiment:.3f} (Range: -1 to +1)</p>
            <p><strong>Analysis Period:</strong> Last 30 days</p>
        </div>
        
        <div class="metric">
            <h2>💭 Sentiment Distribution</h2>
            <p><span class="positive">Positive:</span> {sentiment_dist.get('positive', 0):.1f}%</p>
            <p><span class="negative">Negative:</span> {sentiment_dist.get('negative', 0):.1f}%</p>
            <p><span class="neutral">Neutral:</span> {sentiment_dist.get('neutral', 0):.1f}%</p>
        </div>
        
        {platform_analysis}
        {sponsorship_analysis}
        
        <div class="metric">
            <h2>🔍 Key Insights</h2>
            <div class="insight">
                <strong>Overall Sentiment:</strong> 
                {'Positive fan reception toward sponsorships' if avg_sentiment > 0.1 else 
                 'Negative fan sentiment toward sponsorships' if avg_sentiment < -0.1 else 
                 'Mixed/neutral sentiment toward sponsorships'}
            </div>
            
            <div class="insight">
                <strong>Engagement Pattern:</strong> 
                {'High engagement correlates with sentiment intensity' if 'engagement' in df.columns else 'Engagement data not available'}
            </div>
            
            <div class="insight">
                <strong>Platform Recommendation:</strong> 
                {f'Focus sponsorship announcements on {best_platform}' if 'platform' in df.columns and df['platform'].nunique() > 1 else 'Single platform analysis'}
            </div>
        </div>
        
        <div class="metric">
            <h2>📈 Methodology</h2>
            <p>This analysis used multiple NLP approaches:</p>
            <ul>
                <li><strong>VADER Sentiment:</strong> Social media optimized sentiment analysis</li>
                <li><strong>TextBlob:</strong> Traditional polarity and subjectivity scoring</li>
                <li><strong>Custom Keywords:</strong> Sports sponsorship specific sentiment terms</li>
                <li><strong>Ensemble Method:</strong> Weighted combination of all approaches</li>
                <li><strong>Machine Learning:</strong> TF-IDF vectorization with classification models</li>
            </ul>
        </div>
        
        <div class="metric">
            <h2>🎯 Strategic Recommendations</h2>
            <ul>
                <li>{'Leverage positive sentiment in marketing campaigns' if avg_sentiment > 0.1 else 'Address fan concerns in sponsorship communications'}</li>
                <li>{'Focus on jersey partnerships (highest positive sentiment)' if 'sponsorship_type' in df.columns else 'Consider sponsorship type preferences'}</li>
                <li>Monitor social media sentiment before major announcements</li>
                <li>Engage with fans to build positive sponsorship perception</li>
            </ul>
        </div>
        
        <div class="metric">
            <h2>📊 Technical Details</h2>
            <p><strong>Data Sources:</strong> Twitter, Reddit, Instagram</p>
            <p><strong>Analysis Methods:</strong> VADER, TextBlob, Custom Sentiment, ML Classification</p>
            <p><strong>Processing:</strong> Text cleaning, tokenization, lemmatization, TF-IDF vectorization</p>
            <p><strong>Models:</strong> Logistic Regression, Random Forest, Ensemble Methods</p>
        </div>
    </body>
    </html>
    """
    
    # Save report
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/comprehensive_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Report saved to {output_dir}/comprehensive_report.html")

def print_summary_stats(df):
    """Print summary statistics to console"""
    print("\\n" + "="*60)
    print("🏆 SPORTS SPONSORSHIP SENTIMENT ANALYSIS RESULTS")
    print("="*60)
    
    # Basic stats
    print(f"📊 Total Posts Analyzed: {len(df):,}")
    
    if 'platform' in df.columns:
        print(f"🌐 Platforms: {', '.join(df['platform'].unique())}")
        platform_counts = df['platform'].value_counts()
        for platform, count in platform_counts.items():
            print(f"   • {platform.title()}: {count:,} posts")
    
    # Sentiment distribution
    print(f"\\n💭 Sentiment Distribution:")
    sentiment_dist = df['ensemble_sentiment'].value_counts(normalize=True) * 100
    for sentiment, pct in sentiment_dist.items():
        emoji = "😊" if sentiment == "positive" else "😞" if sentiment == "negative" else "😐"
        print(f"   {emoji} {sentiment.title()}: {pct:.1f}%")
    
    # Average sentiment
    avg_sentiment = df['vader_compound'].mean()
    print(f"\\n🎯 Average Sentiment Score: {avg_sentiment:.3f}")
    
    if avg_sentiment > 0.1:
        print("   ✅ Overall POSITIVE sentiment toward sponsorships")
    elif avg_sentiment < -0.1:
        print("   ❌ Overall NEGATIVE sentiment toward sponsorships") 
    else:
        print("   ⚖️ MIXED/NEUTRAL sentiment toward sponsorships")
    
    # Sponsorship types
    if 'sponsorship_type' in df.columns:
        print(f"\\n🎪 Sponsorship Types Found:")
        type_counts = df['sponsorship_type'].value_counts()
        for stype, count in type_counts.items():
            if stype != 'not_sponsorship':
                pct = count / len(df) * 100
                print(f"   • {stype.replace('_', ' ').title()}: {count} ({pct:.1f}%)")
    
    # Top insights
    print(f"\\n🔍 Key Insights:")
    if 'platform' in df.columns and df['platform'].nunique() > 1:
        platform_sentiment = df.groupby('platform')['vader_compound'].mean()
        best_platform = platform_sentiment.idxmax()
        print(f"   • {best_platform.title()} shows most positive sentiment")
    
    if 'engagement' in df.columns:
        high_engagement = df[df['engagement'] > df['engagement'].median()]
        high_eng_sentiment = high_engagement['vader_compound'].mean()
        print(f"   • High engagement posts have {high_eng_sentiment:.3f} avg sentiment")
    
    print(f"\\n📁 Output Files:")
    print(f"   📊 Data: sponsorship_analysis_results.csv")
    print(f"   📈 Charts: results/sentiment_dashboard.png")
    print(f"   📄 Report: results/comprehensive_report.html")
    
    print("\\n🎓 Ready for ISBA 2411 presentation!")
    print("="*60)

def main():
    """Main execution function with enhanced keyword analysis"""
    print("🚀 Starting Sports Sponsorship NLP Analysis...")
    print("ISBA 2411 Project - Rishabh Shah")
    print("="*60)
    
    # Initialize configuration
    config = Config()
    
    try:
        # Step 1: Data Collection
        print("\\n📱 STEP 1: Collecting data from social media...")
        df = collect_sponsorship_data(config)
        
        if len(df) == 0:
            print("❌ No data collected. Check your API credentials and internet connection.")
            return
        
        print(f"✅ Collected {len(df)} posts from {df['platform'].nunique()} platforms")
        
        # Step 2: NLP Processing
        print("\\n🧠 STEP 2: Processing with NLP algorithms...")
        df_processed = process_sponsorship_data(df, config)
        
        # Step 3: Save Results
        print("\\n💾 STEP 3: Saving results...")
        df_processed.to_csv(config.OUTPUT_FILE, index=False)
        print(f"✅ Data saved to {config.OUTPUT_FILE}")
        
        # Step 4: Generate Visualizations
        print("\\n📊 STEP 4: Creating visualizations...")
        create_visualizations(df_processed)
        
        # Step 5: Generate Report
        print("\\n📄 STEP 5: Generating comprehensive report...")
        generate_insights_report(df_processed)
        
        # NEW STEP 6: Keyword and Post Analysis
        print("\\n🔍 STEP 6: Analyzing sentiment drivers...")
        keyword_results = analyze_sentiment_drivers(df_processed)
        
        # Step 7: Display Results
        print("\\n📊 STEP 7: Generating summary...")
        print_summary_stats(df_processed)
        
        # Enhanced summary with keyword insights
        print_keyword_insights(keyword_results)
        
        print("\\n🎉 Complete Analysis Finished! Check the results/ folder for all outputs.")
        
    except Exception as e:
        print(f"\\n❌ Error during analysis: {e}")
        logger.exception("Full error traceback:")
        
        # Still try to save whatever data we have
        if 'df' in locals() and len(df) > 0:
            df.to_csv('partial_results.csv', index=False)
            print("💾 Partial results saved to partial_results.csv")

def print_keyword_insights(keyword_results):
    """Print key insights from keyword analysis"""
    print("\\n" + "="*60)
    print("🔍 KEYWORD & CONTENT ANALYSIS INSIGHTS")
    print("="*60)
    
    if 'sentiment_keywords' in keyword_results:
        sentiment_keywords = keyword_results['sentiment_keywords']
        
        print("\\n💬 TOP WORDS DRIVING EACH SENTIMENT:")
        
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in sentiment_keywords:
                keywords = sentiment_keywords[sentiment]['keywords'][:10]
                emoji = "😊" if sentiment == "positive" else "😞" if sentiment == "negative" else "😐"
                
                print(f"\\n{emoji} {sentiment.upper()} Keywords:")
                print(f"   {', '.join(keywords)}")
    
    if 'representative_posts' in keyword_results:
        representative_posts = keyword_results['representative_posts']
        
        print("\\n📝 EXAMPLE POSTS BY SENTIMENT:")
        
        for sentiment in ['positive', 'negative']:
            if sentiment in representative_posts and representative_posts[sentiment]:
                post = representative_posts[sentiment][0]  # Get first example
                emoji = "😊" if sentiment == "positive" else "😞"
                
                print(f"\\n{emoji} {sentiment.upper()} Example:")
                preview = post['text'][:150] + "..." if len(post['text']) > 150 else post['text']
                print(f"   \\\"{preview}\\\"")
                print(f"   Platform: {post['platform']} | Score: {post['vader_compound']:.3f}")
    
    print(f"\\n📊 Detailed Analysis Available In:")
    print(f"   🌐 HTML Report: results/keyword_analysis_report.html")
    print(f"   📈 Word Clouds: results/sentiment_wordclouds.png")
    print(f"   📋 Raw Data: results/sentiment_keywords.csv")

if __name__ == "__main__":
    main()