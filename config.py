import os

class Config:
    """Configuration settings for the project"""
    
    TWITTER_BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAANLk2AEAAAAAhRDRoXgL%2FXBwxt2%2FX3MobRVRvpQ%3DTFBXtNgQXpZQp131d5Stv50Dbm59p3JFhHz2x9utxbeIGF5Nyn"
    
    # Reddit API (optional)
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', 'uj98xZpKCRXiernnRu-DDQ')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', 'KibZ1g2Da0XpfDCHtiPJDB9_XvAx4w')
    REDDIT_USER_AGENT = 'script:SportsSentimentAnalysis:1.0 (by /u/hermantacobear)'
    
    # Data collection limits
    MAX_TWEETS = 800  # You can go much higher!
    MAX_INSTAGRAM_POSTS = 100
    MAX_REDDIT_POSTS = 150
    
    # Analysis settings
    USE_DISTILGPT2 = True
    MIN_TEXT_LENGTH = 10
    CONFIDENCE_THRESHOLD = 0.6
    
    # Search terms
    SPONSORSHIP_KEYWORDS = [
        'jersey sponsor', 'stadium naming', 'naming rights', 'official partner',
        'brand partnership', 'title sponsor', 'kit supplier', 'jersey deal',
        'sponsorship agreement', 'partnership announcement', 'Nike deal',
        'Adidas partnership', 'Emirates sponsor'
    ]
    
    # Target subreddits
    REDDIT_SUBREDDITS = [
        'soccer', 'nba', 'nfl', 'baseball', 'hockey', 'MLS', 
        'PremierLeague', 'ChampionsLeague', 'sports'
    ]
    
    # Sports accounts to monitor
    SPORTS_ACCOUNTS = [
        'ManUtd', 'Arsenal', 'RealMadrid', 'Lakers', 'Warriors',
        'premierleague', 'NBA', 'NFL', 'ESPN'
    ]
    
    # Output settings
    OUTPUT_FILE = 'sponsorship_analysis_results.csv'
    REPORT_FILE = 'sponsorship_report.html'