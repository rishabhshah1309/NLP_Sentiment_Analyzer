# From your previous Reddit setup
from reddit_conn import RedditSportsDataCollector

collector = RedditSportsDataCollector()
data = collector.collect_comprehensive_data(
    subreddits=['nba', 'nfl'],  # Start with 2 popular subreddits
    search_terms=['jersey sponsor', 'stadium naming'],  # Focus on 2 key terms
    posts_per_subreddit=20,
    comments_per_post=10
)
df = collector.save_data(data)

from distilgpt2_sponsorship_analysis import run_distilgpt2_analysis

# Run the analysis (will use CPU)
results, model = run_distilgpt2_analysis(
    data_file='reddit_sports_sponsorship_data.csv',
    train_new_model=True
)

from sponsorship_sentiment_analysis import run_sponsorship_sentiment_analysis

# Run comprehensive sentiment analysis
results, analyzer, models = run_sponsorship_sentiment_analysis('distilgpt2_sponsorship_analysis.csv')
