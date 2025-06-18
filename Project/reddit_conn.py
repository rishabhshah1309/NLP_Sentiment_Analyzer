# Reddit API Setup and Sports Sponsorship Data Collection
# Step-by-step implementation for your NLP project

import praw
import pandas as pd
import time
from datetime import datetime
import json
import os

# First, install required packages:
# pip install praw pandas python-dotenv

class RedditSportsDataCollector:
    def __init__(self, credentials_file='.env'):
        """
        Initialize Reddit API connection
        """
        # Load credentials from environment file
        if os.path.exists(credentials_file):
            from dotenv import load_dotenv
            load_dotenv(credentials_file)
            
            self.reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent=os.getenv('REDDIT_USER_AGENT', 'SportsSponsorship/1.0 by YourUsername')
            )
        else:
            print("Please set up your credentials first using setup_credentials()")
            self.reddit = None
    
    def setup_credentials(self):
        """
        Helper function to create credentials file
        """
        print("Reddit API Setup Instructions:")
        print("1. Go to https://www.reddit.com/prefs/apps")
        print("2. Click 'Create App' or 'Create Another App'")
        print("3. Fill out the form:")
        print("   - Name: Your project name (e.g., 'Sports Sentiment Analysis')")
        print("   - App type: Select 'script'")
        print("   - Description: Optional")
        print("   - About URL: Leave blank")
        print("   - Redirect URI: http://localhost:8080")
        print("4. Click 'Create app'")
        print("5. Note down your credentials:")
        print("   - Client ID: The string under the app name")
        print("   - Client Secret: The 'secret' field")
        
        client_id = input("\nEnter your Reddit Client ID: ")
        client_secret = input("Enter your Reddit Client Secret: ")
        user_agent = input("Enter your User Agent (e.g., 'SportsAnalysis/1.0 by YourUsername'): ")
        
        # Create .env file
        with open('.env', 'w') as f:
            f.write(f"REDDIT_CLIENT_ID={client_id}\n")
            f.write(f"REDDIT_CLIENT_SECRET={client_secret}\n")
            f.write(f"REDDIT_USER_AGENT={user_agent}\n")
        
        print("\nCredentials saved to .env file!")
        return client_id, client_secret, user_agent
    
    def test_connection(self):
        """
        Test if Reddit API connection works
        """
        try:
            print(f"Connected as: {self.reddit.user.me()}")
            print("Connection successful!")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            print("You're using read-only mode (which is fine for data collection)")
            return True
    
    def search_sponsorship_posts(self, subreddit_name, search_terms, limit=100, time_filter='month'):
        """
        Search for sponsorship-related posts in a subreddit
        """
        posts_data = []
        subreddit = self.reddit.subreddit(subreddit_name)
        
        print(f"Searching r/{subreddit_name} for sponsorship content...")
        
        for term in search_terms:
            print(f"  Searching for: '{term}'")
            try:
                # Search posts
                for submission in subreddit.search(term, sort='relevance', time_filter=time_filter, limit=limit):
                    post_data = {
                        'post_id': submission.id,
                        'subreddit': subreddit_name,
                        'title': submission.title,
                        'text': submission.selftext,
                        'score': submission.score,
                        'upvote_ratio': submission.upvote_ratio,
                        'num_comments': submission.num_comments,
                        'created_utc': submission.created_utc,
                        'created_date': datetime.fromtimestamp(submission.created_utc),
                        'url': submission.url,
                        'search_term': term,
                        'type': 'post'
                    }
                    posts_data.append(post_data)
                    
                    # Add small delay to be respectful to Reddit's servers
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Error searching for '{term}': {e}")
        
        print(f"Found {len(posts_data)} posts")
        return posts_data
    
    def get_post_comments(self, submission_id, max_comments=50):
        """
        Get comments from a specific post
        """
        comments_data = []
        
        try:
            submission = self.reddit.submission(id=submission_id)
            submission.comments.replace_more(limit=0)  # Remove "more comments" objects
            
            for comment in submission.comments.list()[:max_comments]:
                if hasattr(comment, 'body') and comment.body != '[deleted]':
                    comment_data = {
                        'comment_id': comment.id,
                        'post_id': submission_id,
                        'text': comment.body,
                        'score': comment.score,
                        'created_utc': comment.created_utc,
                        'created_date': datetime.fromtimestamp(comment.created_utc),
                        'type': 'comment'
                    }
                    comments_data.append(comment_data)
        
        except Exception as e:
            print(f"Error getting comments for post {submission_id}: {e}")
        
        return comments_data
    
    def collect_comprehensive_data(self, subreddits, search_terms, posts_per_subreddit=50, comments_per_post=20):
        """
        Comprehensive data collection across multiple subreddits
        """
        all_data = []
        
        for subreddit in subreddits:
            print(f"\n=== Processing r/{subreddit} ===")
            
            # Get posts
            posts = self.search_sponsorship_posts(subreddit, search_terms, limit=posts_per_subreddit)
            all_data.extend(posts)
            
            # Get comments for each post
            print(f"Getting comments for {len(posts)} posts...")
            for i, post in enumerate(posts[:10]):  # Limit to first 10 posts to avoid rate limits
                print(f"  Getting comments for post {i+1}/10")
                comments = self.get_post_comments(post['post_id'], max_comments=comments_per_post)
                
                # Add subreddit info to comments
                for comment in comments:
                    comment['subreddit'] = subreddit
                    comment['post_title'] = post['title']
                    comment['search_term'] = post['search_term']
                
                all_data.extend(comments)
                time.sleep(0.5)  # Be respectful to Reddit's servers
        
        return all_data
    
    def save_data(self, data, filename='reddit_sports_sponsorship_data.csv'):
        """
        Save collected data to CSV
        """
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
        
        # Print summary statistics
        print(f"\nData Summary:")
        print(f"Total entries: {len(df)}")
        print(f"Posts: {len(df[df['type'] == 'post'])}")
        print(f"Comments: {len(df[df['type'] == 'comment'])}")
        print(f"Subreddits: {df['subreddit'].nunique()}")
        print(f"Date range: {df['created_date'].min()} to {df['created_date'].max()}")
        
        return df

# Example usage and main execution
def main():
    # Initialize collector
    collector = RedditSportsDataCollector()
    
    # If credentials not set up, run setup
    if collector.reddit is None:
        collector.setup_credentials()
        collector = RedditSportsDataCollector()  # Reinitialize
    
    # Test connection
    collector.test_connection()
    
    # Define your search parameters
    
    # In your Reddit data collection
    sponsorship_terms = [
        'jersey sponsor deal',
        'Emirates Arsenal',
        'Nike Manchester United', 
        'stadium naming rights',
        'Allianz Arena',
        'Mercedes Benz Stadium',
        'official partner announcement',
        'sponsorship agreement',
        'kit supplier'
    ]

    sports_subreddits = ['soccer', 'MLS', 'PremierLeague', 'nba', 'nfl', 'sports']
        
    print("Starting data collection...")
    print(f"Subreddits: {sports_subreddits}")
    print(f"Search terms: {sponsorship_terms}")
    
    # Collect data
    data = collector.collect_comprehensive_data(
        subreddits=sports_subreddits,
        search_terms=sponsorship_terms,
        posts_per_subreddit=25,  # Start small for testing
        comments_per_post=15
    )
    
    # Save data
    df = collector.save_data(data)
    
    # Show sample data
    print("\nSample data:")
    print(df[['subreddit', 'type', 'text', 'score']].head(10))
    
    return df

# Additional utility functions
def analyze_collected_data(df):
    """
    Quick analysis of collected data
    """
    print("=== Data Analysis ===")
    
    # Engagement metrics
    print(f"Average post score: {df[df['type'] == 'post']['score'].mean():.2f}")
    print(f"Average comment score: {df[df['type'] == 'comment']['score'].mean():.2f}")
    
    # Most active subreddits
    print(f"\nMost active subreddits:")
    print(df['subreddit'].value_counts().head())
    
    # Most common search terms
    print(f"\nMost successful search terms:")
    print(df['search_term'].value_counts().head())
    
    # Text length distribution
    df['text_length'] = df['text'].str.len()
    print(f"\nAverage text length: {df['text_length'].mean():.0f} characters")

if __name__ == "__main__":
    # Run the main data collection
    collected_df = main()
    
    # Analyze the results
    analyze_collected_data(collected_df)