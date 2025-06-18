"""
Enhanced Sports Sponsorship Data Collector
ISBA 2411 - Rishabh Shah

Focuses specifically on sponsorship and partnership content with improved search strategies
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import re
from typing import List, Dict, Tuple
import logging
import random
import json

# Social media libraries
try:
    import tweepy
    import praw
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
except ImportError as e:
    print(f"Some libraries missing: {e}")
    print("Install with: pip install tweepy praw selenium webdriver-manager")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSponsorshipDataCollector:
    """Enhanced collector focused specifically on sponsorship and partnership content"""
    
    def __init__(self, config):
        self.config = config
        self.all_data = []
        
        # Enhanced sponsorship-specific search terms
        self.sponsorship_search_terms = [
            # Direct sponsorship terms
            "jersey sponsorship deal",
            "stadium naming rights", 
            "partnership announcement",
            "official sponsor",
            "title sponsor",
            "kit supplier deal",
            "brand partnership",
            "sponsorship agreement",
            
            # Brand + sport combinations
            "Nike NBA deal",
            "Adidas football partnership", 
            "Emirates stadium sponsorship",
            "Coca Cola sports sponsor",
            "Pepsi league partner",
            "Microsoft sports partnership",
            "Amazon sports deal",
            
            # Event/league sponsorships
            "Premier League sponsor",
            "Champions League partner", 
            "Super Bowl sponsor",
            "Olympics partnership",
            "World Cup sponsor",
            "NBA finals sponsor",
            
            # Reaction terms
            "sponsorship reaction",
            "fans react sponsor",
            "new sponsor announcement",
            "sponsor controversy", 
            "sponsorship criticism",
            "love new sponsor",
            "hate sponsor deal"
        ]
        
        # Major sports brands to track
        self.major_sponsors = [
            "Nike", "Adidas", "Puma", "Under Armour", "New Balance",
            "Emirates", "Etihad", "Qatar Airways", "Turkish Airlines",
            "Coca Cola", "Pepsi", "Red Bull", "Monster Energy",
            "Microsoft", "Amazon", "Google", "Apple", "Samsung",
            "Visa", "Mastercard", "American Express", "PayPal",
            "McDonald's", "KFC", "Subway", "Pizza Hut",
            "BMW", "Mercedes", "Audi", "Ford", "Toyota"
        ]
        
        # Sports entities to monitor
        self.sports_entities = [
            # Football/Soccer
            "Manchester United", "Real Madrid", "Barcelona", "Chelsea", "Arsenal",
            "Liverpool", "Manchester City", "Bayern Munich", "Juventus", "PSG",
            
            # American Sports
            "Lakers", "Warriors", "Celtics", "Knicks", "Heat", "Bulls",
            "Cowboys", "Patriots", "Packers", "49ers", "Giants", "Eagles",
            "Yankees", "Red Sox", "Dodgers", "Mets", "Cubs", "Astros",
            
            # Leagues
            "Premier League", "Champions League", "NBA", "NFL", "MLB", "NHL", "MLS"
        ]

    def get_enhanced_twitter_data(self):
        """Enhanced Twitter data collection with focused sponsorship queries"""
        logger.info("🐦 Enhanced Twitter data collection...")
        twitter_data = []
        
        if not self.config.TWITTER_BEARER_TOKEN:
            logger.warning("No Twitter API token - generating enhanced demo data")
            return self._generate_enhanced_twitter_demo()
        
        try:
            # Initialize Twitter API v2
            client = tweepy.Client(bearer_token=self.config.TWITTER_BEARER_TOKEN)
            
            # Test API connection
            me = client.get_me()
            logger.info(f"✅ Twitter API connected as: {me.data.username}")
            
            tweets_collected = 0
            max_per_query = min(100, self.config.MAX_TWEETS // len(self.sponsorship_search_terms))
            
            for search_term in self.sponsorship_search_terms[:10]:  # Limit to avoid rate limits
                logger.info(f"   Searching: '{search_term}'")
                
                try:
                    # Enhanced search query with filters
                    query = f'"{search_term}" -is:retweet lang:en'
                    
                    tweets = tweepy.Paginator(
                        client.search_recent_tweets,
                        query=query,
                        tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations'],
                        max_results=min(100, max_per_query)
                    ).flatten(limit=max_per_query)
                    
                    for tweet in tweets:
                        if self._is_sponsorship_relevant(tweet.text):
                            twitter_data.append({
                                'platform': 'twitter',
                                'id': tweet.id,
                                'text': tweet.text,
                                'username': f'user_{tweet.author_id}',
                                'created_date': tweet.created_at,
                                'likes': tweet.public_metrics['like_count'],
                                'retweets': tweet.public_metrics['retweet_count'],
                                'replies': tweet.public_metrics['reply_count'],
                                'engagement': (
                                    tweet.public_metrics['like_count'] + 
                                    tweet.public_metrics['retweet_count'] + 
                                    tweet.public_metrics['reply_count']
                                ),
                                'search_term': search_term,
                                'context_annotations': tweet.context_annotations
                            })
                            tweets_collected += 1
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    logger.warning(f"Error searching '{search_term}': {e}")
                    continue
            
            logger.info(f"✅ Collected {tweets_collected} relevant Twitter posts")
            
        except Exception as e:
            logger.error(f"Twitter API error: {e}")
            logger.info("Generating enhanced demo data...")
            return self._generate_enhanced_twitter_demo()
        
        return twitter_data if twitter_data else self._generate_enhanced_twitter_demo()

    def get_enhanced_reddit_data(self):
        """Enhanced Reddit data with sponsorship-focused subreddits and queries"""
        logger.info("📱 Enhanced Reddit data collection...")
        reddit_data = []
        
        # Sponsorship-focused subreddits
        sponsorship_subreddits = [
            'soccer', 'football', 'nba', 'nfl', 'baseball', 'hockey',
            'sports', 'SportsBusiness', 'MLS', 'PremierLeague', 
            'chelseafc', 'reddevils', 'Gunners', 'LiverpoolFC',
            'lakers', 'warriors', 'bostonceltics', 'chicagobulls'
        ]
        
        if not self.config.REDDIT_CLIENT_ID:
            logger.warning("No Reddit credentials - generating enhanced demo data")
            return self._generate_enhanced_reddit_demo()
        
        try:
            reddit = praw.Reddit(
                client_id=self.config.REDDIT_CLIENT_ID,
                client_secret=self.config.REDDIT_CLIENT_SECRET,
                user_agent=self.config.REDDIT_USER_AGENT
            )
            
            for subreddit_name in sponsorship_subreddits[:8]:  # Limit to avoid blocks
                logger.info(f"   Searching r/{subreddit_name}")
                
                try:
                    subreddit = reddit.subreddit(subreddit_name)
                    
                    # Search for sponsorship-specific terms
                    sponsorship_terms = [
                        "sponsor", "sponsorship", "partnership", "deal", "contract",
                        "Nike", "Adidas", "Emirates", "jersey", "kit", "naming rights"
                    ]
                    
                    for term in sponsorship_terms[:3]:  # Limit terms per subreddit
                        try:
                            for submission in subreddit.search(term, time_filter='month', limit=8):
                                if self._is_sponsorship_relevant(f"{submission.title} {submission.selftext}"):
                                    reddit_data.append({
                                        'platform': 'reddit',
                                        'id': submission.id,
                                        'text': f"{submission.title}. {submission.selftext}".strip(),
                                        'username': f"reddit_user_{submission.id[:8]}",
                                        'created_date': datetime.fromtimestamp(submission.created_utc),
                                        'likes': submission.score,
                                        'retweets': 0,
                                        'replies': submission.num_comments,
                                        'engagement': submission.score + submission.num_comments,
                                        'search_term': term,
                                        'subreddit': subreddit_name,
                                        'url': f"https://reddit.com{submission.permalink}"
                                    })
                            
                            time.sleep(2)  # Rate limiting
                        except Exception as e:
                            logger.warning(f"Error searching r/{subreddit_name} for {term}: {e}")
                            continue
                    
                except Exception as e:
                    logger.warning(f"Error accessing r/{subreddit_name}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Reddit collection error: {e}")
            return self._generate_enhanced_reddit_demo()
        
        logger.info(f"✅ Collected {len(reddit_data)} Reddit posts")
        return reddit_data if reddit_data else self._generate_enhanced_reddit_demo()

    def get_enhanced_selenium_data(self):
        """Enhanced Selenium scraping from multiple sports sites"""
        logger.info("🕷️ Enhanced Selenium multi-site scraping...")
        
        try:
            # Import the enhanced scraper from the file you just created
            from selenium_scraper import EnhancedSponsorshipScraper
            scraper = EnhancedSponsorshipScraper(self.config)
            return scraper.scrape_all_sites_enhanced()
        except ImportError:
            logger.warning("Enhanced Selenium scraper not available - using original selenium scraper")
            try:
                # Fall back to original selenium scraper
                from selenium_scraper import MultiSiteSponsorshipScraper
                scraper = MultiSiteSponsorshipScraper(self.config)
                return scraper.scrape_all_sites()
            except ImportError:
                logger.warning("Original selenium scraper also not available - using demo data")
                return self._generate_selenium_demo_data()
        except Exception as e:
            logger.error(f"Selenium scraping failed: {e}")
            return self._generate_selenium_demo_data()

    def _generate_selenium_demo_data(self):
        """Generate demo data simulating enhanced selenium scraping"""
        logger.info("Generating enhanced selenium demo data...")
        
        enhanced_headlines = [
            {
                'text': "Nike Extends Manchester United Partnership in Record-Breaking $900M Deal Through 2034",
                'platform': 'espn',
                'source': 'ESPN',
                'engagement_level': 'high'
            },
            {
                'text': "Fan Reaction Mixed as Chelsea Announces Infinite Athlete as New Front-of-Shirt Sponsor Worth £40M Annually",
                'platform': 'bleacher_report', 
                'source': 'Bleacher Report',
                'engagement_level': 'high'
            },
            {
                'text': "Real Madrid's Adidas Partnership Extension: €1.1 Billion Investment Signals Confidence in Global Football Growth and Brand Value",
                'platform': 'front_office_sports',
                'source': 'Front Office Sports',
                'engagement_level': 'medium'
            },
            {
                'text': "NBA Jersey Sponsorship Revenue Reaches Record $1.8 Billion as League Expands Partnership Program with New Guidelines",
                'platform': 'espn',
                'source': 'ESPN',
                'engagement_level': 'medium'
            },
            {
                'text': "Arsenal Emirates Stadium: 18 Years Later, Naming Rights Deal Still Divides Fanbase According to New Survey",
                'platform': 'bleacher_report',
                'source': 'Bleacher Report',
                'engagement_level': 'high'
            },
            {
                'text': "Lakers Crypto.com Arena Rebrand Analysis: $700M Deal Brings Modern Naming to Historic Venue, But Fan Sentiment Remains Mixed",
                'platform': 'front_office_sports',
                'source': 'Front Office Sports',
                'engagement_level': 'medium'
            },
            {
                'text': "Premier League Sponsorship Revenue Analysis: How Emirates, Etihad, and Other Partnerships Shape Modern Football Economics",
                'platform': 'front_office_sports',
                'source': 'Front Office Sports',
                'engagement_level': 'medium'
            },
            {
                'text': "Golden State Warriors Rakuten Partnership Success Story: How $60M Deal Created Win-Win Model for NBA Sponsorships",
                'platform': 'espn',
                'source': 'ESPN',
                'engagement_level': 'low'
            },
            {
                'text': "Tottenham's New Stadium Naming Rights: Corporate Sponsors Circle as Club Seeks Premium Partnership Deal",
                'platform': 'bleacher_report',
                'source': 'Bleacher Report',
                'engagement_level': 'medium'
            },
            {
                'text': "PSG's Nike Partnership vs Barcelona's Nike Deal: Comparing Two of Football's Biggest Sponsorship Agreements",
                'platform': 'front_office_sports',
                'source': 'Front Office Sports',
                'engagement_level': 'low'
            }
        ]
        
        selenium_data = []
        base_date = datetime.now() - timedelta(days=18)
        
        for i, headline in enumerate(enhanced_headlines):
            # Generate engagement based on level
            if headline['engagement_level'] == 'high':
                likes = random.randint(1000, 5000)
                retweets = random.randint(200, 1000)
                replies = random.randint(100, 500)
            elif headline['engagement_level'] == 'medium':
                likes = random.randint(300, 1500)
                retweets = random.randint(50, 300)
                replies = random.randint(30, 150)
            else:  # low
                likes = random.randint(100, 600)
                retweets = random.randint(20, 100)
                replies = random.randint(10, 60)
            
            selenium_data.append({
                'platform': headline['platform'],
                'id': f'enhanced_selenium_{i+1}',
                'text': headline['text'],
                'username': headline['source'],
                'created_date': base_date + timedelta(days=i*1.5, hours=random.randint(0, 23)),
                'likes': likes,
                'retweets': retweets,
                'replies': replies,
                'engagement': likes + retweets + replies,
                'search_term': 'enhanced_selenium_demo',
                'source': 'enhanced_selenium_demo'
            })
        
        return selenium_data

    def _is_sponsorship_relevant(self, text):
        """Enhanced relevance checking for sponsorship content"""
        if not text or len(text) < 20:
            return False
        
        text_lower = text.lower()
        
        # Must contain sponsorship indicators
        sponsorship_indicators = [
            'sponsor', 'sponsorship', 'partnership', 'deal', 'contract', 'agreement',
            'naming rights', 'official partner', 'title sponsor', 'jersey', 'kit', 
            'supplier', 'brand partner', 'exclusive partner', 'strategic partner'
        ]
        
        # Must contain sports context
        sports_context = [
            'team', 'league', 'club', 'stadium', 'arena', 'player', 'athlete',
            'football', 'soccer', 'basketball', 'baseball', 'hockey', 'tennis',
            'nfl', 'nba', 'mlb', 'nhl', 'mls', 'premier league', 'champions league'
        ]
        
        # Must contain brand/entity
        brand_context = any(brand.lower() in text_lower for brand in self.major_sponsors) or \
                       any(entity.lower() in text_lower for entity in self.sports_entities)
        
        has_sponsorship = any(indicator in text_lower for indicator in sponsorship_indicators)
        has_sports = any(context in text_lower for context in sports_context)
        
        # At least 2 of 3 criteria must be met
        criteria_met = sum([has_sponsorship, has_sports, brand_context])
        return criteria_met >= 2

    def _generate_enhanced_twitter_demo(self):
        """Generate more realistic Twitter demo data focused on sponsorships"""
        logger.info("Generating enhanced Twitter demo data...")
        
        demo_tweets = [
            {
                'text': "🔥 BREAKING: Manchester United announces massive new Nike partnership deal worth £900M over 10 years! This could change everything for the club's transfer strategy. Thoughts? #MUFC #Nike #Partnership",
                'engagement_level': 'high',
                'sentiment_hint': 'positive'
            },
            {
                'text': "Absolutely devastated by the Emirates Stadium naming rights deal. Arsenal's history and tradition sold for corporate money. The fans deserve better than this commercialization. #Arsenal #Emirates #Sellout",
                'engagement_level': 'high', 
                'sentiment_hint': 'negative'
            },
            {
                'text': "The new Lakers x Crypto.com Arena partnership makes sense business-wise but Lakers fans are split. Some love the revenue boost, others miss the old Staples Center name. #Lakers #CryptoArena",
                'engagement_level': 'medium',
                'sentiment_hint': 'neutral'
            },
            {
                'text': "Real Madrid's Adidas jersey deal extension is HUGE! €1.1 billion over 10 years shows the power of sports partnerships. Both brands benefit massively from this collaboration 🔥 #RealMadrid #Adidas",
                'engagement_level': 'high',
                'sentiment_hint': 'positive'
            },
            {
                'text': "These jersey sponsorship patches are getting ridiculous. NFL players looking like NASCAR drivers with all these corporate logos. When did sports become such a billboard? #NFL #Sponsorship",
                'engagement_level': 'medium',
                'sentiment_hint': 'negative'
            },
            {
                'text': "Love how @NBA handles sponsor integration compared to other leagues. The small jersey patches feel natural and don't overwhelm the design. Well done! #NBA #SponsorshipDoneRight",
                'engagement_level': 'medium',
                'sentiment_hint': 'positive'
            },
            {
                'text': "Chelsea's new front-of-shirt sponsor deal with Infinite Athlete brings £40M per year. Smart business by the club to secure financial stability in uncertain times. #Chelsea #Partnership",
                'engagement_level': 'low',
                'sentiment_hint': 'neutral'
            },
            {
                'text': "DISGUSTED with the Tottenham stadium naming rights sale to corporate sponsors. Another piece of football history erased for money. Fans don't matter anymore 😠 #Tottenham #CorporateGreed",
                'engagement_level': 'high',
                'sentiment_hint': 'negative'
            },
            {
                'text': "The Golden State Warriors x Rakuten partnership has been brilliant for both brands. International exposure for Rakuten, financial boost for Warriors. Win-win sponsorship model! #Warriors #Rakuten",
                'engagement_level': 'medium',
                'sentiment_hint': 'positive'
            },
            {
                'text': "PSG's Nike deal vs Real Madrid's Adidas deal - which partnership offers better value? Both clubs getting massive money but different brand positioning strategies #PSG #RealMadrid #Comparison",
                'engagement_level': 'medium',
                'sentiment_hint': 'neutral'
            }
        ]
        
        twitter_data = []
        base_date = datetime.now() - timedelta(days=21)
        
        for i, tweet_template in enumerate(demo_tweets):
            # Generate engagement based on level
            if tweet_template['engagement_level'] == 'high':
                likes = random.randint(500, 5000)
                retweets = random.randint(100, 1000)
                replies = random.randint(50, 500)
            elif tweet_template['engagement_level'] == 'medium':
                likes = random.randint(100, 800)
                retweets = random.randint(20, 200)
                replies = random.randint(10, 100)
            else:  # low
                likes = random.randint(10, 150)
                retweets = random.randint(2, 50)
                replies = random.randint(1, 25)
            
            twitter_data.append({
                'platform': 'twitter',
                'id': f'enhanced_tw_{i+1}',
                'text': tweet_template['text'],
                'username': f'sports_fan_{i+1}',
                'created_date': base_date + timedelta(days=i*2, hours=random.randint(0, 23)),
                'likes': likes,
                'retweets': retweets,
                'replies': replies,
                'engagement': likes + retweets + replies,
                'search_term': 'enhanced_demo',
                'context_annotations': None
            })
        
        return twitter_data

    def _generate_enhanced_reddit_demo(self):
        """Generate enhanced Reddit demo focused on sponsorship discussions"""
        logger.info("Generating enhanced Reddit demo data...")
        
        demo_posts = [
            {
                'text': "Analysis: How the Emirates Stadium naming rights deal changed Arsenal's financial landscape. The £200M investment over 15 years allowed for better player acquisitions and stadium improvements, but many fans still debate whether losing the 'Arsenal Stadium' name was worth it. What's your take on trading tradition for financial stability?",
                'subreddit': 'Gunners',
                'engagement_level': 'high'
            },
            {
                'text': "Discussion: NBA jersey sponsors vs European football shirt sponsors - which approach works better? The NBA's small patches seem more tasteful than the massive front-of-shirt sponsors in Premier League. European fans seem more accepting though. Cultural differences in sports commercialization?",
                'subreddit': 'nba',
                'engagement_level': 'high'
            },
            {
                'text': "Manchester United's Nike deal breakdown: £900M over 10 years seems massive but is it actually good value? Comparing to Real Madrid's Adidas deal (€1.1B) and other top clubs. United's global fanbase vs on-field performance - which matters more to sponsors?",
                'subreddit': 'reddevils',
                'engagement_level': 'medium'
            },
            {
                'text': "Unpopular opinion: Stadium naming rights deals are necessary evil in modern sports. Yes, it hurts tradition and connection to history, but the revenue allows teams to compete financially with oil clubs and corporations. Sometimes business reality trumps sentiment.",
                'subreddit': 'soccer',
                'engagement_level': 'high'
            },
            {
                'text': "Data analysis: Fan sentiment vs sponsorship revenue correlation across major leagues. Interesting patterns showing that successful partnerships (like Warriors-Rakuten) actually improve fan satisfaction over time, while controversial ones (like some stadium naming deals) maintain negative sentiment for years.",
                'subreddit': 'sports',
                'engagement_level': 'medium'
            },
            {
                'text': "Lakers fans: How do you feel about the Crypto.com Arena name change? I know it brings in $700M over 20 years, but Staples Center had 22 years of history. Is the financial benefit worth losing that connection? Genuinely curious about different perspectives.",
                'subreddit': 'lakers',
                'engagement_level': 'medium'
            }
        ]
        
        reddit_data = []
        base_date = datetime.now() - timedelta(days=25)
        
        for i, post in enumerate(demo_posts):
            if post['engagement_level'] == 'high':
                likes = random.randint(200, 1000)
                replies = random.randint(50, 200)
            else:  # medium
                likes = random.randint(50, 300)
                replies = random.randint(15, 80)
            
            reddit_data.append({
                'platform': 'reddit',
                'id': f'enhanced_rd_{i+1}',
                'text': post['text'],
                'username': f'reddit_analyst_{i+1}',
                'created_date': base_date + timedelta(days=i*3, hours=random.randint(0, 23)),
                'likes': likes,
                'retweets': 0,
                'replies': replies,
                'engagement': likes + replies,
                'search_term': 'enhanced_demo',
                'subreddit': post['subreddit']
            })
        
        return reddit_data

    def _generate_selenium_demo_data(self):
        """Generate demo data simulating sports news sites"""
        logger.info("Generating Selenium demo data...")
        
        news_headlines = [
            {
                'text': "Nike and Manchester United Extend Partnership in Record-Breaking $900M Deal",
                'platform': 'espn',
                'source': 'ESPN'
            },
            {
                'text': "Analysis: Why Arsenal's Emirates Stadium Naming Rights Deal Still Divides Fans 15 Years Later",
                'platform': 'bleacher_report', 
                'source': 'Bleacher Report'
            },
            {
                'text': "Real Madrid's Adidas Partnership Extension: €1.1 Billion Investment in Football's Future",
                'platform': 'front_office_sports',
                'source': 'Front Office Sports'
            },
            {
                'text': "NBA Jersey Sponsorship Revenue Reaches Record $1.8 Billion as League Expands Partnership Program",
                'platform': 'espn',
                'source': 'ESPN'  
            },
            {
                'text': "Chelsea Secures £40M Annual Front-of-Shirt Sponsorship with Infinite Athlete in Multi-Year Deal",
                'platform': 'front_office_sports',
                'source': 'Front Office Sports'
            }
        ]
        
        selenium_data = []
        base_date = datetime.now() - timedelta(days=14)
        
        for i, headline in enumerate(news_headlines):
            selenium_data.append({
                'platform': headline['platform'],
                'id': f'selenium_{i+1}',
                'text': headline['text'],
                'username': headline['source'],
                'created_date': base_date + timedelta(days=i*2),
                'likes': random.randint(100, 2000),
                'retweets': random.randint(20, 400),
                'replies': random.randint(10, 200),
                'engagement': random.randint(200, 3000),
                'search_term': 'selenium_demo'
            })
        
        return selenium_data

    def collect_enhanced_sponsorship_data(self):
        """Main collection function with enhanced sponsorship focus"""
        logger.info("🚀 Starting Enhanced Sponsorship Data Collection")
        logger.info("="*60)
        
        all_data = []
        
        # Collect from each enhanced source
        try:
            twitter_data = self.get_enhanced_twitter_data()
            all_data.extend(twitter_data)
            logger.info(f"✅ Enhanced Twitter: {len(twitter_data)} posts")
        except Exception as e:
            logger.error(f"❌ Enhanced Twitter failed: {e}")
        
        try:
            reddit_data = self.get_enhanced_reddit_data()
            all_data.extend(reddit_data)
            logger.info(f"✅ Enhanced Reddit: {len(reddit_data)} posts")
        except Exception as e:
            logger.error(f"❌ Enhanced Reddit failed: {e}")
        
        try:
            selenium_data = self.get_enhanced_selenium_data()
            all_data.extend(selenium_data)
            logger.info(f"✅ Enhanced Selenium: {len(selenium_data)} posts")
        except Exception as e:
            logger.error(f"❌ Enhanced Selenium failed: {e}")
        
        if not all_data:
            logger.warning("⚠️ No data collected - using comprehensive demo dataset")
            all_data = self._generate_comprehensive_sponsorship_demo()
        
        # Convert to DataFrame and enhance
        df = pd.DataFrame(all_data)
        
        if len(df) > 0:
            # Standardize and clean
            df['created_date'] = pd.to_datetime(df['created_date'])
            df['text'] = df['text'].astype(str)
            df['text_length'] = df['text'].str.len()
            
            # Enhanced filtering for sponsorship content
            df = df[df['text_length'] >= 30]  # Longer minimum for quality
            df = df[df['text'].apply(self._is_sponsorship_relevant)]  # Strong relevance filter
            df = df.drop_duplicates(subset=['text'])
            
            # Add sponsorship metadata
            df = self._add_sponsorship_metadata(df)
            
            logger.info("="*60)
            logger.info(f"📊 ENHANCED COLLECTION SUMMARY")
            logger.info("="*60)
            logger.info(f"Total sponsorship-relevant posts: {len(df)}")
            
            platform_counts = df['platform'].value_counts()
            for platform, count in platform_counts.items():
                percentage = count / len(df) * 100
                logger.info(f"  {platform.replace('_', ' ').title()}: {count} posts ({percentage:.1f}%)")
            
            if 'sponsor_brand' in df.columns:
                brand_counts = df['sponsor_brand'].value_counts().head(5)
                logger.info(f"\\nTop mentioned brands:")
                for brand, count in brand_counts.items():
                    logger.info(f"  {brand}: {count} mentions")
            
            logger.info(f"\\nDate range: {df['created_date'].min().date()} to {df['created_date'].max().date()}")
            logger.info("="*60)
        
        return df

    def _add_sponsorship_metadata(self, df):
        """Add enhanced metadata about sponsorship content"""
        logger.info("Adding sponsorship metadata...")
        
        # Detect mentioned brands
        df['sponsor_brand'] = df['text'].apply(self._extract_sponsor_brand)
        
        # Detect sports entity
        df['sports_entity'] = df['text'].apply(self._extract_sports_entity)
        
        # Enhanced sponsorship type classification
        df['sponsorship_type'] = df['text'].apply(self._classify_sponsorship_type_enhanced)
        
        # Deal value extraction (if mentioned)
        df['deal_value_mentioned'] = df['text'].apply(self._extract_deal_value)
        
        return df

    def _extract_sponsor_brand(self, text):
        """Extract the main sponsor brand mentioned"""
        text_lower = text.lower()
        for brand in self.major_sponsors:
            if brand.lower() in text_lower:
                return brand
        return 'Other'

    def _extract_sports_entity(self, text):
        """Extract the sports team/league mentioned"""
        text_lower = text.lower()
        for entity in self.sports_entities:
            if entity.lower() in text_lower:
                return entity
        return 'General'

    def _classify_sponsorship_type_enhanced(self, text):
        """Enhanced sponsorship type classification"""
        text_lower = text.lower()
        
        # More detailed classification
        if any(term in text_lower for term in ['jersey', 'shirt', 'kit', 'uniform']):
            return 'jersey_kit_sponsorship'
        elif any(term in text_lower for term in ['stadium', 'arena', 'naming rights', 'venue']):
            return 'venue_naming_rights'
        elif any(term in text_lower for term in ['title sponsor', 'presenting sponsor', 'league sponsor']):
            return 'title_sponsorship'
        elif any(term in text_lower for term in ['official partner', 'strategic partner', 'exclusive partner']):
            return 'official_partnership'
        elif any(term in text_lower for term in ['sleeve', 'patch', 'logo placement']):
            return 'logo_placement_sponsorship'
        elif any(term in text_lower for term in ['supplier', 'equipment', 'apparel']):
            return 'equipment_supplier'
        else:
            return 'general_sponsorship'

    def _extract_deal_value(self, text):
        """Extract deal value if mentioned in text"""
        # Look for monetary amounts
        money_pattern = r'[£$€¥]\s*(\d+(?:\.\d+)?)\s*([BMK]?)(?:illion)?|(\d+(?:\.\d+)?)\s*([BMK]?)(?:illion)?\s*[£$€¥]'
        match = re.search(money_pattern, text, re.IGNORECASE)
        if match:
            return True
        return False

    def _generate_comprehensive_sponsorship_demo(self):
        """Generate comprehensive demo focused on sponsorships"""
        logger.info("Generating comprehensive sponsorship demo dataset...")
        
        # More diverse and realistic sponsorship content
        comprehensive_data = []
        
        # Twitter-style sponsorship posts
        twitter_sponsorship_posts = [
            "🚨 BREAKING: Manchester United confirms new Nike partnership worth £900M over 10 years! Biggest kit deal in football history. This changes everything for transfer budget #MUFC #Nike #Partnership",
            "Absolutely DISGUSTED with Arsenal selling stadium naming rights to Emirates. 60,000 capacity venue now called 'Emirates Stadium' instead of historic Arsenal Stadium. Tradition sold for money 💔 #Arsenal #Emirates #Heritage",
            "Real Madrid's Adidas deal extension: €1.1 billion over 10 years is MASSIVE! Shows the global power of Los Blancos brand. Both club and Adidas benefit from this partnership 🔥 #RealMadrid #Adidas #HalaMadrid",
            "These NBA jersey patches are getting out of hand. Lakers with Bibigo, Warriors with Rakuten, Celtics with Vistaprint. When did basketball become a walking advertisement? #NBA #JerseySponsors #Commercialization",
            "Love how @premierleague handles sponsorships compared to other leagues. Emirates, Etihad Airways partnerships feel natural and enhance the global reach. Smart business model! #PremierLeague #Sponsorship #GlobalFootball",
            "Chelsea's new front-of-shirt sponsor 'Infinite Athlete' paying £40M per year is smart business. Club needs revenue stability after recent ownership changes. Good deal for both parties #Chelsea #InfiniteAthlete #Partnership",
            "The Crypto.com Arena name change from Staples Center still hurts Lakers fans. $700M over 20 years but 22 years of history and memories gone. Was it worth it? #Lakers #CryptoArena #StaplesCenter #Tradition",
            "PSG's Nike partnership vs Real Madrid's Adidas deal - which offers better value? Both massive contracts but different brand positioning strategies. European football = big business #PSG #RealMadrid #Nike #Adidas"
        ]
        
        # Reddit-style discussion posts
        reddit_sponsorship_posts = [
            "Detailed Analysis: Emirates Stadium naming rights deal impact on Arsenal (2006-2024). The £100M over 15 years helped fund new training ground and player acquisitions, but fan surveys show 67% still prefer 'Arsenal Stadium' name. Financial necessity vs emotional connection - what matters more in modern football?",
            "NBA jersey sponsorship evolution: From no sponsors (pre-2017) to $1.8B annual revenue. Data shows fan acceptance improved from 34% to 58% over 5 years. Key factors: patch size, brand alignment, team success correlation. Which partnerships work best and why?",
            "Manchester United's Nike deal vs Real Madrid's Adidas partnership: Comprehensive financial breakdown. United: £90M/year, Madrid: €110M/year. But Madrid gets better performance bonuses and global marketing support. Who got the better deal long-term?",
            "Unpopular opinion: Stadium naming rights are essential for modern football economics. Arsenal (Emirates), Manchester City (Etihad), Bayern Munich (Allianz Arena) - all top clubs with naming partnerships. Tradition vs financial reality in 2024.",
            "Chelsea's sponsor history analysis: Samsung (2005-2015) £18M/year, Yokohama (2015-2020) £40M/year, Three (2020-2023) £40M/year, now Infinite Athlete £40M/year. How do these partnerships affect on-field performance and fan engagement?",
            "Golden State Warriors + Rakuten partnership case study: $60M over 3 years delivered massive ROI for both parties. Warriors get revenue, Rakuten gets NBA exposure in US market. Perfect example of win-win sports sponsorship model."
        ]
        
        # Sports news headlines from various sources
        news_headlines = [
            "Nike Extends Manchester United Partnership in Record $900M Deal Through 2034",
            "Fan Reaction Mixed as Chelsea Announces Infinite Athlete as New Front-of-Shirt Sponsor",
            "Real Madrid's Adidas Partnership: €1.1 Billion Extension Signals Confidence in Global Football Growth",
            "NBA Jersey Sponsorship Revenue Hits All-Time High as League Expands Partnership Programs",
            "Arsenal Emirates Stadium: 18 Years Later, Naming Rights Deal Still Divides Fanbase",
            "Lakers Crypto.com Arena Rebrand: $700M Deal Brings Modern Naming to Historic Venue",
            "Premier League Sponsorship Analysis: How Emirates, Etihad, and Other Partnerships Shape Modern Football"
        ]
        
        # Generate varied content with timestamps
        base_date = datetime.now() - timedelta(days=35)
        
        # Add Twitter content
        for i, post in enumerate(twitter_sponsorship_posts):
            comprehensive_data.append({
                'platform': 'twitter',
                'id': f'comp_tw_{i+1}',
                'text': post,
                'username': f'football_analyst_{i+1}',
                'created_date': base_date + timedelta(days=i*3, hours=random.randint(0, 23)),
                'likes': random.randint(200, 3000),
                'retweets': random.randint(50, 800),
                'replies': random.randint(20, 400),
                'engagement': random.randint(300, 4000),
                'search_term': 'comprehensive_demo'
            })
        
        # Add Reddit content
        for i, post in enumerate(reddit_sponsorship_posts):
            comprehensive_data.append({
                'platform': 'reddit',
                'id': f'comp_rd_{i+1}',
                'text': post,
                'username': f'sports_analyst_{i+1}',
                'created_date': base_date + timedelta(days=i*4, hours=random.randint(0, 23)),
                'likes': random.randint(100, 800),
                'retweets': 0,
                'replies': random.randint(30, 150),
                'engagement': random.randint(150, 1000),
                'search_term': 'comprehensive_demo',
                'subreddit': random.choice(['soccer', 'nba', 'reddevils', 'Gunners', 'sports'])
            })
        
        # Add news content
        for i, headline in enumerate(news_headlines):
            comprehensive_data.append({
                'platform': random.choice(['espn', 'bleacher_report', 'front_office_sports']),
                'id': f'comp_news_{i+1}',
                'text': headline,
                'username': random.choice(['ESPN', 'Bleacher Report', 'Front Office Sports']),
                'created_date': base_date + timedelta(days=i*2, hours=random.randint(0, 23)),
                'likes': random.randint(500, 2000),
                'retweets': random.randint(100, 500),
                'replies': random.randint(50, 300),
                'engagement': random.randint(700, 3000),
                'search_term': 'comprehensive_demo'
            })
        
        logger.info(f"Generated {len(comprehensive_data)} comprehensive sponsorship posts")
        return comprehensive_data


def collect_enhanced_sponsorship_data(config):
    """Main function to collect enhanced sponsorship data"""
    collector = EnhancedSponsorshipDataCollector(config)
    return collector.collect_enhanced_sponsorship_data()


# Usage instructions
print("🚀 ENHANCED SPONSORSHIP DATA COLLECTOR READY!")
print("="*60)
print("✨ Enhanced Features:")
print("  🎯 Focused sponsorship search terms")
print("  🔍 Advanced relevance filtering")
print("  🏷️ Sponsor brand detection")
print("  📊 Sponsorship type classification")
print("  💰 Deal value extraction")
print("  🌐 Multi-platform integration")
print("\\n📈 Quality over quantity approach")
print("💡 Use: collector.collect_enhanced_sponsorship_data()")
print("="*60)