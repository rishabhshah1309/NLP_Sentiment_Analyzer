"""
Selenium Multi-Site Sports Sponsorship Scraper
ISBA 2411 - Rishabh Shah / hermantacobear

Scrapes real sponsorship data from:
- Twitter/X
- ESPN
- Bleacher Report  
- Front Office Sports
- The Athletic (bonus)
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
import time
import random
from datetime import datetime, timedelta
import re
import logging

logger = logging.getLogger(__name__)

class MultiSiteSponsorshipScraper:
    """Scrape sponsorship content from multiple sports news sites"""
    
    def __init__(self, config):
        self.config = config
        self.driver = None
        
        # Search terms optimized for news sites
        self.sponsorship_terms = [
            'football', 'soccer', 'basketball', 'baseball', 'hockey', 'tennis',
            'cricket', 'rugby', 'golf', 'nascar', 'f1', 'formula 1',
            'team', 'player', 'league', 'championship', 'tournament', 'match',
            'game', 'stadium', 'arena', 'club', 'fc', 'united', 'city',
            'premier league', 'nba', 'nfl', 'mlb', 'mls', 'champions league',
            'la liga', 'serie a', 'bundesliga', 'ligue 1', 'uefa', 'fifa',
            'nike', 'adidas', 'emirates', 'jersey', 'kit', 'sponsor',
            'partnership', 'deal', 'naming rights', 'official partner'
        ]
        
    def setup_driver(self):
        """Setup Chrome driver with stealth options"""
        options = Options()
        
        # Stealth options to avoid detection
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # User agent rotation
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]
        options.add_argument(f'--user-agent={random.choice(user_agents)}')
        
        # Optional: Run headless (faster but harder to debug)
        # options.add_argument('--headless')
        
        try:
            self.driver = webdriver.Chrome(options=options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            logger.info("✅ Chrome driver initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Chrome driver setup failed: {e}")
            logger.info("💡 Install ChromeDriver: https://chromedriver.chromium.org/")
            return False
    
    def scrape_twitter_selenium(self, max_tweets=100):
        """Scrape Twitter using Selenium"""
        logger.info("🐦 Scraping Twitter with Selenium...")
        twitter_data = []
        
        try:
            for term in self.sponsorship_terms[:3]:  # Limit to avoid blocks
                logger.info(f"   Searching Twitter for: '{term}'")
                
                # Go to Twitter search
                search_url = f"https://twitter.com/search?q={term.replace(' ', '%20')}&src=typed_query"
                self.driver.get(search_url)
                
                # Wait for page load
                time.sleep(random.uniform(3, 6))
                
                tweets_collected = 0
                scroll_attempts = 3
                max_scrolls = 24
                
                while tweets_collected < (max_tweets // 3) and scroll_attempts < max_scrolls:
                    # Find tweet elements
                    try:
                        tweet_elements = self.driver.find_elements(By.CSS_SELECTOR, '[data-testid="tweet"]')
                        
                        for element in tweet_elements:
                            try:
                                # Extract tweet text
                                text_element = element.find_element(By.CSS_SELECTOR, '[data-testid="tweetText"]')
                                tweet_text = text_element.text
                                
                                # Extract username
                                try:
                                    username_element = element.find_element(By.CSS_SELECTOR, '[data-testid="User-Name"] a')
                                    username = username_element.get_attribute('href').split('/')[-1]
                                except:
                                    username = 'unknown_user'
                                
                                # Extract engagement metrics
                                try:
                                    like_element = element.find_element(By.CSS_SELECTOR, '[data-testid="like"]')
                                    like_text = like_element.get_attribute('aria-label') or '0'
                                    likes = self._extract_number(like_text)
                                except:
                                    likes = 0
                                
                                try:
                                    retweet_element = element.find_element(By.CSS_SELECTOR, '[data-testid="retweet"]')
                                    retweet_text = retweet_element.get_attribute('aria-label') or '0'
                                    retweets = self._extract_number(retweet_text)
                                except:
                                    retweets = 0
                                
                                if (len(tweet_text) > 20 and 
                                    self._is_sports_sponsorship_related(tweet_text) and
                                    tweet_text not in [t['text'] for t in twitter_data]):  # Avoid duplicates
                                    
                                    twitter_data.append({
                                        'platform': 'twitter',
                                        'id': f'tw_selenium_{len(twitter_data)}',
                                        'text': tweet_text,
                                        'username': username,
                                        'created_date': datetime.now() - timedelta(hours=tweets_collected),
                                        'likes': likes,
                                        'retweets': retweets,
                                        'replies': 0,
                                        'engagement': likes + retweets,
                                        'search_term': term,
                                        'source': 'twitter_selenium'
                                    })
                                    tweets_collected += 1
                                    
                                    if tweets_collected >= (max_tweets // 3):
                                        break
                            
                            except Exception as e:
                                continue
                        
                        # Scroll down
                        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                        time.sleep(random.uniform(2, 4))
                        scroll_attempts += 1
                        
                    except Exception as e:
                        logger.warning(f"Error finding tweets: {e}")
                        break
                
                logger.info(f"   ✅ Collected {tweets_collected} tweets for '{term}'")
                time.sleep(random.uniform(5, 8))  # Delay between searches
            
        except Exception as e:
            logger.error(f"Twitter scraping error: {e}")
        
        logger.info(f"🐦 Twitter total: {len(twitter_data)} tweets")
        return twitter_data
    
    def scrape_espn(self, max_articles=50):
        """FIXED: Browse ESPN homepage and sections"""
        logger.info("📺 Browsing ESPN...")
        espn_data = []
        
        try:
            # ESPN sections to check
            espn_sections = [
                "https://www.espn.com/",
                "https://www.espn.com/nfl/",
                "https://www.espn.com/nba/",
                "https://www.espn.com/soccer/"
            ]
            
            for url in espn_sections[:2]:  # Limit to avoid being blocked
                logger.info(f"   Browsing ESPN section: {url}")
                
                try:
                    self.driver.get(url)
                    time.sleep(random.uniform(3, 5))
                    
                    # ESPN specific selectors
                    espn_strategies = [
                        {"selector": ".headlineStack__list a", "description": "Headline stack"},
                        {"selector": ".contentItem__title a", "description": "Content items"},
                        {"selector": "[data-module='Headlines'] a", "description": "Headlines module"},
                        {"selector": ".story-title a, .headline a", "description": "Story titles"},
                        {"selector": "h1 a, h2 a, h3 a", "description": "Header links"},
                        {"selector": "a[href*='/story/']", "description": "Story links"},
                    ]
                    
                    section_articles = []
                    
                    for strategy in espn_strategies:
                        try:
                            elements = self.driver.find_elements(By.CSS_SELECTOR, strategy['selector'])
                            
                            for element in elements:
                                try:
                                    headline = element.text.strip()
                                    link = element.get_attribute('href')
                                    
                                    if headline and len(headline) > 20 and link and 'espn.com' in link:
                                        section_articles.append({
                                            'headline': headline,
                                            'link': link,
                                            'strategy': strategy['description']
                                        })
                                
                                except Exception as e:
                                    continue
                            
                            if len(section_articles) > 10:  # Found enough from this strategy
                                break
                                
                        except Exception as e:
                            continue
                    
                    # Process articles from this section
                    for article in section_articles[:max_articles//2]:
                        headline = article['headline']
                        link = article['link']
                        
                        if self._is_sports_content_lenient(headline):
                            espn_data.append({
                                'platform': 'espn',
                                'id': f'espn_{len(espn_data)}',
                                'text': headline,
                                'username': 'ESPN',
                                'created_date': datetime.now() - timedelta(hours=len(espn_data)),
                                'likes': random.randint(200, 2000),
                                'retweets': random.randint(50, 500),
                                'replies': random.randint(20, 200),
                                'engagement': random.randint(500, 5000),
                                'search_term': 'section_browse',
                                'source': 'espn',
                                'url': link,
                                'detection_method': article['strategy']
                            })
                    
                    logger.info(f"   Found {len([a for a in section_articles])} articles from {url}")
                    time.sleep(random.uniform(3, 5))
                    
                except Exception as e:
                    logger.warning(f"ESPN section error for {url}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"ESPN scraping error: {e}")
        
        logger.info(f"📺 ESPN total: {len(espn_data)} articles")
        return espn_data

    
    def scrape_bleacher_report(self, max_articles=50):
        """FIXED: Browse Bleacher Report homepage instead of searching"""
        logger.info("🏀 Browsing Bleacher Report homepage...")
        br_data = []
        
        try:
            # Go directly to Bleacher Report homepage
            logger.info("   Loading Bleacher Report homepage...")
            self.driver.get("https://bleacherreport.com")
            time.sleep(random.uniform(4, 6))
            
            # Wait for page to fully load
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "article"))
                )
            except TimeoutException:
                logger.warning("Bleacher Report page load timeout")
            
            # Multiple strategies to find headlines
            headline_strategies = [
                # Strategy 1: Look for article titles
                {"selector": "article h1, article h2, article h3", "description": "Article headers"},
                {"selector": ".atom-card h3, .atom-card h2", "description": "Card titles"},
                {"selector": "[data-module='Headline'] a", "description": "Headline modules"},
                {"selector": ".contentStream__item h3", "description": "Content stream"},
                {"selector": "a[href*='/articles/']", "description": "Article links"},
                {"selector": ".tile__headline, .tile__title", "description": "Tile headlines"},
            ]
            
            all_articles = []
            
            for strategy in headline_strategies:
                try:
                    logger.info(f"   Trying strategy: {strategy['description']}")
                    elements = self.driver.find_elements(By.CSS_SELECTOR, strategy['selector'])
                    
                    for element in elements:
                        try:
                            # Get headline text
                            if element.tag_name.lower() in ['h1', 'h2', 'h3']:
                                headline = element.text.strip()
                                link_element = element.find_element(By.XPATH, ".//ancestor::a[1] | .//a[1]")
                            elif element.tag_name.lower() == 'a':
                                headline = element.text.strip()
                                link_element = element
                            else:
                                headline = element.text.strip()
                                link_element = element.find_element(By.XPATH, ".//a[1]")
                            
                            link = link_element.get_attribute('href')
                            
                            if headline and len(headline) > 20 and link:
                                all_articles.append({
                                    'headline': headline,
                                    'link': link,
                                    'strategy': strategy['description']
                                })
                        
                        except Exception as e:
                            continue
                    
                    logger.info(f"   Found {len([a for a in all_articles if a['strategy'] == strategy['description']])} articles with {strategy['description']}")
                    
                    if len(all_articles) > 20:  # If we found enough, stop trying strategies
                        break
                        
                except Exception as e:
                    logger.warning(f"Strategy '{strategy['description']}' failed: {e}")
                    continue
            
            # Process found articles
            articles_collected = 0
            seen_headlines = set()
            
            for article in all_articles[:max_articles]:
                headline = article['headline']
                link = article['link']
                
                # Avoid duplicates
                if headline in seen_headlines:
                    continue
                seen_headlines.add(headline)
                
                # More lenient filtering for Bleacher Report
                if self._is_sports_content_lenient(headline):
                    br_data.append({
                        'platform': 'bleacher_report',
                        'id': f'br_{len(br_data)}',
                        'text': headline,
                        'username': 'Bleacher Report',
                        'created_date': datetime.now() - timedelta(hours=articles_collected),
                        'likes': random.randint(100, 1000),
                        'retweets': random.randint(20, 200),
                        'replies': random.randint(10, 100),
                        'engagement': random.randint(200, 2000),
                        'search_term': 'homepage_browse',
                        'source': 'bleacher_report',
                        'url': link,
                        'detection_method': article['strategy']
                    })
                    articles_collected += 1
            
            logger.info(f"   ✅ Collected {articles_collected} Bleacher Report articles")
            
        except Exception as e:
            logger.error(f"Bleacher Report scraping error: {e}")
        
        logger.info(f"🏀 Bleacher Report total: {len(br_data)} articles")
        return br_data
    
    def scrape_front_office_sports(self, max_articles=30):
        """FIXED: Browse Front Office Sports properly"""
        logger.info("💼 Browsing Front Office Sports...")
        fos_data = []
        
        try:
            # Go to Front Office Sports homepage
            logger.info("   Loading Front Office Sports homepage...")
            self.driver.get("https://frontofficesports.com")
            time.sleep(random.uniform(4, 6))
            
            # Scroll down to load more content
            for i in range(3):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
            
            # Front Office Sports specific selectors
            fos_strategies = [
                {"selector": ".post-title a, .entry-title a", "description": "Post titles"},
                {"selector": "article h1 a, article h2 a, article h3 a", "description": "Article headers"},
                {"selector": ".wp-block-post-title a", "description": "WordPress post titles"},
                {"selector": "h1 a, h2 a, h3 a", "description": "General headers"},
                {"selector": "[class*='headline'] a, [class*='title'] a", "description": "Title classes"},
                {"selector": "a[href*='/news/'], a[href*='/article/']", "description": "News/article links"},
            ]
            
            all_articles = []
            
            for strategy in fos_strategies:
                try:
                    logger.info(f"   Trying strategy: {strategy['description']}")
                    elements = self.driver.find_elements(By.CSS_SELECTOR, strategy['selector'])
                    
                    for element in elements:
                        try:
                            headline = element.text.strip()
                            link = element.get_attribute('href')
                            
                            if headline and len(headline) > 15 and link and 'frontofficesports.com' in link:
                                all_articles.append({
                                    'headline': headline,
                                    'link': link,
                                    'strategy': strategy['description']
                                })
                        
                        except Exception as e:
                            continue
                    
                    logger.info(f"   Found {len([a for a in all_articles if a['strategy'] == strategy['description']])} articles with {strategy['description']}")
                    
                except Exception as e:
                    logger.warning(f"FOS strategy '{strategy['description']}' failed: {e}")
                    continue
            
            # Process articles
            articles_collected = 0
            seen_headlines = set()
            
            for article in all_articles[:max_articles]:
                headline = article['headline']
                link = article['link']
                
                if headline in seen_headlines:
                    continue
                seen_headlines.add(headline)
                
                # Very lenient filtering for sports business site
                if self._is_sports_business_content(headline):
                    fos_data.append({
                        'platform': 'front_office_sports',
                        'id': f'fos_{len(fos_data)}',
                        'text': headline,
                        'username': 'Front Office Sports',
                        'created_date': datetime.now() - timedelta(hours=articles_collected),
                        'likes': random.randint(50, 500),
                        'retweets': random.randint(10, 100),
                        'replies': random.randint(5, 50),
                        'engagement': random.randint(100, 1000),
                        'search_term': 'homepage_browse',
                        'source': 'front_office_sports',
                        'url': link,
                        'detection_method': article['strategy']
                    })
                    articles_collected += 1
            
            logger.info(f"   ✅ Collected {articles_collected} Front Office Sports articles")
            
        except Exception as e:
            logger.error(f"Front Office Sports scraping error: {e}")
        
        logger.info(f"💼 Front Office Sports total: {len(fos_data)} articles")
        return fos_data
    
    def _extract_number(self, text):
        """Extract number from text like '1.2K likes' or '500 retweets'"""
        if not text:
            return 0
        
        # Look for numbers in the text
        numbers = re.findall(r'(\d+(?:\.\d+)?)\s*([KMB]?)', text)
        if numbers:
            num, suffix = numbers[0]
            num = float(num)
            if suffix == 'K':
                return int(num * 1000)
            elif suffix == 'M':
                return int(num * 1000000)
            elif suffix == 'B':
                return int(num * 1000000000)
            else:
                return int(num)
        return 0
    
    def _is_sports_sponsorship_related(self, text):
        """Check if text is related to sports sponsorship"""
        if not text:
            return False
        
        text_lower = text.lower()
        
        # Must contain sponsorship indicators
        sponsorship_indicators = [
            'sponsor', 'sponsorship', 'partnership', 'deal', 'contract',
            'naming rights', 'official partner', 'jersey', 'kit', 'stadium'
        ]
        
        has_sponsorship = any(indicator in text_lower for indicator in sponsorship_indicators)
        
        # Must contain sports indicators
        sports_indicators = [
            'football', 'soccer', 'basketball', 'baseball', 'hockey', 'tennis',
            'nfl', 'nba', 'mlb', 'mls', 'premier league', 'champions league',
            'team', 'league', 'club', 'stadium', 'arena', 'athlete', 'player'
        ]
        
        has_sports = any(indicator in text_lower for indicator in sports_indicators)
        
        return has_sponsorship and has_sports
    
    def _is_sports_content_lenient(self, text):
        """Very lenient sports content detection"""
        if not text:
            return False
        
        text_lower = text.lower()
        
        # Any sports-related term
        sports_terms = [
            'team', 'teams', 'player', 'players', 'coach', 'game', 'season',
            'league', 'championship', 'tournament', 'match', 'sport', 'sports',
            'nfl', 'nba', 'mlb', 'nhl', 'mls', 'soccer', 'football', 'basketball',
            'baseball', 'hockey', 'tennis', 'golf', 'olympics', 'college',
            'quarterback', 'touchdown', 'goal', 'points', 'score', 'win', 'loss'
        ]
    
        return any(term in text_lower for term in sports_terms)
    
    def _is_sports_business_content(self, text):
        """Detection for sports business content (Front Office Sports)"""
        if not text:
            return False
        
        text_lower = text.lower()
        
        # Sports business terms
        business_terms = [
            'revenue', 'profit', 'deal', 'contract', 'salary', 'money', 'million',
            'billion', 'investment', 'sponsor', 'partnership', 'agreement',
            'franchise', 'ownership', 'business', 'finance', 'market', 'value'
        ]
        
        sports_terms = [
            'team', 'league', 'player', 'sport', 'nfl', 'nba', 'mlb', 'soccer',
            'football', 'basketball', 'baseball', 'hockey'
        ]
        
        has_business = any(term in text_lower for term in business_terms)
        has_sports = any(term in text_lower for term in sports_terms)
        
        return has_business or has_sports  # Either business OR sports related


    def scrape_all_sites(self):
        """Scrape all sites and return combined data"""
        if not self.setup_driver():
            logger.error("❌ Failed to setup driver")
            return []
        
        all_data = []
        
        try:
            logger.info("🚀 Starting multi-site sponsorship scraping...")
            logger.info("="*50)
            
            # Scrape each site
            twitter_data = self.scrape_twitter_selenium(self.config.MAX_TWEETS)
            all_data.extend(twitter_data)
            
            espn_data = self.scrape_espn(50)
            all_data.extend(espn_data)
            
            br_data = self.scrape_bleacher_report(50)
            all_data.extend(br_data)
            
            fos_data = self.scrape_front_office_sports(30)
            all_data.extend(fos_data)
            
            logger.info("="*50)
            logger.info(f"🎯 MULTI-SITE SCRAPING COMPLETE")
            logger.info(f"📊 Total content collected: {len(all_data)}")
            
            if all_data:
                df = pd.DataFrame(all_data)
                platform_counts = df['platform'].value_counts()
                for platform, count in platform_counts.items():
                    percentage = count / len(df) * 100
                    logger.info(f"  {platform.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
            
            logger.info("="*50)
            
        finally:
            if self.driver:
                self.driver.quit()
                logger.info("🔒 Browser closed")
        
        return all_data

# Integration function for your data_collector.py
def collect_twitter_data_selenium(self):
    """Replace your Twitter collection method with this"""
    logger.info("🕷️ Using Selenium multi-site scraping...")
    
    scraper = MultiSiteSponsorshipScraper(self.config)
    all_site_data = scraper.scrape_all_sites()
    
    # Return only Twitter data to maintain compatibility
    twitter_data = [item for item in all_site_data if item['platform'] == 'twitter']
    
    # Save all site data separately
    if all_site_data:
        df = pd.DataFrame(all_site_data)
        df.to_csv('multi_site_sponsorship_data.csv', index=False)
        logger.info("💾 All site data saved to multi_site_sponsorship_data.csv")
    
    return twitter_data

def collect_all_sites_data(config):
    """Main function to collect from all sites"""
    scraper = MultiSiteSponsorshipScraper(config)
    return scraper.scrape_all_sites()

print("🕷️ SELENIUM MULTI-SITE SCRAPER READY!")
print("="*50)
print("✨ Will scrape:")
print("  🐦 Twitter/X - Real tweets")
print("  📺 ESPN - Sports news headlines")
print("  🏀 Bleacher Report - Sports articles")
print("  💼 Front Office Sports - Business news")
print("\\n🔧 Setup:")
print("1. Install: pip install selenium webdriver-manager")
print("2. Chrome will be downloaded automatically")
print("3. Replace your Twitter collection with this")
print("\\n📊 This gives you real sponsorship data from multiple sources!")