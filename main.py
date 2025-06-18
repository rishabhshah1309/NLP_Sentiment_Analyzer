"""
Complete Main Script for Sports Sponsorship Analysis (always runs Selenium for X)
Rishabh Shah
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ---- Logging setup ----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('complete_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---- Import config ----
try:
    from config import Config as config
    logger.info("Loaded main config.")
except Exception as e:
    logger.warning(f"Config import failed: {e}, using minimal config.")
    class MinimalConfig:
        MAX_TWEETS = 100
        MAX_REDDIT_POSTS = 50
        MAX_SELENIUM_POSTS = 50
        OUTPUT_FILE = 'analysis_results.csv'
        RESULTS_DIR = 'results'
        MIN_TEXT_LENGTH = 15
        CONFIDENCE_THRESHOLD = 0.6
        GENERATE_VISUALIZATIONS = True
        DEBUG_MODE = False
    config = MinimalConfig()

#os.makedirs(config.RESULTS_DIR, exist_ok=True)

# ---- Data extraction ----
try:
    from data_collector import collect_enhanced_sponsorship_data
    logger.info("Using enhanced API data collector.")
except Exception as e:
    from data_collector import collect_sponsorship_data
    logger.warning(f"Enhanced collector failed: {e}, using basic data collector.")
    collect_enhanced_sponsorship_data = collect_sponsorship_data

try:
    api_data = collect_enhanced_sponsorship_data(config)
    logger.info(f"Extracted {len(api_data)} posts from Reddit & Instagram via API.")
except Exception as e:
    logger.error(f"API data collection failed: {e}")
    api_data = []

try:
    from selenium_scraper import XSeleniumScraper
    logger.info("XSeleniumScraper imported successfully.")
except Exception as e:
    logger.error(f"XSeleniumScraper import failed: {e}")
    XSeleniumScraper = None

if XSeleniumScraper:
    try:
        logger.info("Extracting X (Twitter) data via Selenium...")
        x_scraper = XSeleniumScraper()
        x_data = x_scraper.extract_posts()
        logger.info(f"Extracted {len(x_data)} posts from X via Selenium.")
    except Exception as e:
        logger.error(f"Selenium scraping for X failed: {e}")
        x_data = []
else:
    x_data = []

# ---- Combine all extracted data ----
df_api = pd.DataFrame(api_data)
df_x = pd.DataFrame(x_data)
if not df_api.empty and not df_x.empty:
    df_all = pd.concat([df_api, df_x], ignore_index=True)
elif not df_api.empty:
    df_all = df_api.copy()
elif not df_x.empty:
    df_all = df_x.copy()
else:
    logger.error("No data extracted from any source.")
    sys.exit(1)

logger.info(f"Total posts after merging: {len(df_all)}")

# ---- NLP and Sponsorship/Partnership Extraction ----
try:
    from nlp_processor import process_advanced_sponsorship_data
    nlp_process_func = process_advanced_sponsorship_data
    logger.info("Using advanced NLP processor.")
except Exception as e:
    from nlp_processor import process_sponsorship_data
    nlp_process_func = process_sponsorship_data
    logger.warning(f"Advanced NLP processor failed: {e}, using basic NLP processor.")

try:
    processed_df = nlp_process_func(df_all, config)
    logger.info("NLP processing complete.")
except Exception as e:
    logger.error(f"NLP processing failed: {e}")
    processed_df = df_all

# ---- GPT-2 Fine-Tuning (if used in your pipeline) ----
try:
    from integrated_gpt2_system import refine_with_gpt2
    processed_df = refine_with_gpt2(processed_df, config)
    logger.info("GPT-2 refinement complete.")
except Exception as e:
    logger.warning(f"GPT-2 refinement skipped/failed: {e}")

# ---- Sentiment & Keyword Analysis ----
try:
    from sentiment_keyword_analyzer import analyze_keyword_sentiments_comprehensive
    keyword_results = analyze_keyword_sentiments_comprehensive(processed_df, config)
    logger.info("Keyword sentiment analysis complete.")
except Exception as e:
    from sentiment_keyword_analyzer import analyze_sentiment_drivers
    keyword_results = analyze_sentiment_drivers(processed_df, config)
    logger.warning(f"Comprehensive sentiment analyzer failed: {e}, using basic.")

# ---- Save Results ----
analysis_outfile = os.path.join("results", config.OUTPUT_FILE)
try:
    processed_df.to_csv(analysis_outfile, index=False)
    logger.info(f"Processed data saved to {analysis_outfile}")
except Exception as e:
    logger.error(f"Failed to save processed data: {e}")

# ---- Visualization ----
if getattr(config, "GENERATE_VISUALIZATIONS", True):
    try:
        from sentiment_keyword_analyzer import generate_all_visualizations
        generate_all_visualizations(processed_df, keyword_results, config)
        logger.info("Visualizations generated.")
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")

logger.info("Pipeline complete. Results in results/ directory.")

