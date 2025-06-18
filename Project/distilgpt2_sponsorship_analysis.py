import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    GPT2Tokenizer, GPT2LMHeadModel, GPT2ForSequenceClassification,
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import re
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class SponsorshipDataset(Dataset):
    """Custom dataset for sponsorship classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DistilGPT2SponsorshipAnalyzer:
    """
    Sponsorship analysis using DistilGPT2 and other transformer models
    """
    
    def __init__(self, model_name='distilgpt2'):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        
        # Define sponsorship categories
        self.sponsorship_categories = {
            0: 'not_sponsorship',
            1: 'jersey_partnership', 
            2: 'naming_rights',
            3: 'brand_partnership',
            4: 'title_sponsorship',
            5: 'equipment_deal'
        }
        
        # Sponsorship keywords for enhanced filtering
        self.sponsorship_keywords = {
            'jersey_partnership': [
                'jersey sponsor', 'shirt sponsor', 'kit sponsor', 'front of shirt',
                'jersey deal', 'uniform sponsor', 'kit deal', 'sleeve sponsor'
            ],
            'naming_rights': [
                'stadium naming', 'arena naming', 'naming rights', 'venue sponsor',
                'ballpark naming', 'field naming', 'stadium sponsor'
            ],
            'brand_partnership': [
                'official partner', 'global partner', 'strategic partner',
                'exclusive partner', 'official sponsor', 'partnership deal'
            ],
            'title_sponsorship': [
                'title sponsor', 'presenting sponsor', 'league sponsor',
                'tournament sponsor', 'competition sponsor'
            ],
            'equipment_deal': [
                'kit supplier', 'equipment sponsor', 'apparel deal',
                'gear sponsor', 'uniform supplier'
            ]
        }
    
    def setup_model_for_classification(self):
        """
        Set up DistilGPT2 model for sequence classification
        """
        print(f"Loading {self.model_name} for sequence classification...")
        
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        
        # Add padding token (GPT2 doesn't have one by default)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model for classification
        self.model = GPT2ForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.sponsorship_categories),
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        self.model.to(self.device)
        print("Model setup complete!")
    
    def create_sponsorship_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create training labels based on keyword matching and manual rules
        """
        print("Creating sponsorship labels...")
        
        def classify_text(text):
            if pd.isna(text):
                return 'not_sponsorship'
            
            text_lower = str(text).lower()
            
            # More lenient sponsorship detection
            sponsorship_indicators = [
                'sponsor', 'sponsorship', 'deal', 'partnership', 'agreement',
                'jersey', 'shirt', 'kit', 'stadium', 'arena', 'naming',
                'official partner', 'brand', 'logo', 'contract'
            ]
            
            if not any(indicator in text_lower for indicator in sponsorship_indicators):
                return 'not_sponsorship'
            
            # Classify by specific type with more flexible matching
            for category, keywords in self.sponsorship_keywords.items():
                # Use partial matching for better coverage
                if any(keyword in text_lower for keyword in keywords):
                    return category
            
            # Check for general sponsorship terms
            general_sponsor_terms = ['sponsor', 'sponsorship', 'deal', 'contract', 'agreement']
            if any(term in text_lower for term in general_sponsor_terms):
                return 'brand_partnership'  # Default sponsorship type
                
            return 'not_sponsorship'
        
        df['sponsorship_label'] = df['text'].apply(classify_text)
        
        # Encode labels for the original distribution
        df['label_encoded'] = self.label_encoder.fit_transform(df['sponsorship_label'])
        
        print("Label distribution:")
        label_counts = df['sponsorship_label'].value_counts()
        print(label_counts)
        
        # Check if we have enough sponsorship data
        sponsorship_count = len(df[df['sponsorship_label'] != 'not_sponsorship'])
        print(f"Total sponsorship-related posts: {sponsorship_count}")
        
        if sponsorship_count < 10:
            print("WARNING: Very few sponsorship examples found. Consider:")
            print("1. Using more specific search terms when collecting data")
            print("2. Collecting from more subreddits")
            print("3. Looking for recent major sponsorship announcements")
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple:
        """
        Prepare data for training with better handling of small classes
        """
        print("Preparing training data...")
        
        # Check class distribution
        class_counts = df['sponsorship_label'].value_counts()
        print("Original class distribution:")
        print(class_counts)
        
        # Filter classes with at least 4 examples (minimum for stratified split)
        valid_classes = class_counts[class_counts >= 4].index.tolist()
        
        # Keep only classes with enough examples
        df_filtered = df[df['sponsorship_label'].isin(valid_classes)].copy()
        
        print(f"Keeping classes with ≥4 examples: {valid_classes}")
        print(f"Filtered dataset size: {len(df_filtered)}")
        
        # If we have very few sponsorship examples, create a simplified binary classification
        sponsorship_df = df_filtered[df_filtered['sponsorship_label'] != 'not_sponsorship'].copy()
        non_sponsorship_df = df_filtered[df_filtered['sponsorship_label'] == 'not_sponsorship'].copy()
        
        print(f"Sponsorship examples: {len(sponsorship_df)}")
        print(f"Non-sponsorship examples: {len(non_sponsorship_df)}")
        
        # If very few sponsorship examples, do binary classification instead
        if len(sponsorship_df) < 20:
            print("Too few sponsorship examples for multi-class. Using binary classification.")
            
            # Create binary labels: 0 = not_sponsorship, 1 = sponsorship
            df_filtered['binary_label'] = (df_filtered['sponsorship_label'] != 'not_sponsorship').astype(int)
            
            # Balance the dataset
            min_samples = min(len(sponsorship_df), len(non_sponsorship_df), 500)
            
            if len(sponsorship_df) > 0 and len(non_sponsorship_df) > 0:
                sponsorship_sample = sponsorship_df.sample(n=min(len(sponsorship_df), min_samples), random_state=42)
                non_sponsorship_sample = non_sponsorship_df.sample(n=min(len(non_sponsorship_df), min_samples), random_state=42)
                
                training_df = pd.concat([sponsorship_sample, non_sponsorship_sample], ignore_index=True)
                
                # Use binary labels
                labels = training_df['binary_label'].tolist()
            else:
                print("Not enough data for training. Need both sponsorship and non-sponsorship examples.")
                return [], [], [], []
        else:
            # Multi-class classification
            # Balance classes - sample up to 100 examples per class
            balanced_dfs = []
            for class_name in valid_classes:
                class_df = df_filtered[df_filtered['sponsorship_label'] == class_name]
                if len(class_df) > 100:
                    class_df = class_df.sample(n=100, random_state=42)
                balanced_dfs.append(class_df)
            
            training_df = pd.concat(balanced_dfs, ignore_index=True)
            
            # Re-encode labels for the filtered classes
            training_df['filtered_label_encoded'] = self.label_encoder.fit_transform(training_df['sponsorship_label'])
            labels = training_df['filtered_label_encoded'].tolist()
        
        # Get texts
        texts = training_df['text'].tolist()
        
        # Check if we have enough data for stratified split
        if len(set(labels)) == 1 or len(texts) < 4:
            print("Not enough diverse data for stratified split. Using simple split.")
            split_idx = int(len(texts) * (1 - test_size))
            train_texts = texts[:split_idx]
            val_texts = texts[split_idx:]
            train_labels = labels[:split_idx]
            val_labels = labels[split_idx:]
        else:
            # Try stratified split
            try:
                train_texts, val_texts, train_labels, val_labels = train_test_split(
                    texts, labels,
                    test_size=test_size,
                    random_state=42,
                    stratify=labels
                )
            except ValueError as e:
                print(f"Stratified split failed: {e}")
                print("Using simple random split instead.")
                train_texts, val_texts, train_labels, val_labels = train_test_split(
                    texts, labels,
                    test_size=test_size,
                    random_state=42
                )
        
        print(f"Training samples: {len(train_texts)}")
        print(f"Validation samples: {len(val_texts)}")
        print(f"Training label distribution: {pd.Series(train_labels).value_counts().to_dict()}")
        
        return train_texts, val_texts, train_labels, val_labels
    
    def create_datasets(self, train_texts, val_texts, train_labels, val_labels):
        """
        Create PyTorch datasets
        """
        train_dataset = SponsorshipDataset(
            train_texts, train_labels, self.tokenizer, max_length=256
        )
        val_dataset = SponsorshipDataset(
            val_texts, val_labels, self.tokenizer, max_length=256
        )
        
        return train_dataset, val_dataset
    
    def train_model(self, train_dataset, val_dataset, output_dir='./sponsorship_model'):
        """
        Fine-tune DistilGPT2 for sponsorship classification
        """
        print("Starting model training...")
        
        # Check if we have data to train on
        if len(train_dataset) == 0:
            print("No training data available. Skipping model training.")
            return None
        
        # Training arguments (compatible with older transformers versions)
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,  # Smaller batch size for stability
            per_device_eval_batch_size=4,
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
            eval_steps=100,  # Use eval_steps instead of evaluation_strategy
            save_steps=100,  # Use save_steps instead of save_strategy
            load_best_model_at_end=True,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Train the model
        try:
            trainer.train()
            
            # Save the model
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            print(f"Model saved to {output_dir}")
            return trainer
        except Exception as e:
            print(f"Training failed: {e}")
            print("Continuing with pre-trained model for inference...")
            return None
    
    def compute_metrics(self, eval_pred):
        """
        Compute metrics for evaluation
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy
        }
    
    def predict_sponsorship_type(self, text: str) -> Dict:
        """
        Predict sponsorship type for a single text
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=256
        ).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        # Get category name
        category = self.sponsorship_categories[predicted_class]
        
        return {
            'predicted_category': category,
            'confidence': confidence,
            'all_probabilities': {
                self.sponsorship_categories[i]: prob.item() 
                for i, prob in enumerate(predictions[0])
            }
        }
    
    def batch_predict(self, texts: List[str], batch_size: int = 16) -> List[Dict]:
        """
        Predict sponsorship types for multiple texts
        """
        print(f"Predicting sponsorship types for {len(texts)} texts...")
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            for text in batch_texts:
                try:
                    prediction = self.predict_sponsorship_type(text)
                    results.append(prediction)
                except Exception as e:
                    print(f"Error predicting for text: {e}")
                    results.append({
                        'predicted_category': 'not_sponsorship',
                        'confidence': 0.0,
                        'all_probabilities': {}
                    })
        
        return results
    
    def analyze_sponsorship_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze entire dataset with trained model
        """
        print("Analyzing dataset with DistilGPT2 model...")
        
        # Get predictions
        predictions = self.batch_predict(df['text'].tolist())
        
        # Add predictions to dataframe
        df['gpt2_predicted_category'] = [p['predicted_category'] for p in predictions]
        df['gpt2_confidence'] = [p['confidence'] for p in predictions]
        
        # Add probability scores for each category
        for category in self.sponsorship_categories.values():
            df[f'gpt2_prob_{category}'] = [
                p['all_probabilities'].get(category, 0.0) for p in predictions
            ]
        
        # Filter for high-confidence sponsorship predictions
        sponsorship_mask = (
            (df['gpt2_predicted_category'] != 'not_sponsorship') &
            (df['gpt2_confidence'] > 0.6)
        )
        
        print(f"High-confidence sponsorship predictions: {sponsorship_mask.sum()}")
        
        return df
    
    def generate_analysis_report(self, df: pd.DataFrame):
        """
        Generate comprehensive analysis report
        """
        print("\n" + "="*70)
        print("DISTILGPT2 SPONSORSHIP ANALYSIS REPORT")
        print("="*70)
        
        # Overall predictions
        print("\n1. MODEL PREDICTIONS OVERVIEW")
        prediction_counts = df['gpt2_predicted_category'].value_counts()
        total_predictions = len(df)
        
        for category, count in prediction_counts.items():
            percentage = count / total_predictions * 100
            print(f"   {category.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        # High confidence predictions
        high_conf_mask = df['gpt2_confidence'] > 0.7
        high_conf_count = high_conf_mask.sum()
        print(f"\n   High confidence predictions (>0.7): {high_conf_count} ({high_conf_count/total_predictions*100:.1f}%)")
        
        # Sponsorship type analysis
        sponsorship_df = df[df['gpt2_predicted_category'] != 'not_sponsorship']
        
        if len(sponsorship_df) > 0:
            print("\n2. SPONSORSHIP TYPE BREAKDOWN")
            sponsorship_types = sponsorship_df['gpt2_predicted_category'].value_counts()
            
            for stype, count in sponsorship_types.items():
                pct = count / len(sponsorship_df) * 100
                avg_conf = sponsorship_df[sponsorship_df['gpt2_predicted_category'] == stype]['gpt2_confidence'].mean()
                print(f"   {stype.replace('_', ' ').title()}: {count} posts ({pct:.1f}%) - Avg confidence: {avg_conf:.3f}")
        
        # Confidence distribution
        print("\n3. CONFIDENCE DISTRIBUTION")
        conf_stats = df['gpt2_confidence'].describe()
        print(f"   Mean confidence: {conf_stats['mean']:.3f}")
        print(f"   Median confidence: {conf_stats['50%']:.3f}")
        print(f"   Std deviation: {conf_stats['std']:.3f}")
        
        # Top confident predictions by category
        print("\n4. MOST CONFIDENT PREDICTIONS BY CATEGORY")
        for category in ['jersey_partnership', 'naming_rights', 'brand_partnership']:
            category_df = df[df['gpt2_predicted_category'] == category]
            if len(category_df) > 0:
                top_prediction = category_df.loc[category_df['gpt2_confidence'].idxmax()]
                print(f"\n   {category.replace('_', ' ').title()}:")
                print(f"   Confidence: {top_prediction['gpt2_confidence']:.3f}")
                print(f"   Text: {top_prediction['text'][:100]}...")
        
        # Subreddit analysis if available
        if 'subreddit' in df.columns:
            print("\n5. SPONSORSHIP PREDICTIONS BY SUBREDDIT")
            subreddit_sponsorship = df[df['gpt2_predicted_category'] != 'not_sponsorship'].groupby('subreddit').size().sort_values(ascending=False)
            
            for subreddit, count in subreddit_sponsorship.head().items():
                total_in_subreddit = len(df[df['subreddit'] == subreddit])
                pct = count / total_in_subreddit * 100
                print(f"   r/{subreddit}: {count}/{total_in_subreddit} posts ({pct:.1f}%)")

def run_distilgpt2_analysis(data_file: str = 'reddit_sports_sponsorship_data.csv',
                           train_new_model: bool = True):
    """
    Run complete DistilGPT2-based sponsorship analysis
    """
    # Initialize analyzer
    analyzer = DistilGPT2SponsorshipAnalyzer()
    
    # Load data
    try:
        df = pd.read_csv(data_file)
        print(f"Loaded {len(df)} records from {data_file}")
    except FileNotFoundError:
        print(f"Data file {data_file} not found.")
        return None, None
    
    # Setup model
    analyzer.setup_model_for_classification()
    
    # Create labels first to see what we're working with
    df_labeled = analyzer.create_sponsorship_labels(df)
    
    # Check if we have enough sponsorship data
    sponsorship_count = len(df_labeled[df_labeled['sponsorship_label'] != 'not_sponsorship'])
    
    if sponsorship_count < 5:
        print(f"\n⚠️  WARNING: Only found {sponsorship_count} sponsorship-related posts!")
        print("This suggests your current data doesn't contain much sponsorship discussion.")
        print("\n🎯 RECOMMENDATIONS:")
        print("1. Collect data with more specific sponsorship search terms:")
        print("   - 'jersey sponsor deal'")
        print("   - 'stadium naming rights'") 
        print("   - 'Emirates Arsenal'")
        print("   - 'Nike Manchester United'")
        print("2. Try sports business subreddits: r/MLS, r/soccer, r/business")
        print("3. Search for recent major sponsorship announcements")
        
        print("\n📊 For now, I'll analyze with the base DistilGPT2 model (no fine-tuning)")
        train_new_model = False
    
    trainer = None
    if train_new_model and sponsorship_count >= 5:
        # Prepare training data
        train_texts, val_texts, train_labels, val_labels = analyzer.prepare_training_data(df_labeled)
        
        if len(train_texts) > 0:
            # Create datasets
            train_dataset, val_dataset = analyzer.create_datasets(
                train_texts, val_texts, train_labels, val_labels
            )
            
            # Train model
            trainer = analyzer.train_model(train_dataset, val_dataset)
        else:
            print("No valid training data created. Using base model for analysis.")
    
    # Analyze the full dataset (even with base model)
    print("\n🔍 Analyzing dataset with available model...")
    analyzed_df = analyzer.analyze_sponsorship_dataset(df_labeled)
    
    # Generate report
    analyzer.generate_analysis_report(analyzed_df)
    
    # Save results
    analyzed_df.to_csv('distilgpt2_sponsorship_analysis.csv', index=False)
    print(f"\nResults saved to distilgpt2_sponsorship_analysis.csv")
    
    return analyzed_df, analyzer

if __name__ == "__main__":
    # Run the analysis
    results, model = run_distilgpt2_analysis(
        data_file='reddit_sports_sponsorship_data.csv',
        train_new_model=True
    )
    
    # Example of using the trained model for new predictions
    if model:
        sample_text = "Nike just signed a new 10-year jersey sponsorship deal with Manchester United worth $100 million per year"
        prediction = model.predict_sponsorship_type(sample_text)
        print(f"\nSample prediction:")
        print(f"Text: {sample_text}")
        print(f"Predicted category: {prediction['predicted_category']}")
        print(f"Confidence: {prediction['confidence']:.3f}")