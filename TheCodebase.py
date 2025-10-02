import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    get_scheduler,
    get_linear_schedule_with_warmup,
    pipeline
)
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import logging
import time
import warnings
import json
from typing import Dict, List, Tuple, Optional, Union
import random
from collections import Counter
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
import sys
import codecs
import re

warnings.filterwarnings('ignore')

# Ensure stdout can handle Unicode
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dravidian_hos_finetune.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)  # This will now use the UTF-8 encoded stdout
    ]
)
logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    from nltk.corpus import wordnet
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available. Some augmentation methods will be disabled.")

try:
    from googletrans import Translator
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    GOOGLETRANS_AVAILABLE = False
    logger.warning("googletrans not available. Back translation will be disabled.")

try:
    import nlpaug.augmenter.word as naw
    import nlpaug.augmenter.sentence as nas
    NLPAUG_AVAILABLE = True
except ImportError:
    NLPAUG_AVAILABLE = False
    logger.warning("nlpaug not available. Some augmentation methods will be disabled.")

class DataAugmenter:
    """
    Advanced data augmentation pipeline for Dravidian language text classification.
    Implements multiple augmentation techniques to increase training data diversity.
    """
    
    def __init__(self, tokenizer, model_name="ai4bharat/IndicBERTv2-MLM-only", device=None):
        """
        Initialize the data augmenter.
        
        Args:
            tokenizer: Transformer tokenizer
            model_name: Model name for augmentation
            device: Device to run augmentation on
        """
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize augmentation models
        self._initialize_augmenters()
        
        # Download required NLTK data
        if NLTK_AVAILABLE:
            try:
                nltk.download('wordnet', quiet=True)
                nltk.download('punkt', quiet=True)
            except:
                logger.warning("Could not download NLTK data. Some augmentation methods may not work.")
        else:
            logger.warning("NLTK not available - skipping NLTK data download")
        
        logger.info("DataAugmenter initialized successfully")
    
    def _initialize_augmenters(self):
        """Initialize various augmentation models."""
        try:
            # Synonym replacement using WordNet
            if NLTK_AVAILABLE:
                self.synonym_aug = naw.SynonymAug(aug_src='wordnet', lang='eng')
            else:
                self.synonym_aug = None
                logger.warning("NLTK not available - synonym replacement disabled")
            
            # Contextual augmentation using BERT
            if NLPAUG_AVAILABLE:
                self.contextual_aug = naw.ContextualWordEmbsAug(
                    model_path=self.model_name,
                    action="substitute",
                    aug_p=0.3
                )
            else:
                self.contextual_aug = None
                logger.warning("nlpaug not available - contextual augmentation disabled")
            
            # Back translation pipeline
            if GOOGLETRANS_AVAILABLE:
                self.translator = Translator()
            else:
                self.translator = None
                logger.warning("googletrans not available - back translation disabled")
            
            logger.info("Augmentation models initialized successfully")
        except Exception as e:
            logger.warning(f"Some augmentation models could not be initialized: {e}")
            self.synonym_aug = None
            self.contextual_aug = None
            self.translator = None
    
    def synonym_replacement(self, text, replacement_ratio=0.3):
        """
        Replace words with synonyms while preserving meaning.
        
        Args:
            text: Input text
            replacement_ratio: Ratio of words to replace
            
        Returns:
            Augmented text
        """
        try:
            if self.synonym_aug is not None:
                words = text.split()
                n_words = len(words)
                n_replacements = max(1, int(n_words * replacement_ratio))
                
                # Get synonyms for random words
                augmented_text = text
                for _ in range(n_replacements):
                    try:
                        augmented_text = self.synonym_aug.augment(augmented_text)[0]
                    except:
                        break
                        
                return augmented_text if augmented_text != text else text
            else:
                # Fallback: simple word replacement with common synonyms
                return self._simple_synonym_replacement(text, replacement_ratio)
        except Exception as e:
            logger.warning(f"Synonym replacement failed: {e}")
            return text
    
    def _simple_synonym_replacement(self, text, replacement_ratio=0.3):
        """
        Simple synonym replacement without external libraries.
        
        Args:
            text: Input text
            replacement_ratio: Ratio of words to replace
            
        Returns:
            Augmented text
        """
        # Simple synonym dictionary for common words
        simple_synonyms = {
            'good': ['great', 'excellent', 'fine', 'nice'],
            'bad': ['terrible', 'awful', 'horrible', 'poor'],
            'big': ['large', 'huge', 'enormous', 'massive'],
            'small': ['tiny', 'little', 'mini', 'petite'],
            'happy': ['joyful', 'cheerful', 'glad', 'pleased'],
            'sad': ['unhappy', 'miserable', 'depressed', 'gloomy'],
            'fast': ['quick', 'rapid', 'swift', 'speedy'],
            'slow': ['sluggish', 'leisurely', 'gradual', 'unhurried'],
            'beautiful': ['gorgeous', 'stunning', 'lovely', 'attractive'],
            'ugly': ['hideous', 'repulsive', 'unattractive', 'unsightly']
        }
        
        words = text.split()
        n_words = len(words)
        n_replacements = max(1, int(n_words * replacement_ratio))
        
        # Find words that have synonyms
        replaceable_words = []
        for i, word in enumerate(words):
            word_lower = word.lower()
            if word_lower in simple_synonyms:
                replaceable_words.append(i)
        
        # Randomly replace some words
        if replaceable_words and n_replacements > 0:
            indices_to_replace = random.sample(replaceable_words, 
                                             min(n_replacements, len(replaceable_words)))
            
            augmented_words = words.copy()
            for idx in indices_to_replace:
                word_lower = augmented_words[idx].lower()
                if word_lower in simple_synonyms:
                    synonym = random.choice(simple_synonyms[word_lower])
                    # Preserve original case
                    if augmented_words[idx].isupper():
                        synonym = synonym.upper()
                    elif augmented_words[idx].istitle():
                        synonym = synonym.title()
                    augmented_words[idx] = synonym
            
            return ' '.join(augmented_words)
        
        return text
    
    def contextual_augmentation(self, text):
        """
        Use contextual embeddings to replace words.
        
        Args:
            text: Input text
            
        Returns:
            Augmented text
        """
        try:
            if self.contextual_aug is not None:
                augmented_text = self.contextual_aug.augment(text)[0]
                return augmented_text if augmented_text != text else text
            else:
                # Fallback: simple word substitution
                return self._simple_contextual_replacement(text)
        except Exception as e:
            logger.warning(f"Contextual augmentation failed: {e}")
            return text
    
    def _simple_contextual_replacement(self, text):
        """
        Simple contextual replacement without external libraries.
        
        Args:
            text: Input text
            
        Returns:
            Augmented text
        """
        # Simple word substitutions that maintain context
        contextual_replacements = {
            'is': ['was', 'becomes', 'remains'],
            'are': ['were', 'become', 'remain'],
            'was': ['is', 'had been', 'became'],
            'were': ['are', 'had been', 'became'],
            'will': ['would', 'shall', 'might'],
            'can': ['could', 'may', 'might'],
            'should': ['could', 'would', 'might'],
            'very': ['really', 'extremely', 'quite'],
            'really': ['very', 'extremely', 'quite'],
            'extremely': ['very', 'really', 'quite']
        }
        
        words = text.split()
        if len(words) < 2:
            return text
        
        # Randomly replace one word if possible
        for i, word in enumerate(words):
            word_lower = word.lower()
            if word_lower in contextual_replacements:
                replacement = random.choice(contextual_replacements[word_lower])
                # Preserve original case
                if word.isupper():
                    replacement = replacement.upper()
                elif word.istitle():
                    replacement = replacement.title()
                
                augmented_words = words.copy()
                augmented_words[i] = replacement
                return ' '.join(augmented_words)
        
        return text
    
    def back_translation(self, text, target_lang='en'):
        """
        Perform back translation for data augmentation.
        
        Args:
            text: Input text
            target_lang: Target language for translation
            
        Returns:
            Back-translated text
        """
        try:
            if self.translator is not None:
                # Translate to target language
                translated = self.translator.translate(text, dest=target_lang)
                if translated.text == text:
                    return text
                    
                # Translate back to original language
                back_translated = self.translator.translate(translated.text, dest='auto')
                
                return back_translated.text if back_translated.text != text else text
            else:
                # Fallback: simple text variation
                return self._simple_text_variation(text)
        except Exception as e:
            logger.warning(f"Back translation failed: {e}")
            return text
    
    def _simple_text_variation(self, text):
        """
        Simple text variation without translation.
        
        Args:
            text: Input text
            
        Returns:
            Varied text
        """
        # Simple text variations that simulate translation effects
        variations = [
            # Add/remove articles
            lambda t: t.replace(' the ', ' ').replace(' a ', ' ').replace(' an ', ' '),
            lambda t: t.replace(' is ', ' was ').replace(' are ', ' were '),
            lambda t: t.replace(' will ', ' would ').replace(' can ', ' could '),
            # Change punctuation
            lambda t: t.replace('!', '.').replace('?', '.'),
            lambda t: t.replace('.', '!') if t.endswith('.') else t,
            # Add emphasis words
            lambda t: t.replace(' very ', ' really ').replace(' really ', ' very '),
            lambda t: t.replace(' good ', ' great ').replace(' bad ', ' terrible '),
        ]
        
        # Apply a random variation
        variation = random.choice(variations)
        varied_text = variation(text)
        
        # Only return if the text actually changed
        return varied_text if varied_text != text else text
    
    def random_insertion(self, text, insertion_ratio=0.1):
        """
        Randomly insert words from the same sentence.
        
        Args:
            text: Input text
            insertion_ratio: Ratio of words to insert
            
        Returns:
            Augmented text
        """
        try:
            words = text.split()
            n_words = len(words)
            n_insertions = max(1, int(n_words * insertion_ratio))
            
            augmented_words = words.copy()
            for _ in range(n_insertions):
                if len(augmented_words) > 0:
                    random_word = random.choice(augmented_words)
                    random_position = random.randint(0, len(augmented_words))
                    augmented_words.insert(random_position, random_word)
            
            return ' '.join(augmented_words)
        except Exception as e:
            logger.warning(f"Random insertion failed: {e}")
            return text
    
    def random_deletion(self, text, deletion_ratio=0.1):
        """
        Randomly delete words from the text.
        
        Args:
            text: Input text
            deletion_ratio: Ratio of words to delete
            
        Returns:
            Augmented text
        """
        try:
            words = text.split()
            n_words = len(words)
            n_deletions = max(1, int(n_words * deletion_ratio))
            
            if n_deletions >= n_words:
                return text
                
            indices_to_delete = random.sample(range(n_words), n_deletions)
            augmented_words = [word for i, word in enumerate(words) if i not in indices_to_delete]
            
            return ' '.join(augmented_words) if augmented_words else text
        except Exception as e:
            logger.warning(f"Random deletion failed: {e}")
            return text
    
    def sentence_paraphrasing(self, text):
        """
        Create paraphrased versions of the sentence.
        
        Args:
            text: Input text
            
        Returns:
            Paraphrased text
        """
        try:
            # Simple paraphrasing by changing sentence structure
            words = text.split()
            if len(words) < 3:
                return text
                
            # Randomly swap adjacent words (simple paraphrasing)
            if len(words) >= 2:
                idx = random.randint(0, len(words) - 2)
                words[idx], words[idx + 1] = words[idx + 1], words[idx]
                
            return ' '.join(words)
        except Exception as e:
            logger.warning(f"Sentence paraphrasing failed: {e}")
            return text
    
    def augment_text(self, text, label, augmentation_factor=2, methods=None):
        """
        Apply multiple augmentation techniques to create synthetic data.
        
        Args:
            text: Input text
            label: Original label
            augmentation_factor: Number of augmented samples to create
            methods: List of augmentation methods to use
            
        Returns:
            List of augmented (text, label) pairs
        """
        if methods is None:
            methods = ['synonym', 'contextual', 'back_translation', 'insertion', 'deletion', 'paraphrase']
        
        augmented_samples = []
        
        for i in range(augmentation_factor):
            augmented_text = text
            
            # Apply random augmentation method
            method = random.choice(methods)
            
            if method == 'synonym':
                augmented_text = self.synonym_replacement(text, replacement_ratio=0.2)
            elif method == 'contextual':
                augmented_text = self.contextual_augmentation(text)
            elif method == 'back_translation':
                augmented_text = self.back_translation(text)
            elif method == 'insertion':
                augmented_text = self.random_insertion(text, insertion_ratio=0.1)
            elif method == 'deletion':
                augmented_text = self.random_deletion(text, deletion_ratio=0.1)
            elif method == 'paraphrase':
                augmented_text = self.sentence_paraphrasing(text)
            
            # Only add if the text actually changed
            if augmented_text != text and len(augmented_text.strip()) > 0:
                augmented_samples.append((augmented_text, label))
        
        return augmented_samples
    
    def augment_dataset(self, texts, labels, augmentation_factor=2, balance_classes=True):
        """
        Augment entire dataset while maintaining class balance.
        
        Args:
            texts: List of input texts
            labels: List of labels
            augmentation_factor: Number of augmented samples per original sample
            balance_classes: Whether to balance class distribution
            
        Returns:
            Tuple of (augmented_texts, augmented_labels)
        """
        logger.info(f"Starting dataset augmentation with factor {augmentation_factor}")
        
        augmented_texts = texts.copy()
        augmented_labels = labels.copy()
        
        if balance_classes:
            # Calculate class distribution
            label_counts = Counter(labels)
            max_count = max(label_counts.values())
            
            # Augment minority classes more
            for label, count in label_counts.items():
                if count < max_count:
                    # Find texts for this class
                    class_indices = [i for i, l in enumerate(labels) if l == label]
                    
                    # Calculate how many more samples we need
                    needed_samples = max_count - count
                    samples_per_text = max(1, needed_samples // len(class_indices))
                    
                    for idx in class_indices:
                        text = texts[idx]
                        augmented_samples = self.augment_text(
                            text, label, 
                            augmentation_factor=samples_per_text
                        )
                        
                        for aug_text, aug_label in augmented_samples:
                            augmented_texts.append(aug_text)
                            augmented_labels.append(aug_label)
        else:
            # Simple augmentation for all samples
            for text, label in zip(texts, labels):
                augmented_samples = self.augment_text(
                    text, label, 
                    augmentation_factor=augmentation_factor
                )
                
                for aug_text, aug_label in augmented_samples:
                    augmented_texts.append(aug_text)
                    augmented_labels.append(aug_label)
        
        logger.info(f"Dataset augmentation completed. Original: {len(texts)}, Augmented: {len(augmented_texts)}")
        return augmented_texts, augmented_labels

class DravidianTextDataset(Dataset):
    """Dataset for Dravidian language text classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=None, label_to_id=None):
        """
        Initialize dataset.
        
        Args:
            texts: List of input texts
            labels: List of labels
            tokenizer: Transformer tokenizer
            max_length: Maximum sequence length (if None, will be determined dynamically)
            label_to_id: Dictionary mapping labels to IDs
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        
        # Determine appropriate max length if not provided
        if max_length is None:
            # Sample up to 1000 texts to determine average length
            sample_size = min(1000, len(texts))
            sample_texts = random.sample(texts, sample_size)
            sample_encodings = self.tokenizer(sample_texts, padding=False, truncation=False)
            lengths = [len(enc) for enc in sample_encodings['input_ids']]
            # Set max_length to 95th percentile of lengths to cover most cases efficiently
            self.max_length = int(np.percentile(lengths, 95))
            logger.info(f"Dynamically set max_length to {self.max_length} based on data distribution")
        else:
            self.max_length = max_length
            
        # Create label mapping if not provided
        if label_to_id is None and labels is not None:
            unique_labels = sorted(set(labels))
            self.label_to_id = {label: i for i, label in enumerate(unique_labels)}
        else:
            self.label_to_id = label_to_id
            
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Convert to expected format (remove batch dimension added by tokenizer)
        item = {
            key: val.squeeze(0) for key, val in encoding.items()
        }
        
        # Add label if available
        if self.labels is not None:
            item['labels'] = torch.tensor(self.label_to_id[self.labels[idx]])
            
        return item

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance better than standard cross entropy."""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Class weights (can be used for class imbalance)
            gamma: Focusing parameter that reduces relative loss for well-classified examples
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.cross_entropy = nn.CrossEntropyLoss(weight=alpha, reduction='none')
        
    def forward(self, inputs, targets):
        """
        Calculate focal loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Calculated loss
        """
        ce_loss = self.cross_entropy(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Loss for better generalization."""
    
    def __init__(self, num_classes, smoothing=0.1, reduction='mean'):
        """
        Initialize Label Smoothing Loss.
        
        Args:
            num_classes: Number of classes
            smoothing: Smoothing factor
            reduction: Reduction method
        """
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Calculate label smoothing loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Calculated loss
        """
        log_probs = torch.log_softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_one_hot = targets_one_hot * (1 - self.smoothing) + self.smoothing / self.num_classes
        
        loss = -torch.sum(targets_one_hot * log_probs, dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class CombinedLoss(nn.Module):
    """Combined loss function that combines multiple loss types."""
    
    def __init__(self, num_classes, focal_weight=0.7, label_smoothing_weight=0.3, 
                 focal_gamma=2.0, smoothing=0.1, class_weights=None):
        """
        Initialize Combined Loss.
        
        Args:
            num_classes: Number of classes
            focal_weight: Weight for focal loss component
            label_smoothing_weight: Weight for label smoothing component
            focal_gamma: Gamma parameter for focal loss
            smoothing: Smoothing factor for label smoothing
            class_weights: Class weights for handling imbalance
        """
        super(CombinedLoss, self).__init__()
        self.focal_weight = focal_weight
        self.label_smoothing_weight = label_smoothing_weight
        
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        self.label_smoothing_loss = LabelSmoothingLoss(num_classes, smoothing)
        
    def forward(self, inputs, targets):
        """
        Calculate combined loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Combined loss
        """
        focal_loss = self.focal_loss(inputs, targets)
        label_smoothing_loss = self.label_smoothing_loss(inputs, targets)
        
        combined_loss = (self.focal_weight * focal_loss + 
                        self.label_smoothing_weight * label_smoothing_loss)
        
        return combined_loss

class CurriculumLearningScheduler:
    """Curriculum Learning scheduler for progressive difficulty training."""
    
    def __init__(self, total_epochs, difficulty_schedule='linear'):
        """
        Initialize Curriculum Learning Scheduler.
        
        Args:
            total_epochs: Total number of training epochs
            difficulty_schedule: Schedule type ('linear', 'exponential', 'step')
        """
        self.total_epochs = total_epochs
        self.difficulty_schedule = difficulty_schedule
        self.current_epoch = 0
        
    def get_difficulty_factor(self, epoch):
        """
        Get difficulty factor for current epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Difficulty factor between 0 and 1
        """
        if self.difficulty_schedule == 'linear':
            return min(1.0, epoch / self.total_epochs)
        elif self.difficulty_schedule == 'exponential':
            return min(1.0, (epoch / self.total_epochs) ** 2)
        elif self.difficulty_schedule == 'step':
            if epoch < self.total_epochs // 3:
                return 0.3
            elif epoch < 2 * self.total_epochs // 3:
                return 0.6
            else:
                return 1.0
        else:
            return 1.0
    
    def filter_samples_by_difficulty(self, texts, labels, epoch):
        """
        Filter samples based on curriculum difficulty.
        
        Args:
            texts: List of input texts
            labels: List of labels
            epoch: Current epoch
            
        Returns:
            Filtered (texts, labels) based on difficulty
        """
        difficulty_factor = self.get_difficulty_factor(epoch)
        
        # Simple difficulty measure: text length
        text_lengths = [len(text.split()) for text in texts]
        max_length = max(text_lengths)
        min_length = min(text_lengths)
        
        # Calculate threshold based on difficulty
        threshold = min_length + difficulty_factor * (max_length - min_length)
        
        # Filter samples
        filtered_texts = []
        filtered_labels = []
        
        for text, label in zip(texts, labels):
            if len(text.split()) <= threshold:
                filtered_texts.append(text)
                filtered_labels.append(label)
        
        return filtered_texts, filtered_labels

class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence representations."""
    
    def __init__(self, hidden_size):
        """
        Initialize attention pooling.
        
        Args:
            hidden_size: Size of hidden representations
        """
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, hidden_states, attention_mask):
        """
        Apply attention pooling.
        
        Args:
            hidden_states: Sequence of hidden states
            attention_mask: Mask for padding tokens
            
        Returns:
            Pooled representation
        """
        # Expand attention mask to match dimensions
        extended_attention_mask = attention_mask.unsqueeze(-1)
        
        # Calculate attention scores
        attention_scores = self.attention(hidden_states)
        
        # Apply mask to attention scores (set padding tokens to 0 attention)
        attention_scores = attention_scores * extended_attention_mask
        
        # Normalize attention scores
        attention_scores = attention_scores / (attention_scores.sum(dim=1, keepdim=True) + 1e-8)
        
        # Get weighted sum of hidden states
        context = torch.bmm(attention_scores.transpose(1, 2), hidden_states)
        
        return context.squeeze(1)

class DravidianHOSTransformer(nn.Module):
    """Transformer model with custom classification head for Dravidian HOS detection."""
    
    def __init__(self, model_name, num_labels, pooling_type='cls'):
        """
        Initialize the model.
        
        Args:
            model_name: Name of the pre-trained transformer model
            num_labels: Number of output classes
            pooling_type: Type of pooling to use ('cls', 'mean', 'attention')
        """
        super(DravidianHOSTransformer, self).__init__()
        
        # Load pre-trained model with classification head
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            return_dict=True,
            output_hidden_states=True
        )
        
        self.pooling_type = pooling_type
        hidden_size = self.transformer.config.hidden_size
        
        # Add attention pooling if needed
        if pooling_type == 'attention':
            self.attention_pooler = AttentionPooling(hidden_size)
            
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Replace classifier with a custom one
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_labels)
        )
        
        # Replace the transformer's classification head with our own
        if hasattr(self.transformer, 'classifier'):
            self.transformer.classifier = self.classifier
        elif hasattr(self.transformer, 'score'):
            self.transformer.score = self.classifier
            
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for padding
            labels: Ground truth labels (optional)
            
        Returns:
            Model outputs including loss and logits
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        
        hidden_states = outputs.hidden_states[-1]
        
        if self.pooling_type == 'cls':
            # Use CLS token representation
            pooled_output = hidden_states[:, 0]
        elif self.pooling_type == 'mean':
            # Use mean pooling
            pooled_output = (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1).unsqueeze(-1)
        elif self.pooling_type == 'attention':
            # Use attention pooling
            pooled_output = self.attention_pooler(hidden_states, attention_mask)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
            
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Return outputs in the same format as AutoModelForSequenceClassification
        return outputs

class DravidianHOSDetector:
    """Main class for Hate and Offensive Speech detection in Dravidian languages using Transformers."""
    
    def __init__(self, model_name="ai4bharat/IndicBERTv2-MLM-only", data_dir="./cleaned-datasets", 
                 output_dir="./output", pooling_type='cls'):
        """
        Initialize the HOS detector.
        
        Args:
            model_name: Name of the pre-trained model to use
            data_dir: Directory containing the data files
            output_dir: Directory to save outputs
            pooling_type: Type of pooling to use ('cls', 'mean', 'attention')
        """
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.pooling_type = pooling_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Language-specific sequence lengths
        self.lang_seq_lengths = {
            'kannada': 128,
            'tamil': 150,
            'mal': 180  # Malayalam
        }
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Initialize tokenizer
        try:
            logger.info(f"Loading tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        
        # Initialize data augmenter
        self.data_augmenter = DataAugmenter(self.tokenizer, model_name, self.device)
            
        # Task types and languages
        self.tasks = ['offensive', 'sentiment']
        self.languages = ['kannada', 'tamil', 'mal']
        self.splits = ['train', 'test', 'dev']
        
        # Models dictionary to store trained models
        self.models = {}
        # Label mappings dictionary
        self.label_mappings = {}
        # Training configurations
        self.config = {
            'learning_rate': 5e-5,
            'weight_decay': 0.01,
            'batch_size': 16,
            'grad_acc_steps': 2,  # Gradient accumulation steps
            'epochs': 10,
            'warmup_steps': 500,
            'max_grad_norm': 1.0,
            'early_stopping_patience': 3,
            'early_stopping_min_delta': 0.005,
            'use_focal_loss': True,
            'focal_loss_gamma': 2.0,
            'mixed_precision': True,
            # New advanced configurations
            'use_data_augmentation': True,
            'augmentation_factor': 2,
            'use_curriculum_learning': True,
            'use_combined_loss': True,
            'label_smoothing': 0.1,
            'progressive_fine_tuning': True,
            'ensemble_methods': ['voting', 'stacking', 'bagging']
        }
        
    def progressive_fine_tune(self, language, task, stages=['multilingual', 'language_specific', 'task_specific']):
        """
        Progressive fine-tuning approach for better performance.
        
        Args:
            language: Target language
            task: Task type (offensive/sentiment)
            stages: List of fine-tuning stages
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Starting progressive fine-tuning for {language}_{task}")
        
        results = {}
        
        for stage_idx, stage in enumerate(stages):
            logger.info(f"Stage {stage_idx + 1}/{len(stages)}: {stage}")
            
            if stage == 'multilingual':
                # Stage 1: Train on all languages combined
                stage_result = self._train_multilingual_stage(language, task)
            elif stage == 'language_specific':
                # Stage 2: Fine-tune on specific language
                stage_result = self._train_language_specific_stage(language, task)
            elif stage == 'task_specific':
                # Stage 3: Fine-tune on specific task
                stage_result = self._train_task_specific_stage(language, task)
            else:
                logger.warning(f"Unknown stage: {stage}")
                continue
            
            results[stage] = stage_result
            
            if stage_result and 'model' in stage_result:
                # Use the trained model for next stage
                self.models[f"{language}_{task}_{stage}"] = stage_result['model']
        
        # Final evaluation
        final_result = self._evaluate_progressive_model(language, task, results)
        
        return final_result
    
    def _train_multilingual_stage(self, target_language, target_task):
        """Train on all languages combined for better representation learning."""
        logger.info("Training multilingual stage")
        
        # Collect data from all languages for the target task
        all_texts = []
        all_labels = []
        
        for lang in self.languages:
            train_df = self.load_data(lang, target_task, 'train')
            if train_df is not None:
                all_texts.extend(train_df['Text'].tolist())
                all_labels.extend(train_df['Label'].tolist())
        
        if not all_texts:
            logger.error("No data found for multilingual training")
            return None
        
        # Create label mapping
        unique_labels = sorted(set(all_labels))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        
        # Apply data augmentation
        if self.config['use_data_augmentation']:
            all_texts, all_labels = self.data_augmenter.augment_dataset(
                all_texts, all_labels, 
                augmentation_factor=self.config['augmentation_factor']
            )
        
        # Split data (80% train, 20% dev)
        from sklearn.model_selection import train_test_split
        train_texts, dev_texts, train_labels, dev_labels = train_test_split(
            all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        
        # Create datasets
        train_dataset = DravidianTextDataset(
            train_texts, train_labels, self.tokenizer, 
            max_length=128, label_to_id=label_to_id
        )
        
        dev_dataset = DravidianTextDataset(
            dev_texts, dev_labels, self.tokenizer,
            max_length=128, label_to_id=label_to_id
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        # Initialize model
        model = DravidianHOSTransformer(
            self.model_name, num_labels=len(unique_labels), pooling_type=self.pooling_type
        ).to(self.device)
        
        # Train with lower learning rate for multilingual stage
        stage_config = self.config.copy()
        stage_config['learning_rate'] = 3e-5  # Lower learning rate for multilingual
        stage_config['epochs'] = 5  # Fewer epochs for multilingual
        
        training_result = self.train_model(
            model, {'train': train_loader, 'dev': dev_loader}, **stage_config
        )
        
        return training_result
    
    def _train_language_specific_stage(self, language, task):
        """Fine-tune on specific language data."""
        logger.info(f"Training language-specific stage for {language}")
        
        # Load language-specific data
        train_df = self.load_data(language, task, 'train')
        dev_df = self.load_data(language, task, 'dev')
        
        if train_df is None or dev_df is None:
            logger.error(f"No data found for {language}_{task}")
            return None
        
        # Apply data augmentation
        if self.config['use_data_augmentation']:
            train_texts, train_labels = self.data_augmenter.augment_dataset(
                train_df['Text'].tolist(), train_df['Label'].tolist(),
                augmentation_factor=self.config['augmentation_factor']
            )
        else:
            train_texts, train_labels = train_df['Text'].tolist(), train_df['Label'].tolist()
        
        # Create label mapping
        unique_labels = sorted(set(train_labels))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        
        # Create datasets
        train_dataset = DravidianTextDataset(
            train_texts, train_labels, self.tokenizer,
            max_length=self.lang_seq_lengths.get(language, 128),
            label_to_id=label_to_id
        )
        
        dev_dataset = DravidianTextDataset(
            dev_df['Text'].tolist(), dev_df['Label'].tolist(), self.tokenizer,
            max_length=self.lang_seq_lengths.get(language, 128),
            label_to_id=label_to_id
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        # Initialize model (could load from multilingual stage if available)
        model = DravidianHOSTransformer(
            self.model_name, num_labels=len(unique_labels), pooling_type=self.pooling_type
        ).to(self.device)
        
        # Train with standard configuration
        training_result = self.train_model(
            model, {'train': train_loader, 'dev': dev_loader}, **self.config
        )
        
        return training_result
    
    def _train_task_specific_stage(self, language, task):
        """Final fine-tuning stage for specific task."""
        logger.info(f"Training task-specific stage for {language}_{task}")
        
        # This stage focuses on the specific task with more aggressive training
        # Load all available data for this language-task combination
        train_df = self.load_data(language, task, 'train')
        dev_df = self.load_data(language, task, 'dev')
        test_df = self.load_data(language, task, 'test')
        
        if train_df is None:
            logger.error(f"No data found for {language}_{task}")
            return None
        
        # Combine train and dev for final training
        combined_df = pd.concat([train_df, dev_df], ignore_index=True)
        
        # Apply aggressive data augmentation
        if self.config['use_data_augmentation']:
            combined_texts, combined_labels = self.data_augmenter.augment_dataset(
                combined_df['Text'].tolist(), combined_df['Label'].tolist(),
                augmentation_factor=self.config['augmentation_factor'] * 2  # More augmentation
            )
        else:
            combined_texts, combined_labels = combined_df['Text'].tolist(), combined_df['Label'].tolist()
        
        # Create label mapping
        unique_labels = sorted(set(combined_labels))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        
        # Split for final training
        from sklearn.model_selection import train_test_split
        train_texts, dev_texts, train_labels, dev_labels = train_test_split(
            combined_texts, combined_labels, test_size=0.2, random_state=42, stratify=combined_labels
        )
        
        # Create datasets
        train_dataset = DravidianTextDataset(
            train_texts, train_labels, self.tokenizer,
            max_length=self.lang_seq_lengths.get(language, 128),
            label_to_id=label_to_id
        )
        
        dev_dataset = DravidianTextDataset(
            dev_texts, dev_labels, self.tokenizer,
            max_length=self.lang_seq_lengths.get(language, 128),
            label_to_id=label_to_id
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        # Initialize model
        model = DravidianHOSTransformer(
            self.model_name, num_labels=len(unique_labels), pooling_type=self.pooling_type
        ).to(self.device)
        
        # Use more aggressive training for final stage
        final_config = self.config.copy()
        final_config['learning_rate'] = 2e-5  # Lower learning rate for fine-tuning
        final_config['epochs'] = 15  # More epochs for final stage
        
        training_result = self.train_model(
            model, {'train': train_loader, 'dev': dev_loader}, **final_config
        )
        
        return training_result
    
    def _evaluate_progressive_model(self, language, task, stage_results):
        """Evaluate the final progressive model."""
        logger.info(f"Evaluating progressive model for {language}_{task}")
        
        # Load test data
        test_df = self.load_data(language, task, 'test')
        if test_df is None:
            logger.error(f"No test data found for {language}_{task}")
            return None
        
        # Use the final stage model for evaluation
        final_stage = 'task_specific'
        if final_stage in stage_results and 'model' in stage_results[final_stage]:
            model = stage_results[final_stage]['model']
        else:
            logger.error("No final model found for evaluation")
            return None
        
        # Create test dataset
        unique_labels = sorted(set(test_df['Label'].tolist()))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        
        test_dataset = DravidianTextDataset(
            test_df['Text'].tolist(), test_df['Label'].tolist(), self.tokenizer,
            max_length=self.lang_seq_lengths.get(language, 128),
            label_to_id=label_to_id
        )
        
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        # Evaluate
        test_metrics = self.evaluate(model, test_loader)
        
        return {
            'stage_results': stage_results,
            'test_metrics': test_metrics,
            'final_model': model
        }
    
    def advanced_ensemble_predict(self, texts, language, task, ensemble_methods=None):
        """
        Advanced ensemble prediction using multiple methods.
        
        Args:
            texts: List of texts to predict
            language: Language of the texts
            task: Task type (offensive/sentiment)
            ensemble_methods: List of ensemble methods to use
            
        Returns:
            Dictionary with ensemble predictions
        """
        if ensemble_methods is None:
            ensemble_methods = self.config['ensemble_methods']
        
        logger.info(f"Running advanced ensemble prediction with methods: {ensemble_methods}")
        
        # Get multiple models for ensemble
        models = self._get_ensemble_models(language, task)
        
        if not models:
            logger.error("No models found for ensemble prediction")
            return None
        
        # Make predictions with each model
        all_predictions = []
        all_probabilities = []
        
        for model_name, model in models.items():
            logger.info(f"Making predictions with {model_name}")
            
            # Set current model key for label mapping
            self.current_model_key = f"{language}_{task}"
            
            # Make predictions
            predictions = self._predict_with_model(model, texts, language, task)
            
            if predictions:
                all_predictions.append(predictions)
                all_probabilities.append([pred['probabilities'] for pred in predictions])
        
        if not all_predictions:
            logger.error("No valid predictions from ensemble models")
            return None
        
        # Combine predictions using different methods
        ensemble_results = {}
        
        if 'voting' in ensemble_methods:
            ensemble_results['voting'] = self._voting_ensemble(all_predictions, all_probabilities)
        
        if 'stacking' in ensemble_methods:
            ensemble_results['stacking'] = self._stacking_ensemble(all_predictions, all_probabilities)
        
        if 'bagging' in ensemble_methods:
            ensemble_results['bagging'] = self._bagging_ensemble(all_predictions, all_probabilities)
        
        return ensemble_results
    
    def _get_ensemble_models(self, language, task):
        """Get multiple models for ensemble prediction."""
        models = {}
        
        # Try to load different model variants
        model_variants = [
            f"{language}_{task}",
            f"{language}_{task}_multilingual",
            f"{language}_{task}_language_specific",
            f"{language}_{task}_task_specific"
        ]
        
        for variant in model_variants:
            model_path = os.path.join(self.output_dir, variant, f"{variant}_model.pt")
            metadata_path = model_path.replace('.pt', '_metadata.json')
            
            if os.path.exists(model_path) and os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    model = self.load_model(model_path, metadata['num_classes'])
                    if model:
                        models[variant] = model
                        logger.info(f"Loaded ensemble model: {variant}")
                except Exception as e:
                    logger.warning(f"Could not load model {variant}: {e}")
        
        return models
    
    def _predict_with_model(self, model, texts, language, task):
        """Make predictions with a specific model."""
        try:
            model.eval()
            
            # Determine sequence length
            max_length = self.lang_seq_lengths.get(language, 128)
            
            # Tokenize texts
            encodings = self.tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Make predictions
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
            
            # Convert to labels
            id_to_label = self.label_mappings.get(self.current_model_key, {}).get('id_to_label', {})
            
            results = []
            for i, (text, pred, prob) in enumerate(zip(texts, preds, probs)):
                pred_label = id_to_label.get(str(pred.item()), f"unknown_{pred.item()}")
                prob_dict = {id_to_label.get(str(j), f"unknown_{j}"): prob[j].item() for j in range(len(prob))}
                
                results.append({
                    'text': text,
                    'prediction': pred_label,
                    'probabilities': prob_dict
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in model prediction: {e}")
            return None
    
    def _voting_ensemble(self, all_predictions, all_probabilities):
        """Combine predictions using voting."""
        if not all_predictions:
            return None
        
        # Get all unique labels
        all_labels = set()
        for predictions in all_predictions:
            for pred in predictions:
                all_labels.add(pred['prediction'])
        
        ensemble_results = []
        
        for i in range(len(all_predictions[0])):
            # Count votes for each label
            vote_counts = {label: 0 for label in all_labels}
            
            for predictions in all_predictions:
                if i < len(predictions):
                    pred_label = predictions[i]['prediction']
                    vote_counts[pred_label] += 1
            
            # Get majority vote
            majority_label = max(vote_counts, key=vote_counts.get)
            
            # Average probabilities
            avg_probabilities = {}
            for label in all_labels:
                probs = []
                for probabilities in all_probabilities:
                    if i < len(probabilities) and label in probabilities[i]:
                        probs.append(probabilities[i][label])
                avg_probabilities[label] = np.mean(probs) if probs else 0.0
            
            ensemble_results.append({
                'text': all_predictions[0][i]['text'],
                'prediction': majority_label,
                'probabilities': avg_probabilities,
                'vote_counts': vote_counts
            })
        
        return ensemble_results
    
    def _stacking_ensemble(self, all_predictions, all_probabilities):
        """Combine predictions using stacking (meta-learning)."""
        # This is a simplified stacking implementation
        # In practice, you would train a meta-learner on the predictions
        
        # For now, use weighted average based on model confidence
        ensemble_results = []
        
        for i in range(len(all_predictions[0])):
            # Calculate confidence scores for each model
            confidences = []
            for probabilities in all_probabilities:
                if i < len(probabilities):
                    max_prob = max(probabilities[i].values())
                    confidences.append(max_prob)
                else:
                    confidences.append(0.0)
            
            # Weight predictions by confidence
            weighted_probs = {}
            total_weight = sum(confidences)
            
            if total_weight > 0:
                for j, probabilities in enumerate(all_probabilities):
                    if i < len(probabilities):
                        weight = confidences[j] / total_weight
                        for label, prob in probabilities[i].items():
                            if label not in weighted_probs:
                                weighted_probs[label] = 0.0
                            weighted_probs[label] += prob * weight
                
                # Get prediction with highest weighted probability
                if weighted_probs:
                    prediction = max(weighted_probs, key=weighted_probs.get)
                else:
                    prediction = all_predictions[0][i]['prediction']
            else:
                prediction = all_predictions[0][i]['prediction']
                weighted_probs = all_probabilities[0][i]
            
            ensemble_results.append({
                'text': all_predictions[0][i]['text'],
                'prediction': prediction,
                'probabilities': weighted_probs,
                'model_confidences': confidences
            })
        
        return ensemble_results
    
    def _bagging_ensemble(self, all_predictions, all_probabilities):
        """Combine predictions using bagging approach."""
        # Simple bagging: average probabilities and take argmax
        ensemble_results = []
        
        for i in range(len(all_predictions[0])):
            # Get all unique labels
            all_labels = set()
            for probabilities in all_probabilities:
                if i < len(probabilities):
                    all_labels.update(probabilities[i].keys())
            
            # Average probabilities across all models
            avg_probabilities = {}
            for label in all_labels:
                probs = []
                for probabilities in all_probabilities:
                    if i < len(probabilities) and label in probabilities[i]:
                        probs.append(probabilities[i][label])
                avg_probabilities[label] = np.mean(probs) if probs else 0.0
            
            # Get prediction with highest average probability
            prediction = max(avg_probabilities, key=avg_probabilities.get)
            
            ensemble_results.append({
                'text': all_predictions[0][i]['text'],
                'prediction': prediction,
                'probabilities': avg_probabilities
            })
        
        return ensemble_results
    
    def load_data(self, language, task, split):
        """
        Load data from CSV files.
        
        Args:
            language: Language of the dataset
            task: Task type (offensive/sentiment)
            split: Data split (train/test/dev)
            
        Returns:
            DataFrame containing the data
        """
        try:
            # Construct the file path based on the directory structure
            file_path = os.path.join(
                self.data_dir,
                language,
                f"cleaned_{language}_{task}_{split}.csv"
            )
            
            logger.info(f"Loading data from {file_path}")
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
                
            df = pd.read_csv(file_path)
            
            # Basic validation
            if 'Text' not in df.columns or 'Label' not in df.columns:
                logger.warning(f"Expected columns not found in {file_path}")
                # Try to infer column names or use defaults
                if len(df.columns) == 2:
                    df.columns = ['Text', 'Label']
                else:
                    logger.error(f"Unexpected column structure in {file_path}")
                    return None
                    
            logger.info(f"Loaded {len(df)} records from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return None
            
    def prepare_dataloaders(self, train_df, test_df, dev_df, label_to_id, batch_size, max_length, use_sampler=True):
        """
        Prepare DataLoader objects for training, testing, and validation.
        
        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame
            dev_df: Validation DataFrame
            label_to_id: Dictionary mapping labels to IDs
            batch_size: Batch size
            max_length: Maximum sequence length
            use_sampler: Whether to use weighted sampler for class imbalance
            
        Returns:
            Dictionary of DataLoader objects
        """
        # Create datasets
        train_dataset = DravidianTextDataset(
            train_df['Text'].tolist(),
            train_df['Label'].tolist(),
            self.tokenizer,
            max_length=max_length,
            label_to_id=label_to_id
        )
        
        test_dataset = DravidianTextDataset(
            test_df['Text'].tolist(),
            test_df['Label'].tolist(),
            self.tokenizer,
            max_length=max_length,
            label_to_id=label_to_id
        )
        
        dev_dataset = DravidianTextDataset(
            dev_df['Text'].tolist(),
            dev_df['Label'].tolist(),
            self.tokenizer,
            max_length=max_length,
            label_to_id=label_to_id
        )
        
        # Create weighted sampler for handling class imbalance
        train_sampler = None
        if use_sampler:
            label_counter = Counter(train_df['Label'].tolist())
            class_weights = {label: 1.0 / count for label, count in label_counter.items()}
            sample_weights = [class_weights[label] for label in train_df['Label'].tolist()]
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_df),
                replacement=True
            )
            logger.info(f"Created weighted sampler with class weights: {class_weights}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler if train_sampler else None,
            shuffle=train_sampler is None,  # Only shuffle if no sampler
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return {
            'train': train_loader,
            'test': test_loader,
            'dev': dev_loader
        }
        
    def compute_class_weights(self, labels):
        """
        Compute class weights inversely proportional to class frequencies.
        
        Args:
            labels: List of labels
            
        Returns:
            Tensor of class weights
        """
        label_counter = Counter(labels)
        num_samples = len(labels)
        num_classes = len(label_counter)
        
        # Compute weights as inverse of frequency
        weights = torch.zeros(num_classes)
        for label, count in label_counter.items():
            label_id = self.label_mappings[self.current_model_key]['label_to_id'][label]
            weights[label_id] = num_samples / (count * num_classes)
            
        logger.info(f"Computed class weights: {weights}")
        return weights.to(self.device)
    
    def train_model(self, model, dataloaders, num_epochs, learning_rate, weight_decay, 
                    class_weights=None, warmup_steps=0, grad_acc_steps=1, max_grad_norm=1.0,
                    early_stopping_patience=3, early_stopping_min_delta=0.005, use_focal_loss=False,
                    focal_loss_gamma=2.0, mixed_precision=True):
        """
        Train the transformer model.
        
        Args:
            model: Model to train
            dataloaders: Dictionary of DataLoader objects
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            class_weights: Weights for handling class imbalance
            warmup_steps: Number of warmup steps for learning rate scheduler
            grad_acc_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            early_stopping_patience: Number of epochs to wait before early stopping
            early_stopping_min_delta: Minimum change to qualify as improvement
            use_focal_loss: Whether to use focal loss instead of cross entropy
            focal_loss_gamma: Gamma parameter for focal loss
            mixed_precision: Whether to use mixed precision training (FP16)
            
        Returns:
            Dictionary with training metrics and best model state
        """
        start_time = time.time()
        
        # Prepare optimizer and scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Calculate total training steps for scheduler
        total_steps = len(dataloaders['train']) * num_epochs // grad_acc_steps
        
        # Create learning rate scheduler with warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Choose loss function
        if use_focal_loss:
            logger.info(f"Using Focal Loss with gamma={focal_loss_gamma}")
            criterion = FocalLoss(alpha=class_weights, gamma=focal_loss_gamma)
        else:
            logger.info("Using Cross Entropy Loss")
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Initialize gradscaler for mixed precision
        scaler = GradScaler() if mixed_precision else None
        
        # For tracking metrics
        train_losses = []
        dev_losses = []
        dev_f1_scores = []
        
        # For early stopping
        best_dev_f1 = 0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
            
            # Training phase
            model.train()
            train_loss = 0
            train_steps = 0
            
            # Use tqdm for progress bar
            train_iterator = tqdm(dataloaders['train'], desc=f"Training Epoch {epoch+1}")
            
            for step, batch in enumerate(train_iterator):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Zero gradients
                if step % grad_acc_steps == 0:
                    optimizer.zero_grad()
                
                # Forward pass with mixed precision if enabled
                if mixed_precision:
                    with autocast():
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss
                        
                        # If we're using our own loss function
                        if use_focal_loss:
                            logits = outputs.logits
                            loss = criterion(logits, labels)
                        
                        # Scale loss for gradient accumulation
                        loss = loss / grad_acc_steps
                        
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    
                    # Update weights if we've accumulated enough gradients
                    if (step + 1) % grad_acc_steps == 0 or (step + 1) == len(dataloaders['train']):
                        # Clip gradients
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(model.parameters(), max_grad_norm)
                        
                        # Update parameters
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                else:
                    # Standard precision training
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    
                    # If we're using our own loss function
                    if use_focal_loss:
                        logits = outputs.logits
                        loss = criterion(logits, labels)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / grad_acc_steps
                    
                    # Backward pass
                    loss.backward()
                    
                    # Update weights if we've accumulated enough gradients
                    if (step + 1) % grad_acc_steps == 0 or (step + 1) == len(dataloaders['train']):
                        # Clip gradients
                        clip_grad_norm_(model.parameters(), max_grad_norm)
                        
                        # Update parameters
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                
                # Track metrics
                train_loss += loss.item() * grad_acc_steps
                train_steps += 1
                
                # Update progress bar
                train_iterator.set_postfix({
                    'loss': train_loss / train_steps,
                    'lr': scheduler.get_last_lr()[0]
                })
            
            # Calculate average training loss for the epoch
            avg_train_loss = train_loss / train_steps
            train_losses.append(avg_train_loss)
            
            logger.info(f"Epoch {epoch+1} - Training Loss: {avg_train_loss:.4f}")
            
            # Validation phase
            dev_metrics = self.evaluate(model, dataloaders['dev'], criterion)
            dev_loss = dev_metrics['loss']
            dev_f1 = dev_metrics['macro_f1']
            
            # Track metrics
            dev_losses.append(dev_loss)
            dev_f1_scores.append(dev_f1)
            
            logger.info(f"Epoch {epoch+1} - Validation Loss: {dev_loss:.4f}, F1: {dev_f1:.4f}")
            
            # Check for improvement
            improvement = dev_f1 - best_dev_f1
            if improvement > early_stopping_min_delta:
                logger.info(f"Validation F1 improved from {best_dev_f1:.4f} to {dev_f1:.4f}")
                best_dev_f1 = dev_f1
                best_model_state = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'dev_f1': dev_f1,
                    'dev_loss': dev_loss
                }
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"No improvement in validation F1. Patience: {patience_counter}/{early_stopping_patience}")
                
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Calculate training time
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Load best model state
        if best_model_state:
            model.load_state_dict(best_model_state['model_state_dict'])
            logger.info(f"Loaded best model from epoch {best_model_state['epoch']}")
        
        return {
            'model': model,
            'best_model_state': best_model_state,
            'train_losses': train_losses,
            'dev_losses': dev_losses,
            'dev_f1_scores': dev_f1_scores,
            'best_dev_f1': best_dev_f1,
            'training_time': training_time
        }
    
    def evaluate(self, model, dataloader, criterion=None):
        """
        Evaluate the model on a dataset.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader for evaluation
            criterion: Loss function (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                logits = outputs.logits
                
                # Calculate loss if criterion is provided
                if criterion:
                    if isinstance(criterion, FocalLoss):
                        loss = criterion(logits, labels)
                    else:
                        loss = outputs.loss
                    total_loss += loss.item()
                
                # Get predictions
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                # Append to lists
                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
        
        # Convert numeric predictions and labels back to original labels
        id_to_label = self.label_mappings[self.current_model_key]['id_to_label']
        pred_labels = [id_to_label[id] for id in all_predictions]
        true_labels = [id_to_label[id] for id in all_labels]
        
        # Calculate metrics
        report = classification_report(true_labels, pred_labels, output_dict=True)
        cm = confusion_matrix(true_labels, pred_labels)
        
        # Calculate average loss
        avg_loss = total_loss / len(dataloader) if criterion else 0
        
        return {
            'loss': avg_loss,
            'report': report,
            'confusion_matrix': cm,
            'predictions': pred_labels,
            'true_labels': true_labels,
            'probabilities': all_probabilities,
            'accuracy': report['accuracy'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score']
        }
    
    def plot_confusion_matrix(self, cm, labels, title, output_path):
        """
        Create and save confusion matrix visualization.
        
        Args:
            cm: Confusion matrix
            labels: Label names
            title: Plot title
            output_path: Path to save the plot
        """
        try:
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(title)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Confusion matrix saved to {output_path}")
        except Exception as e:
            logger.error(f"Error creating confusion matrix: {e}")
    
    def plot_training_curves(self, train_losses, dev_losses, dev_f1_scores, output_path):
        """
        Plot training and validation curves.
        
        Args:
            train_losses: List of training losses
            dev_losses: List of validation losses
            dev_f1_scores: List of validation F1 scores
            output_path: Path to save the plot
        """
        try:
            epochs = range(1, len(train_losses) + 1)
            
            plt.figure(figsize=(12, 10))
            
            # Plot losses
            plt.subplot(2, 1, 1)
            plt.plot(epochs, train_losses, 'b-', label='Training Loss')
            plt.plot(epochs, dev_losses, 'r-', label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            
            # Plot F1 scores
            plt.subplot(2, 1, 2)
            plt.plot(epochs, dev_f1_scores, 'g-', label='Validation F1')
            plt.title('Validation F1 Score')
            plt.xlabel('Epochs')
            plt.ylabel('F1 Score')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Training curves saved to {output_path}")
        except Exception as e:
            logger.error(f"Error creating training curves: {e}")
    
    def plot_per_class_metrics(self, report, labels, title, output_path):
        """
        Plot per-class performance metrics.
        
        Args:
            report: Classification report dictionary
            labels: Label names
            title: Plot title
            output_path: Path to save the plot
        """
        try:
            metrics = ['precision', 'recall', 'f1-score']
            classes = [label for label in labels if label in report]
            
            plt.figure(figsize=(15, 10))
            
            # Create bar plot for each metric
            for i, metric in enumerate(metrics):
                plt.subplot(3, 1, i+1)
                values = [report[cls][metric] for cls in classes]
                bars = plt.bar(classes, values, color='skyblue')
                
                # Add value labels on top of each bar
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                             f'{height:.2f}', ha='center', va='bottom')
                
                plt.title(f'Per-class {metric}')
                plt.ylim(0, 1.1)  # Set y-axis limit
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
            
            plt.suptitle(title)
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Per-class metrics plot saved to {output_path}")
        except Exception as e:
            logger.error(f"Error creating per-class metrics plot: {e}")
    
    def save_model(self, model, path, metadata=None):
        """
        Save model and metadata.
        
        Args:
            model: Model to save
            path: Path to save the model
            metadata: Additional metadata to save
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model state
            torch.save(model.state_dict(), path)
            
            # Save metadata if provided
            if metadata:
                metadata_path = path.replace('.pt', '_metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, path, num_labels):
        """
        Load saved model.
        
        Args:
            path: Path to load the model from
            num_labels: Number of output classes
            
        Returns:
            Loaded model
        """
        try:
            # Create a new model instance
            model = DravidianHOSTransformer(
                self.model_name,
                num_labels=num_labels,
                pooling_type=self.pooling_type
            ).to(self.device)
            
            # Load saved state
            model.load_state_dict(torch.load(path, map_location=self.device))
            
            logger.info(f"Model loaded from {path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def fine_tune_model(self, language, task):
        """
        Fine-tune a model for a specific language and task.
        
        Args:
            language: Target language
            task: Task type (offensive/sentiment)
            
        Returns:
            Dictionary with training results
        """
        try:
            # Create a unique key for this model
            model_key = f"{language}_{task}"
            self.current_model_key = model_key
            
            # Load data for all splits
            train_df = self.load_data(language, task, 'train')
            test_df = self.load_data(language, task, 'test')
            dev_df = self.load_data(language, task, 'dev')
            
            if train_df is None or test_df is None or dev_df is None:
                logger.error(f"Failed to load data for {language}_{task}")
                return None
                
            # Create label mappings
            unique_labels = sorted(set(train_df['Label'].tolist()))
            label_to_id = {label: i for i, label in enumerate(unique_labels)}
            id_to_label = {i: label for label, i in label_to_id.items()}
            
            # Store mappings for later use
            self.label_mappings[model_key] = {
                'label_to_id': label_to_id,
                'id_to_label': id_to_label
            }
            
            logger.info(f"Label mapping for {model_key}: {label_to_id}")
            
            # Determine sequence length based on language
            max_length = self.lang_seq_lengths.get(language, 128)
            logger.info(f"Using sequence length {max_length} for {language}")
            
            # Prepare dataloaders
            dataloaders = self.prepare_dataloaders(
                train_df, test_df, dev_df,
                label_to_id, self.config['batch_size'],
                max_length, use_sampler=True
            )
            
            # Initialize model
            num_labels = len(unique_labels)
            model = DravidianHOSTransformer(
                self.model_name,
                num_labels=num_labels,
                pooling_type=self.pooling_type
            ).to(self.device)
            
            logger.info(f"Initialized model for {model_key} with {num_labels} labels")
            
            # Compute class weights for handling imbalance
            class_weights = self.compute_class_weights(train_df['Label'].tolist())
            
            # Train the model
            training_results = self.train_model(
                model,
                dataloaders,
                num_epochs=self.config['epochs'],
                learning_rate=self.config['learning_rate'],
                weight_decay=self.config['weight_decay'],
                class_weights=class_weights,
                warmup_steps=self.config['warmup_steps'],
                grad_acc_steps=self.config['grad_acc_steps'],
                max_grad_norm=self.config['max_grad_norm'],
                early_stopping_patience=self.config['early_stopping_patience'],
                early_stopping_min_delta=self.config['early_stopping_min_delta'],
                use_focal_loss=self.config['use_focal_loss'],
                focal_loss_gamma=self.config['focal_loss_gamma'],
                mixed_precision=self.config['mixed_precision']
            )
            
            # Store trained model
            self.models[model_key] = training_results['model']
            
            # Evaluate on test set
            logger.info(f"Evaluating {model_key} on test set")
            test_metrics = self.evaluate(model, dataloaders['test'])
            
            # Generate plots
            output_dir = os.path.join(self.output_dir, model_key)
            os.makedirs(output_dir, exist_ok=True)
            
            # Plot confusion matrix
            cm_path = os.path.join(output_dir, 'confusion_matrix.png')
            self.plot_confusion_matrix(
                test_metrics['confusion_matrix'],
                unique_labels,
                f"Confusion Matrix for {language} {task}",
                cm_path
            )
            
            # Plot training curves
            curves_path = os.path.join(output_dir, 'training_curves.png')
            self.plot_training_curves(
                training_results['train_losses'],
                training_results['dev_losses'],
                training_results['dev_f1_scores'],
                curves_path
            )
            
            # Plot per-class metrics
            metrics_path = os.path.join(output_dir, 'per_class_metrics.png')
            self.plot_per_class_metrics(
                test_metrics['report'],
                unique_labels,
                f"Per-class Performance for {language} {task}",
                metrics_path
            )
            
            # Save model
            model_path = os.path.join(output_dir, f"{model_key}_model.pt")
            metadata = {
                'language': language,
                'task': task,
                'f1_score': test_metrics['macro_f1'],
                'accuracy': test_metrics['accuracy'],
                'label_mapping': label_to_id,
                'num_classes': num_labels,
                'model_name': self.model_name,
                'pooling_type': self.pooling_type,
                'config': self.config,
                'max_sequence_length': max_length
            }
            self.save_model(model, model_path, metadata)
            
            # Compile results
            results = {
                'model': model,
                'test_metrics': test_metrics,
                'training_metrics': training_results,
                'label_mapping': label_to_id,
                'model_path': model_path
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in fine-tuning process for {language}_{task}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def predict(self, texts, language, task):
        """
        Make predictions using a trained model.
        
        Args:
            texts: List of texts to predict (can be a single string or a list)
            language: Language of the texts
            task: Task type (offensive/sentiment)
            
        Returns:
            Dictionary with predictions and probabilities
        """
        try:
            # Ensure texts is a list
            if isinstance(texts, str):
                texts = [texts]
                
            model_key = f"{language}_{task}"
            logger.info(f"Attempting prediction for model key: {model_key}")
            
            # Check if model exists
            if model_key not in self.models or model_key not in self.label_mappings:
                logger.info(f"Model {model_key} not loaded yet, loading from disk")
                model_path = os.path.join(self.output_dir, model_key, f"{model_key}_model.pt")
                metadata_path = model_path.replace('.pt', '_metadata.json')
                
                # Check if model files exist
                if not os.path.exists(model_path):
                    logger.error(f"Model file not found: {model_path}")
                    return None
                    
                if not os.path.exists(metadata_path):
                    logger.error(f"Model metadata not found: {metadata_path}")
                    return None
                
                # Load metadata to get num_labels
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    num_labels = metadata['num_classes']
                    self.label_mappings[model_key] = {
                        'label_to_id': metadata['label_mapping'],
                        'id_to_label': {str(v): k for k, v in metadata['label_mapping'].items()}
                    }
                    logger.info(f"Loaded label mapping: {self.label_mappings[model_key]}")
                except Exception as e:
                    logger.error(f"Error loading metadata: {str(e)}")
                    return None
                
                # Load model
                model = self.load_model(model_path, num_labels)
                if model is None:
                    logger.error("load_model returned None")
                    return None
                self.models[model_key] = model
            
            model = self.models[model_key]
            model.eval()
            
            # Set current model key for label mapping access
            self.current_model_key = model_key
            
            # Determine sequence length
            max_length = self.lang_seq_lengths.get(language, 128)
            logger.info(f"Using sequence length: {max_length} for language: {language}")
            
            # Tokenize texts
            logger.info(f"Tokenizing {len(texts)} texts")
            encodings = self.tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Make predictions
            with torch.no_grad():
                logger.info("Running model inference")
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
            
            # Convert to labels
            id_to_label = self.label_mappings[model_key]['id_to_label']
            logger.info(f"ID to label mapping: {id_to_label}")
            
            pred_labels = []
            for id_val in preds:
                str_id = str(id_val.item())
                if str_id in id_to_label:
                    pred_labels.append(id_to_label[str_id])
                else:
                    logger.error(f"Label ID {str_id} not found in mapping")
                    pred_labels.append(f"unknown_{str_id}")
            
            prob_lists = probs.cpu().numpy().tolist()
            
            # Create detailed output
            results = []
            for i, (text, label, prob) in enumerate(zip(texts, pred_labels, prob_lists)):
                # Create mapping of label to probability
                label_probs = {}
                for j in range(len(prob)):
                    str_j = str(j)
                    if str_j in id_to_label:
                        label_probs[id_to_label[str_j]] = prob[j]
                    else:
                        label_probs[f"unknown_{j}"] = prob[j]
                
                results.append({
                    'text': text,
                    'prediction': label,
                    'probabilities': label_probs
                })
            
            logger.info(f"Prediction completed successfully for {len(texts)} texts")
            return results
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def cross_validate(self, language, task, n_folds=5):
        """
        Perform cross-validation for a specific language and task.
        
        Args:
            language: Target language
            task: Task type (offensive/sentiment)
            n_folds: Number of folds for cross-validation
            
        Returns:
            Dictionary with cross-validation results
        """
        try:
            # Create a unique key for this model
            model_key = f"{language}_{task}_cv"
            
            # Load data
            train_df = self.load_data(language, task, 'train')
            dev_df = self.load_data(language, task, 'dev')
            
            if train_df is None or dev_df is None:
                logger.error(f"Failed to load data for {language}_{task} cross-validation")
                return None
            
            # Combine train and dev for cross-validation
            combined_df = pd.concat([train_df, dev_df], ignore_index=True)
            
            # Create label mapping
            unique_labels = sorted(set(combined_df['Label'].tolist()))
            label_to_id = {label: i for i, label in enumerate(unique_labels)}
            id_to_label = {i: label for label, i in label_to_id.items()}
            
            # Split data into folds while preserving class distribution
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            fold_results = []
            fold_models = []
            
            # Determine sequence length based on language
            max_length = self.lang_seq_lengths.get(language, 128)
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(combined_df, combined_df['Label'])):
                logger.info(f"Starting fold {fold+1}/{n_folds}")
                
                # Split data
                fold_train_df = combined_df.iloc[train_idx].reset_index(drop=True)
                fold_val_df = combined_df.iloc[val_idx].reset_index(drop=True)
                
                # Register current model key for label mappings
                self.current_model_key = f"{model_key}_fold{fold+1}"
                self.label_mappings[self.current_model_key] = {
                    'label_to_id': label_to_id,
                    'id_to_label': id_to_label
                }
                
                # Create datasets and dataloaders
                train_dataset = DravidianTextDataset(
                    fold_train_df['Text'].tolist(),
                    fold_train_df['Label'].tolist(),
                    self.tokenizer,
                    max_length=max_length,
                    label_to_id=label_to_id
                )
                
                val_dataset = DravidianTextDataset(
                    fold_val_df['Text'].tolist(),
                    fold_val_df['Label'].tolist(),
                    self.tokenizer,
                    max_length=max_length,
                    label_to_id=label_to_id
                )
                
                # Create weighted sampler for handling class imbalance
                label_counter = Counter(fold_train_df['Label'].tolist())
                class_weights = {label: 1.0 / count for label, count in label_counter.items()}
                sample_weights = [class_weights[label] for label in fold_train_df['Label'].tolist()]
                train_sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(fold_train_df),
                    replacement=True
                )
                
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.config['batch_size'],
                    sampler=train_sampler,
                    num_workers=4,
                    pin_memory=True
                )
                
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.config['batch_size'],
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True
                )
                
                # Create dataloaders dictionary
                dataloaders = {
                    'train': train_loader,
                    'dev': val_loader
                }
                
                # Initialize model
                num_labels = len(unique_labels)
                model = DravidianHOSTransformer(
                    self.model_name,
                    num_labels=num_labels,
                    pooling_type=self.pooling_type
                ).to(self.device)
                
                # Compute class weights
                tensor_class_weights = self.compute_class_weights(fold_train_df['Label'].tolist())
                
                # Train model
                training_results = self.train_model(
                    model,
                    dataloaders,
                    num_epochs=self.config['epochs'],
                    learning_rate=self.config['learning_rate'],
                    weight_decay=self.config['weight_decay'],
                    class_weights=tensor_class_weights,
                    warmup_steps=self.config['warmup_steps'],
                    grad_acc_steps=self.config['grad_acc_steps'],
                    max_grad_norm=self.config['max_grad_norm'],
                    early_stopping_patience=self.config['early_stopping_patience'],
                    early_stopping_min_delta=self.config['early_stopping_min_delta'],
                    use_focal_loss=self.config['use_focal_loss'],
                    focal_loss_gamma=self.config['focal_loss_gamma'],
                    mixed_precision=self.config['mixed_precision']
                )
                
                # Evaluate on validation set
                val_metrics = self.evaluate(model, val_loader)
                
                # Save fold results
                fold_output_dir = os.path.join(self.output_dir, f"{model_key}_fold{fold+1}")
                os.makedirs(fold_output_dir, exist_ok=True)
                
                # Save model
                model_path = os.path.join(fold_output_dir, f"{model_key}_fold{fold+1}_model.pt")
                self.save_model(
                    model, 
                    model_path, 
                    metadata={
                        'fold': fold + 1,
                        'language': language,
                        'task': task,
                        'f1_score': val_metrics['macro_f1']
                    }
                )
                
                # Store results
                fold_results.append({
                    'fold': fold + 1,
                    'val_metrics': val_metrics,
                    'training_metrics': training_results,
                    'model_path': model_path
                })
                
                fold_models.append(model)
                
                logger.info(f"Completed fold {fold+1} with F1 score: {val_metrics['macro_f1']:.4f}")
            
            # Calculate average metrics across folds
            avg_f1 = np.mean([res['val_metrics']['macro_f1'] for res in fold_results])
            avg_accuracy = np.mean([res['val_metrics']['accuracy'] for res in fold_results])
            
            logger.info(f"Cross-validation completed for {language}_{task}")
            logger.info(f"Average F1 score: {avg_f1:.4f}")
            logger.info(f"Average accuracy: {avg_accuracy:.4f}")
            
            # Compile cross-validation results
            cv_results = {
                'fold_results': fold_results,
                'models': fold_models,
                'avg_f1': avg_f1,
                'avg_accuracy': avg_accuracy,
                'language': language,
                'task': task
            }
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation for {language}_{task}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def ensemble_predict(self, texts, language, task, cv_results=None, n_folds=5):
        """
        Make predictions using an ensemble of models from cross-validation.
        
        Args:
            texts: List of texts to predict
            language: Language of the texts
            task: Task type (offensive/sentiment)
            cv_results: Cross-validation results (if already computed)
            n_folds: Number of folds (used if cv_results not provided)
            
        Returns:
            Dictionary with ensemble predictions
        """
        try:
            model_key = f"{language}_{task}_cv"
            
            # Run cross-validation if results not provided
            if cv_results is None:
                cv_results = self.cross_validate(language, task, n_folds)
                if cv_results is None:
                    return None
            
            # Get label mapping
            if f"{model_key}_fold1" in self.label_mappings:
                # All folds should have the same label mapping
                id_to_label = self.label_mappings[f"{model_key}_fold1"]['id_to_label']
                label_to_id = self.label_mappings[f"{model_key}_fold1"]['label_to_id']
            else:
                # Try to load from first fold's metadata
                model_path = cv_results['fold_results'][0]['model_path']
                metadata_path = model_path.replace('.pt', '_metadata.json')
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    label_to_id = metadata.get('label_mapping')
                    id_to_label = {int(id): label for label, id in label_to_id.items()}
                else:
                    logger.error("Label mapping not found for ensemble prediction")
                    return None
            
            num_labels = len(label_to_id)
            
            # Determine sequence length
            max_length = self.lang_seq_lengths.get(language, 128)
            
            # Tokenize texts
            encodings = self.tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Initialize probabilities array
            all_probs = []
            
            # Make predictions with each model in the ensemble
            for fold_idx, fold_result in enumerate(cv_results['fold_results']):
                # Load model if not in memory
                if 'models' in cv_results and cv_results['models'][fold_idx] is not None:
                    model = cv_results['models'][fold_idx]
                else:
                    model_path = fold_result['model_path']
                    model = self.load_model(model_path, num_labels)
                
                if model is None:
                    logger.warning(f"Could not load model for fold {fold_idx+1}")
                    continue
                
                model.eval()
                
                # Make predictions
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    
                    all_probs.append(probs)
            
            if not all_probs:
                logger.error("No valid predictions from ensemble models")
                return None
            
            # Average probabilities across models
            ensemble_probs = np.mean(all_probs, axis=0)
            ensemble_preds = np.argmax(ensemble_probs, axis=1)
            
            # Convert to labels
            pred_labels = [id_to_label[id] for id in ensemble_preds]
            
            # Create detailed output
            results = []
            for i, (text, label, prob) in enumerate(zip(texts, pred_labels, ensemble_probs)):
                # Create mapping of label to probability
                label_probs = {id_to_label[j]: prob[j] for j in range(len(prob))}
                
                results.append({
                    'text': text,
                    'prediction': label,
                    'probabilities': label_probs
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def visualize_attention(self, text, language, task):
        """
        Visualize attention weights for model interpretation.
        
        Args:
            text: Input text
            language: Language of the text
            task: Task type (offensive/sentiment)
            
        Returns:
            Dictionary with attention visualization data
        """
        try:
            model_key = f"{language}_{task}"
            
            # Check if model exists
            if model_key not in self.models:
                model_path = os.path.join(self.output_dir, model_key, f"{model_key}_model.pt")
                metadata_path = model_path.replace('.pt', '_metadata.json')
                
                # Load metadata to get num_labels
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    num_labels = metadata['num_classes']
                    self.label_mappings[model_key] = {
                        'label_to_id': metadata['label_mapping'],
                        'id_to_label': {str(id): label for label, id in metadata['label_mapping'].items()}
                    }
                else:
                    logger.error(f"Model metadata not found for {model_key}")
                    return None
                
                # Load model
                model = self.load_model(model_path, num_labels)
                if model is None:
                    return None
                self.models[model_key] = model
            
            model = self.models[model_key]
            model.eval()
            
            # Set current model key for label mapping access
            self.current_model_key = model_key
            
            # Determine sequence length
            max_length = self.lang_seq_lengths.get(language, 128)
            
            # Tokenize text
            tokens = self.tokenizer.tokenize(text)
            token_ids = self.tokenizer.encode(text, add_special_tokens=True)
            
            # Truncate if too long
            if len(token_ids) > max_length:
                tokens = tokens[:max_length-2]  # Account for special tokens
                token_ids = token_ids[:max_length]
            
            # Create input tensors
            input_ids = torch.tensor([token_ids]).to(self.device)
            attention_mask = torch.ones_like(input_ids).to(self.device)
            
            # Get model outputs with attention
            with torch.no_grad():
                outputs = model.transformer(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True
                )
                
                # Get attention from last layer
                attentions = outputs.attentions[-1].cpu().numpy()
                
                # Get prediction
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_id = np.argmax(probs)
                
                # Convert to label
                id_to_label = self.label_mappings[model_key]['id_to_label']
                pred_label = id_to_label[str(pred_id)]
                
                # Get token strings from ids
                token_strings = [self.tokenizer.convert_ids_to_tokens(id.item()) for id in input_ids[0]]
            
            # Process attention weights (average over heads)
            avg_attention = np.mean(attentions[0], axis=0)
            
            # Create visualization data
            visualization_data = {
                'text': text,
                'tokens': token_strings,
                'prediction': pred_label,
                'probabilities': {id_to_label[str(i)]: float(prob) for i, prob in enumerate(probs)},
                'attention_weights': avg_attention.tolist()
            }
            
            return visualization_data
            
        except Exception as e:
            logger.error(f"Error in attention visualization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def run_all(self):
        """
        Run fine-tuning for all languages and tasks.
        
        Returns:
            Dictionary with all results
        """
        all_results = {}
        
        for language in self.languages:
            for task in self.tasks:
                logger.info(f"Fine-tuning model for {language} {task}")
                key = f"{language}_{task}"
                
                results = self.fine_tune_model(language, task)
                all_results[key] = results
                
                if results:
                    logger.info(f"Successfully fine-tuned model for {language} {task}")
                    logger.info(f"Test F1: {results['test_metrics']['macro_f1']:.4f}")
                    logger.info(f"Test accuracy: {results['test_metrics']['accuracy']:.4f}")
                else:
                    logger.error(f"Failed to fine-tune model for {language} {task}")
        
        # Save overall results
        results_path = os.path.join(self.output_dir, 'all_results.json')
        
        # Extract serializable metrics
        serializable_results = {}
        for key, result in all_results.items():
            if result:
                serializable_results[key] = {
                    'macro_f1': result['test_metrics']['macro_f1'],
                    'weighted_f1': result['test_metrics']['weighted_f1'],
                    'accuracy': result['test_metrics']['accuracy'],
                    'model_path': result['model_path'],
                    'per_class': {
                        label: {
                            'precision': metrics['precision'],
                            'recall': metrics['recall'],
                            'f1-score': metrics['f1-score']
                        }
                        for label, metrics in result['test_metrics']['report'].items()
                        if label not in ['accuracy', 'macro avg', 'weighted avg']
                    }
                }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"All results saved to {results_path}")
        return all_results

def main():
    """Main function to run the Dravidian HOS detector."""
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Dravidian Hate and Offensive Speech Detection')
    
    # Define arguments
    parser.add_argument('--model_name', type=str, default="ai4bharat/IndicBERTv2-MLM-only",
                        help='Pre-trained model name')
    parser.add_argument('--data_dir', type=str, default="./cleaned-datasets",
                        help='Directory containing the data files')
    parser.add_argument('--output_dir', type=str, default="./output",
                        help='Directory to save outputs')
    parser.add_argument('--pooling_type', type=str, default="cls", choices=['cls', 'mean', 'attention'],
                        help='Type of pooling to use')
    parser.add_argument('--mode', type=str, default="train", choices=['train', 'predict', 'cross_validate', 'ensemble', 'visualize'],
                        help='Operation mode')
    parser.add_argument('--language', type=str, default=None, choices=['kannada', 'tamil', 'mal', 'all'],
                        help='Target language')
    parser.add_argument('--task', type=str, default=None, choices=['offensive', 'sentiment', 'all'],
                        help='Task type')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='Use focal loss instead of cross entropy')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--texts', type=str, nargs='+', default=None,
                        help='Texts to predict (for prediction mode)')
    parser.add_argument('--text_file', type=str, default=None,
                        help='Path to file containing texts to predict (one per line)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize detector
    detector = DravidianHOSDetector(
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        pooling_type=args.pooling_type
    )
    
    # Update configuration with command-line arguments
    detector.config.update({
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'use_focal_loss': args.use_focal_loss,
        'mixed_precision': args.mixed_precision
    })
    
    # Process based on mode
    if args.mode == 'train':
        if args.language == 'all' and args.task == 'all':
            # Train all language-task combinations
            results = detector.run_all()
            logger.info("All models trained successfully")
            
            # Print summary of results
            for key, result in results.items():
                if result:
                    logger.info(f"{key} - F1: {result['test_metrics']['macro_f1']:.4f}, "
                                f"Accuracy: {result['test_metrics']['accuracy']:.4f}")
        
        elif args.language == 'all':
            # Train all languages for specific task
            results = {}
            for language in detector.languages:
                logger.info(f"Training {language} for {args.task}")
                result = detector.fine_tune_model(language, args.task)
                results[f"{language}_{args.task}"] = result
                
                if result:
                    logger.info(f"{language}_{args.task} - F1: {result['test_metrics']['macro_f1']:.4f}, "
                                f"Accuracy: {result['test_metrics']['accuracy']:.4f}")
                    
        elif args.task == 'all':
            # Train all tasks for specific language
            results = {}
            for task in detector.tasks:
                logger.info(f"Training {args.language} for {task}")
                result = detector.fine_tune_model(args.language, task)
                results[f"{args.language}_{task}"] = result
                
                if result:
                    logger.info(f"{args.language}_{task} - F1: {result['test_metrics']['macro_f1']:.4f}, "
                                f"Accuracy: {result['test_metrics']['accuracy']:.4f}")
        
        else:
            # Train specific language-task combination
            if not args.language or not args.task:
                logger.error("Both language and task must be specified for training mode")
                return
                
            logger.info(f"Training {args.language} for {args.task}")
            result = detector.fine_tune_model(args.language, args.task)
            
            if result:
                logger.info(f"Training successful - F1: {result['test_metrics']['macro_f1']:.4f}, "
                            f"Accuracy: {result['test_metrics']['accuracy']:.4f}")
    
    elif args.mode == 'predict':
        # Check if language and task are specified
        if not args.language or not args.task:
            logger.error("Both language and task must be specified for prediction mode")
            return
            
        # Get texts to predict
        texts = []
        if args.texts:
            texts.extend(args.texts)
        
        if args.text_file:
            try:
                with open(args.text_file, 'r', encoding='utf-8') as f:
                    file_texts = [line.strip() for line in f if line.strip()]
                    texts.extend(file_texts)
            except Exception as e:
                logger.error(f"Error reading text file: {e}")
                
        if not texts:
            logger.error("No texts provided for prediction")
            return
            
        # Make predictions
        predictions = detector.predict(texts, args.language, args.task)
        
        if predictions:
            logger.info(f"Predictions for {len(texts)} texts:")
            for i, pred in enumerate(predictions):
                try:
                    # Truncate text for display while preserving Unicode characters
                    text = pred['text']
                    display_text = text[:50] + '...' if len(text) > 50 else text
                    
                    # Log predictions using Unicode-safe formatting
                    logger.info(f"Prediction {i+1}:")
                    logger.info(f"Text: {display_text}")
                    logger.info(f"Predicted Label: {pred['prediction']}")
                    
                    # Format probabilities for better readability
                    prob_str = '\n'.join(f"  {label}: {prob:.4f}" 
                                       for label, prob in pred['probabilities'].items())
                    logger.info(f"Probabilities:\n{prob_str}")
                    logger.info("-" * 50)
                except Exception as e:
                    logger.error(f"Error displaying prediction {i+1}: {str(e)}")
                    continue
    
    elif args.mode == 'cross_validate':
        # Check if language and task are specified
        if not args.language or not args.task:
            logger.error("Both language and task must be specified for cross-validation mode")
            return
            
        logger.info(f"Running {args.n_folds}-fold cross-validation for {args.language} {args.task}")
        cv_results = detector.cross_validate(args.language, args.task, n_folds=args.n_folds)
        
        if cv_results:
            logger.info(f"Cross-validation complete")
            logger.info(f"Average F1 score: {cv_results['avg_f1']:.4f}")
            logger.info(f"Average accuracy: {cv_results['avg_accuracy']:.4f}")
            
            # Print per-fold results
            for fold_result in cv_results['fold_results']:
                fold = fold_result['fold']
                f1 = fold_result['val_metrics']['macro_f1']
                acc = fold_result['val_metrics']['accuracy']
                logger.info(f"Fold {fold}: F1 = {f1:.4f}, Accuracy = {acc:.4f}")
    
    elif args.mode == 'ensemble':
        # Check if language and task are specified
        if not args.language or not args.task:
            logger.error("Both language and task must be specified for ensemble mode")
            return
            
        # Get texts to predict
        texts = []
        if args.texts:
            texts.extend(args.texts)
        
        if args.text_file:
            try:
                with open(args.text_file, 'r', encoding='utf-8') as f:
                    file_texts = [line.strip() for line in f if line.strip()]
                    texts.extend(file_texts)
            except Exception as e:
                logger.error(f"Error reading text file: {e}")
                
        if not texts:
            logger.error("No texts provided for ensemble prediction")
            return
            
        logger.info(f"Running ensemble prediction with {args.n_folds} models")
        # First run cross-validation if not already done
        cv_results = detector.cross_validate(args.language, args.task, n_folds=args.n_folds)
        
        if cv_results:
            # Make ensemble predictions
            predictions = detector.ensemble_predict(texts, args.language, args.task, cv_results, args.n_folds)
            
            if predictions:
                logger.info(f"Ensemble predictions for {len(texts)} texts:")
                for i, pred in enumerate(predictions):
                    # Truncate text for display
                    display_text = pred['text'][:50] + '...' if len(pred['text']) > 50 else pred['text']
                    logger.info(f"{i+1}. Text: {display_text}")
                    logger.info(f"   Prediction: {pred['prediction']}")
                    logger.info(f"   Probabilities: {pred['probabilities']}")
                    logger.info("---")
    
    elif args.mode == 'visualize':
        # Check if language and task are specified
        if not args.language or not args.task:
            logger.error("Both language and task must be specified for visualization mode")
            return
            
        # Get text to visualize (use only the first one if multiple are provided)
        text = None
        if args.texts and len(args.texts) > 0:
            text = args.texts[0]
        
        if not text and args.text_file:
            try:
                with open(args.text_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        text = lines[0].strip()
            except Exception as e:
                logger.error(f"Error reading text file: {e}")
                
        if not text:
            logger.error("No text provided for attention visualization")
            return
            
        logger.info(f"Visualizing attention for text: {text[:50]}...")
        viz_data = detector.visualize_attention(text, args.language, args.task)
        
        if viz_data:
            logger.info(f"Prediction: {viz_data['prediction']}")
            logger.info(f"Probabilities: {viz_data['probabilities']}")
            
            # Create visualization file
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Filter out padding tokens
            tokens = viz_data['tokens']
            weights = viz_data['attention_weights']
            
            # Create heatmap of token self-attention
            plt.figure(figsize=(12, 10))
            sns.heatmap(weights, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
            plt.title(f"Self-attention for '{viz_data['prediction']}' prediction")
            plt.tight_layout()
            
            # Save visualization
            viz_path = os.path.join(detector.output_dir, f"{args.language}_{args.task}_attention.png")
            plt.savefig(viz_path)
            logger.info(f"Visualization saved to {viz_path}")
            
            # Also show token-level attention (simplified)
            token_attention = [weights[0][i] for i in range(len(tokens))]
            plt.figure(figsize=(15, 5))
            plt.bar(range(len(tokens)), token_attention)
            plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
            plt.title("Token-level attention")
            plt.tight_layout()
            
            # Save token attention visualization
            token_viz_path = os.path.join(detector.output_dir, f"{args.language}_{args.task}_token_attention.png")
            plt.savefig(token_viz_path)
            logger.info(f"Token attention visualization saved to {token_viz_path}")
    
    else:
        logger.error(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()