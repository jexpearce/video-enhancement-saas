"""
Linguistic Analysis for Emphasis Detection

This module analyzes linguistic features to identify words and phrases
that are semantically important and likely to be emphasized based on
their role in the sentence structure and meaning.
"""

import spacy
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class LinguisticAnalyzer:
    """
    Analyzes linguistic features to predict word emphasis based on semantic importance.
    
    Uses:
    - Part-of-speech analysis
    - Named entity recognition
    - Syntactic dependency parsing
    - Semantic role labeling
    - Keyword importance scoring
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.error(f"spaCy model '{model_name}' not found. Please install it with: python -m spacy download {model_name}")
            raise
        
        # Define emphasis-prone POS tags (optimized for image-findable content)
        self.emphasis_pos = {
            'PROPN': 0.95,    # Proper nouns (Iran, Netanyahu, Apple) - HIGHEST for images
            'NOUN': 0.85,     # Concrete nouns - HIGH for images
            'NUM': 0.75,      # Numbers (when significant) - MEDIUM
            'VERB': 0.45,     # Verbs - LOWER (harder to visualize)
            'ADJ': 0.25,      # Adjectives (amazing, incredible) - LOW (abstract)
            'ADV': 0.15,      # Adverbs (absolutely) - LOWEST (abstract)
            'INTJ': 0.20,     # Interjections - LOW (not image-relevant)
        }
        
        # Define emphasis-prone dependency relations
        self.emphasis_deps = {
            'ROOT': 0.8,      # Root verb is often emphasized
            'nsubj': 0.7,     # Subject
            'dobj': 0.6,      # Direct object
            'pobj': 0.5,      # Object of preposition
            'amod': 0.6,      # Adjectival modifier
            'advmod': 0.5,    # Adverbial modifier
            'compound': 0.7,  # Compound words
        }
        
        # Stop words that are rarely emphasized
        self.stop_words = set(self.nlp.Defaults.stop_words)
        
    def analyze_linguistic_emphasis(
        self, 
        text: str, 
        word_timestamps: List[Dict]
    ) -> List[Dict]:
        """
        Analyze linguistic features to predict emphasis.
        
        Args:
            text: Full transcribed text
            word_timestamps: List of word timing information
            
        Returns:
            List of linguistic analysis results for each word
        """
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Extract linguistic features
            linguistic_features = self._extract_linguistic_features(doc)
            
            # Analyze each word
            results = []
            for word_info in word_timestamps:
                word_analysis = self._analyze_word_linguistics(
                    word_info, doc, linguistic_features
                )
                results.append(word_analysis)
                
            return results
            
        except Exception as e:
            logger.error(f"Error in linguistic analysis: {e}")
            raise
    
    def _extract_linguistic_features(self, doc) -> Dict:
        """Extract comprehensive linguistic features from the document."""
        features = {}
        
        # 1. Named Entity Recognition
        features['entities'] = self._extract_entities(doc)
        
        # 2. Keyword frequency and TF-IDF scores
        features['keyword_scores'] = self._calculate_keyword_scores(doc)
        
        # 3. Syntactic features
        features['syntax'] = self._extract_syntactic_features(doc)
        
        # 4. Semantic roles
        features['semantic_roles'] = self._extract_semantic_roles(doc)
        
        # 5. Discourse markers
        features['discourse_markers'] = self._identify_discourse_markers(doc)
        
        # 6. Content vs. function words
        features['content_words'] = self._identify_content_words(doc)
        
        return features
    
    def _extract_entities(self, doc) -> Dict:
        """Extract named entities and their importance."""
        entities = {}
        entity_labels = set()
        
        for ent in doc.ents:
            entities[ent.text.lower()] = {
                'label': ent.label_,
                'start_char': ent.start_char,
                'end_char': ent.end_char,
                'importance': self._get_entity_importance(ent.label_)
            }
            entity_labels.add(ent.label_)
        
        return {
            'entities': entities,
            'entity_types': entity_labels,
            'entity_count': len(entities)
        }
    
    def _get_entity_importance(self, label: str) -> float:
        """Get importance score for different entity types (optimized for image retrieval)."""
        importance_map = {
            'PERSON': 0.95,     # Person names (Netanyahu, Biden) - PERFECT for images
            'ORG': 0.90,        # Organizations (Apple, Google) - EXCELLENT for images
            'GPE': 0.90,        # Geopolitical entities (Iran, Paris) - EXCELLENT for images
            'PRODUCT': 0.85,    # Products (iPhone, Tesla) - GREAT for images
            'EVENT': 0.80,      # Events (Olympics, war) - GOOD for images
            'MONEY': 0.75,      # Money amounts ($100M) - MEDIUM for images
            'PERCENT': 0.70,    # Percentages (50%) - MEDIUM for images
            'WORK_OF_ART': 0.80, # Works of art - GOOD for images
            'LAW': 0.60,        # Laws - HARDER for images
            'DATE': 0.40,       # Dates - LOW for images
            'TIME': 0.30,       # Times - LOW for images
            'QUANTITY': 0.50,   # Quantities - MEDIUM for images
        }
        return importance_map.get(label, 0.3)
    
    def _calculate_keyword_scores(self, doc) -> Dict:
        """Calculate keyword importance scores using frequency and position."""
        # Word frequency
        word_freq = Counter()
        total_words = 0
        
        for token in doc:
            if not token.is_stop and not token.is_punct and not token.is_space:
                word_freq[token.lemma_.lower()] += 1
                total_words += 1
        
        # Calculate TF scores
        tf_scores = {}
        for word, freq in word_freq.items():
            tf_scores[word] = freq / total_words if total_words > 0 else 0
        
        # Calculate position-based importance (earlier words often more important)
        position_scores = {}
        for i, token in enumerate(doc):
            if not token.is_stop and not token.is_punct and not token.is_space:
                # Linear decay from beginning of text
                position_score = max(0.1, 1.0 - (i / len(doc)) * 0.5)
                position_scores[token.lemma_.lower()] = position_score
        
        # Combine TF and position scores
        keyword_scores = {}
        for word in set(list(tf_scores.keys()) + list(position_scores.keys())):
            tf_score = tf_scores.get(word, 0)
            pos_score = position_scores.get(word, 0.1)
            keyword_scores[word] = (tf_score * 0.7) + (pos_score * 0.3)
        
        return keyword_scores
    
    def _extract_syntactic_features(self, doc) -> Dict:
        """Extract syntactic dependency features."""
        syntax_features = {}
        
        # Dependency relations count
        dep_counts = Counter()
        for token in doc:
            dep_counts[token.dep_] += 1
        
        syntax_features['dependency_counts'] = dict(dep_counts)
        
        # Find syntactic heads and their importance
        head_importance = {}
        for token in doc:
            if token.dep_ == 'ROOT':
                head_importance[token.lemma_.lower()] = 1.0
            elif token.head != token:
                # Calculate importance based on distance from root
                distance_to_root = self._distance_to_root(token)
                importance = max(0.1, 1.0 - (distance_to_root * 0.2))
                head_importance[token.lemma_.lower()] = importance
        
        syntax_features['head_importance'] = head_importance
        
        return syntax_features
    
    def _distance_to_root(self, token) -> int:
        """Calculate distance from token to syntactic root."""
        distance = 0
        current = token
        
        while current.head != current and distance < 10:  # Prevent infinite loops
            current = current.head
            distance += 1
        
        return distance
    
    def _extract_semantic_roles(self, doc) -> Dict:
        """Extract semantic role information."""
        semantic_roles = {}
        
        for token in doc:
            # Identify semantic roles based on dependency relations
            role_score = 0.0
            
            if token.dep_ in ['nsubj', 'nsubjpass']:
                role_score = 0.8  # Subject roles are important
            elif token.dep_ in ['dobj', 'iobj']:
                role_score = 0.7  # Object roles
            elif token.dep_ in ['agent']:
                role_score = 0.9  # Agent in passive constructions
            elif token.dep_ in ['attr', 'acomp']:
                role_score = 0.6  # Predicate complements
            
            if role_score > 0:
                semantic_roles[token.lemma_.lower()] = role_score
        
        return semantic_roles
    
    def _identify_discourse_markers(self, doc) -> Set[str]:
        """Identify discourse markers that signal emphasis."""
        discourse_markers = {
            'however', 'therefore', 'moreover', 'furthermore', 'nevertheless',
            'consequently', 'specifically', 'particularly', 'especially',
            'importantly', 'significantly', 'notably', 'remarkably',
            'surprisingly', 'interestingly', 'obviously', 'clearly',
            'certainly', 'definitely', 'absolutely', 'indeed', 'actually',
            'really', 'truly', 'literally', 'exactly', 'precisely'
        }
        
        found_markers = set()
        for token in doc:
            if token.lemma_.lower() in discourse_markers:
                found_markers.add(token.lemma_.lower())
        
        return found_markers
    
    def _identify_content_words(self, doc) -> Set[str]:
        """Identify content words vs. function words."""
        content_pos = {'NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV', 'NUM'}
        content_words = set()
        
        for token in doc:
            if (token.pos_ in content_pos and 
                not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 2):
                content_words.add(token.lemma_.lower())
        
        return content_words
    
    def _analyze_word_linguistics(
        self, 
        word_info: Dict, 
        doc, 
        features: Dict
    ) -> Dict:
        """Analyze linguistic emphasis for a specific word."""
        word_text = word_info['word'].lower().strip()
        word_clean = word_text.replace("'s", "").replace("'", "")
        
        # Find corresponding token in spaCy doc
        spacy_token = self._find_token_in_doc(word_clean, doc)
        
        # Calculate linguistic scores
        linguistic_scores = {}
        
        # 1. Part-of-speech emphasis score
        if spacy_token:
            pos_score = self.emphasis_pos.get(spacy_token.pos_, 0.1)
            linguistic_scores['pos_score'] = pos_score
        else:
            linguistic_scores['pos_score'] = 0.1
        
        # 2. Dependency relation emphasis score
        if spacy_token:
            dep_score = self.emphasis_deps.get(spacy_token.dep_, 0.1)
            linguistic_scores['dependency_score'] = dep_score
        else:
            linguistic_scores['dependency_score'] = 0.1
        
        # 3. Named entity emphasis score
        entity_score = 0.0
        for entity_text, entity_info in features['entities']['entities'].items():
            if word_clean in entity_text or entity_text in word_clean:
                entity_score = entity_info['importance']
                break
        linguistic_scores['entity_score'] = entity_score
        
        # 4. Keyword importance score
        keyword_score = features['keyword_scores'].get(word_clean, 0.0)
        linguistic_scores['keyword_score'] = keyword_score
        
        # 5. Semantic role score
        semantic_score = features['semantic_roles'].get(word_clean, 0.0)
        linguistic_scores['semantic_role_score'] = semantic_score
        
        # 6. Discourse marker bonus
        discourse_bonus = 0.8 if word_clean in features['discourse_markers'] else 0.0
        linguistic_scores['discourse_score'] = discourse_bonus
        
        # 7. Content word bonus
        content_bonus = 0.3 if word_clean in features['content_words'] else 0.0
        linguistic_scores['content_word_score'] = content_bonus
        
        # 8. Stop word penalty
        stop_penalty = -0.3 if word_clean in self.stop_words else 0.0
        linguistic_scores['stop_word_penalty'] = stop_penalty
        
        # 9. Word length and complexity bonus
        length_bonus = min(0.2, len(word_clean) / 20) if len(word_clean) > 4 else 0.0
        linguistic_scores['length_bonus'] = length_bonus
        
        # 10. Syntactic importance based on distance to root
        if spacy_token:
            distance_to_root = self._distance_to_root(spacy_token)
            syntax_importance = max(0.1, 1.0 - (distance_to_root * 0.15))
            linguistic_scores['syntax_importance'] = syntax_importance
        else:
            linguistic_scores['syntax_importance'] = 0.1
        
        # Calculate combined linguistic score
        weights = {
            'pos_score': 0.15,
            'dependency_score': 0.15,
            'entity_score': 0.20,
            'keyword_score': 0.15,
            'semantic_role_score': 0.10,
            'discourse_score': 0.10,
            'content_word_score': 0.05,
            'stop_word_penalty': 0.05,
            'length_bonus': 0.03,
            'syntax_importance': 0.12
        }
        
        total_linguistic_score = sum(
            linguistic_scores.get(key, 0) * weight 
            for key, weight in weights.items()
        )
        
        # Normalize to 0-1 range
        total_linguistic_score = max(0.0, min(1.0, total_linguistic_score))
        linguistic_scores['total_linguistic_score'] = total_linguistic_score
        
        return {
            'word': word_info['word'],
            'start': word_info['start'],
            'end': word_info['end'],
            'linguistic_scores': linguistic_scores,
            'spacy_features': {
                'pos': spacy_token.pos_ if spacy_token else None,
                'dep': spacy_token.dep_ if spacy_token else None,
                'lemma': spacy_token.lemma_ if spacy_token else None,
                'is_stop': spacy_token.is_stop if spacy_token else False,
                'is_alpha': spacy_token.is_alpha if spacy_token else False,
            },
            'is_linguistically_emphasized': total_linguistic_score > 0.6
        }
    
    def _find_token_in_doc(self, word: str, doc):
        """Find the corresponding spaCy token for a word."""
        # Try exact match first
        for token in doc:
            if token.text.lower() == word:
                return token
        
        # Try lemma match
        for token in doc:
            if token.lemma_.lower() == word:
                return token
        
        # Try substring match for contractions or partial words
        for token in doc:
            if word in token.text.lower() or token.text.lower() in word:
                return token
        
        return None 