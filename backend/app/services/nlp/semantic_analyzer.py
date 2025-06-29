"""
Semantic Analyzer for Advanced Context Understanding

Provides semantic analysis capabilities:
- Context-aware entity importance scoring
- Semantic similarity and relationships
- Topic modeling and categorization  
- Sentiment and tone analysis
- Temporal and spatial context understanding
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import spacy
from collections import Counter, defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SemanticContext:
    """Semantic context information for text analysis."""
    topics: List[str] = field(default_factory=list)           # Main topics
    sentiment: str = "neutral"                                # Overall sentiment
    sentiment_score: float = 0.0                             # Sentiment strength (-1 to 1)
    tone: str = "informational"                              # Communication tone
    formality: float = 0.5                                   # Formality level (0-1)
    urgency: float = 0.3                                     # Urgency level (0-1)
    temporal_context: Dict[str, float] = field(default_factory=dict)  # Time references
    spatial_context: Dict[str, float] = field(default_factory=dict)   # Location references

@dataclass
class EntityRelationship:
    """Relationship between entities."""
    entity1: str
    entity2: str
    relationship_type: str                                   # Type of relationship
    strength: float                                          # Relationship strength (0-1)
    context: str                                            # Context where relationship appears
    directional: bool = False                               # Whether relationship is directional

@dataclass
class SemanticAnalysis:
    """Complete semantic analysis results."""
    text: str
    semantic_context: SemanticContext
    entity_relationships: List[EntityRelationship] = field(default_factory=list)
    key_concepts: List[str] = field(default_factory=list)
    discourse_markers: List[str] = field(default_factory=list)
    emphasis_indicators: List[str] = field(default_factory=list)
    coherence_score: float = 0.0                           # Text coherence (0-1)
    complexity_score: float = 0.0                          # Text complexity (0-1)

class SemanticAnalyzer:
    """Advanced semantic analysis system."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the semantic analyzer."""
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.error(f"Failed to load spaCy model: {model_name}")
            raise
        
        # Initialize analysis components
        self._init_sentiment_lexicon()
        self._init_topic_keywords()
        self._init_relationship_patterns()
        self._init_discourse_markers()
        
    def _init_sentiment_lexicon(self):
        """Initialize sentiment analysis lexicon."""
        
        # Positive sentiment words
        self.positive_words = {
            'amazing', 'excellent', 'great', 'wonderful', 'fantastic', 
            'outstanding', 'brilliant', 'superb', 'incredible', 'awesome',
            'perfect', 'love', 'best', 'impressive', 'remarkable'
        }
        
        # Negative sentiment words  
        self.negative_words = {
            'terrible', 'awful', 'horrible', 'bad', 'worse', 'worst',
            'disgusting', 'disappointing', 'failed', 'disaster', 'crisis',
            'problem', 'issue', 'concern', 'worried', 'afraid'
        }
        
        # Neutral/factual words
        self.neutral_words = {
            'said', 'reported', 'announced', 'stated', 'mentioned',
            'according', 'shows', 'indicates', 'suggests', 'reveals'
        }
        
    def _init_topic_keywords(self):
        """Initialize topic categorization keywords."""
        
        self.topic_keywords = {
            'politics': {
                'government', 'president', 'election', 'policy', 'congress',
                'senate', 'vote', 'campaign', 'political', 'democracy',
                'republican', 'democrat', 'partisan', 'administration'
            },
            'technology': {
                'ai', 'artificial intelligence', 'machine learning', 'tech',
                'software', 'computer', 'digital', 'internet', 'data',
                'algorithm', 'innovation', 'startup', 'silicon valley'
            },
            'business': {
                'company', 'corporate', 'business', 'market', 'stock',
                'investment', 'economy', 'financial', 'profit', 'revenue',
                'merger', 'acquisition', 'ceo', 'earnings', 'quarterly'
            },
            'health': {
                'health', 'medical', 'doctor', 'hospital', 'treatment',
                'vaccine', 'disease', 'virus', 'pandemic', 'medicine',
                'research', 'clinical', 'study', 'patient', 'therapy'
            },
            'sports': {
                'game', 'team', 'player', 'coach', 'season', 'championship',
                'tournament', 'athlete', 'training', 'competition', 'league',
                'score', 'win', 'defeat', 'match'
            },
            'entertainment': {
                'movie', 'film', 'actor', 'actress', 'director', 'music',
                'artist', 'album', 'concert', 'show', 'tv', 'streaming',
                'netflix', 'hollywood', 'celebrity'
            }
        }
        
    def _init_relationship_patterns(self):
        """Initialize entity relationship patterns."""
        
        # Relationship indicator patterns
        self.relationship_patterns = {
            'ownership': [
                r'(\w+)(?:\s+(?:owns|possesses|has))\s+(\w+)',
                r'(\w+)(?:\'s|s\')\s+(\w+)',
                r'(\w+)\s+(?:of|from)\s+(\w+)'
            ],
            'leadership': [
                r'(\w+)\s+(?:ceo|president|director|head|leader)\s+(?:of\s+)?(\w+)',
                r'(\w+)\s+(?:leads|runs|manages)\s+(\w+)',
                r'(\w+)\s+(?:founded|created|established)\s+(\w+)'
            ],
            'location': [
                r'(\w+)\s+(?:in|at|from)\s+(\w+)',
                r'(\w+)\s+(?:based|located|situated)\s+(?:in\s+)?(\w+)',
                r'(\w+)\s+(?:headquarters|office)\s+(?:in\s+)?(\w+)'
            ],
            'collaboration': [
                r'(\w+)\s+(?:and|with|alongside)\s+(\w+)',
                r'(\w+)\s+(?:partners|works|collaborates)\s+(?:with\s+)?(\w+)',
                r'(\w+)\s+(?:joint|together)\s+(?:with\s+)?(\w+)'
            ]
        }
        
    def _init_discourse_markers(self):
        """Initialize discourse markers for emphasis detection."""
        
        self.discourse_markers = {
            'emphasis': {
                'importantly', 'significantly', 'notably', 'remarkably',
                'especially', 'particularly', 'specifically', 'crucially',
                'above all', 'most importantly', 'key point', 'highlight'
            },
            'contrast': {
                'however', 'but', 'nevertheless', 'nonetheless', 'although',
                'despite', 'in contrast', 'on the other hand', 'whereas',
                'while', 'yet', 'still'
            },
            'addition': {
                'furthermore', 'moreover', 'additionally', 'also', 'plus',
                'in addition', 'besides', 'as well', 'similarly', 'likewise'
            },
            'conclusion': {
                'therefore', 'thus', 'consequently', 'as a result',
                'in conclusion', 'finally', 'ultimately', 'overall',
                'in summary', 'to sum up'
            }
        }
        
    def analyze_semantics(
        self, 
        text: str, 
        entities: List = None,
        emphasized_words: List[str] = None
    ) -> SemanticAnalysis:
        """
        Perform comprehensive semantic analysis of text.
        
        Args:
            text: Input text to analyze
            entities: List of recognized entities
            emphasized_words: List of emphasized words
            
        Returns:
            Complete semantic analysis results
        """
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Analyze semantic context
            semantic_context = self._analyze_context(doc, emphasized_words)
            
            # Find entity relationships
            relationships = self._extract_relationships(doc, entities)
            
            # Extract key concepts
            key_concepts = self._extract_key_concepts(doc)
            
            # Identify discourse markers
            discourse_markers = self._identify_discourse_markers(doc)
            
            # Find emphasis indicators
            emphasis_indicators = self._find_emphasis_indicators(doc, emphasized_words)
            
            # Calculate coherence and complexity
            coherence_score = self._calculate_coherence(doc)
            complexity_score = self._calculate_complexity(doc)
            
            return SemanticAnalysis(
                text=text,
                semantic_context=semantic_context,
                entity_relationships=relationships,
                key_concepts=key_concepts,
                discourse_markers=discourse_markers,
                emphasis_indicators=emphasis_indicators,
                coherence_score=coherence_score,
                complexity_score=complexity_score
            )
            
        except Exception as e:
            logger.error(f"Error in semantic analysis: {e}")
            raise
            
    def _analyze_context(self, doc, emphasized_words: List[str] = None) -> SemanticContext:
        """Analyze overall semantic context of the text."""
        
        # Topic classification
        topics = self._classify_topics(doc)
        
        # Sentiment analysis
        sentiment, sentiment_score = self._analyze_sentiment(doc)
        
        # Tone analysis
        tone = self._analyze_tone(doc)
        
        # Formality and urgency
        formality = self._calculate_formality(doc)
        urgency = self._calculate_urgency(doc, emphasized_words)
        
        # Temporal and spatial context
        temporal_context = self._extract_temporal_context(doc)
        spatial_context = self._extract_spatial_context(doc)
        
        return SemanticContext(
            topics=topics,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            tone=tone,
            formality=formality,
            urgency=urgency,
            temporal_context=temporal_context,
            spatial_context=spatial_context
        )
        
    def _classify_topics(self, doc) -> List[str]:
        """Classify main topics in the text."""
        
        topic_scores = defaultdict(float)
        
        # Count topic-related words
        for token in doc:
            if not token.is_stop and not token.is_punct:
                word = token.lemma_.lower()
                
                for topic, keywords in self.topic_keywords.items():
                    if word in keywords:
                        topic_scores[topic] += 1.0
                    # Check for partial matches
                    elif any(keyword in word or word in keyword for keyword in keywords):
                        topic_scores[topic] += 0.5
        
        # Normalize scores and get top topics
        if topic_scores:
            max_score = max(topic_scores.values())
            if max_score > 0:
                topic_scores = {
                    topic: score / max_score 
                    for topic, score in topic_scores.items()
                }
                
                # Return topics with score > 0.3
                return [
                    topic for topic, score in topic_scores.items() 
                    if score > 0.3
                ]
        
        return ['general']
        
    def _analyze_sentiment(self, doc) -> Tuple[str, float]:
        """Analyze sentiment of the text."""
        
        positive_count = 0
        negative_count = 0
        total_words = 0
        
        for token in doc:
            if not token.is_stop and not token.is_punct:
                word = token.lemma_.lower()
                
                if word in self.positive_words:
                    positive_count += 1
                elif word in self.negative_words:
                    negative_count += 1
                    
                total_words += 1
        
        if total_words == 0:
            return "neutral", 0.0
            
        # Calculate sentiment score (-1 to 1)
        sentiment_score = (positive_count - negative_count) / total_words
        
        # Classify sentiment
        if sentiment_score > 0.1:
            sentiment = "positive"
        elif sentiment_score < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        return sentiment, sentiment_score
        
    def _analyze_tone(self, doc) -> str:
        """Analyze communication tone."""
        
        # Count different tone indicators
        formal_indicators = {'according', 'furthermore', 'however', 'therefore'}
        casual_indicators = {'really', 'pretty', 'quite', 'sort of', 'kind of'}
        urgent_indicators = {'urgent', 'immediately', 'asap', 'crisis', 'emergency'}
        questioning_indicators = {'?', 'what', 'how', 'why', 'when', 'where'}
        
        formal_count = sum(1 for token in doc if token.lemma_.lower() in formal_indicators)
        casual_count = sum(1 for token in doc if token.lemma_.lower() in casual_indicators)
        urgent_count = sum(1 for token in doc if token.lemma_.lower() in urgent_indicators)
        question_count = sum(1 for token in doc if token.text in questioning_indicators)
        
        # Determine tone
        if urgent_count > 0:
            return "urgent"
        elif question_count > len(doc) * 0.1:
            return "questioning"  
        elif formal_count > casual_count:
            return "formal"
        elif casual_count > 0:
            return "casual"
        else:
            return "informational"
            
    def _calculate_formality(self, doc) -> float:
        """Calculate formality level (0-1)."""
        
        formal_features = 0
        total_features = 0
        
        for token in doc:
            if not token.is_punct and not token.is_space:
                total_features += 1
                
                # Formal indicators
                if len(token.text) > 6:  # Longer words tend to be more formal
                    formal_features += 0.5
                if token.pos_ in ['NOUN', 'ADJ'] and len(token.text) > 4:
                    formal_features += 0.3
                if token.lemma_.lower() in {'according', 'furthermore', 'however', 'therefore'}:
                    formal_features += 1.0
                    
        return min(1.0, formal_features / total_features) if total_features > 0 else 0.5
        
    def _calculate_urgency(self, doc, emphasized_words: List[str] = None) -> float:
        """Calculate urgency level (0-1)."""
        
        urgency_score = 0.0
        
        # Urgent words
        urgent_words = {'urgent', 'immediate', 'asap', 'emergency', 'crisis', 'breaking'}
        for token in doc:
            if token.lemma_.lower() in urgent_words:
                urgency_score += 0.3
                
        # Emphasis indicators
        if emphasized_words:
            urgency_score += len(emphasized_words) * 0.1
            
        # Exclamation marks and capitalization
        exclamations = sum(1 for token in doc if '!' in token.text)
        urgency_score += exclamations * 0.2
        
        # ALL CAPS words
        caps_words = sum(1 for token in doc if token.text.isupper() and len(token.text) > 2)
        urgency_score += caps_words * 0.15
        
        return min(1.0, urgency_score)
        
    def _extract_temporal_context(self, doc) -> Dict[str, float]:
        """Extract temporal context information."""
        
        temporal_context = {}
        
        # Time entities
        time_entities = [ent for ent in doc.ents if ent.label_ in ['DATE', 'TIME']]
        if time_entities:
            temporal_context['has_time_references'] = 1.0
            temporal_context['time_entity_count'] = len(time_entities)
        
        # Temporal keywords
        past_words = {'yesterday', 'ago', 'previously', 'former', 'past'}
        present_words = {'now', 'currently', 'today', 'present', 'ongoing'}
        future_words = {'tomorrow', 'future', 'upcoming', 'planned', 'will'}
        
        past_count = sum(1 for token in doc if token.lemma_.lower() in past_words)
        present_count = sum(1 for token in doc if token.lemma_.lower() in present_words)
        future_count = sum(1 for token in doc if token.lemma_.lower() in future_words)
        
        total_temporal = past_count + present_count + future_count
        if total_temporal > 0:
            temporal_context['past_orientation'] = past_count / total_temporal
            temporal_context['present_orientation'] = present_count / total_temporal
            temporal_context['future_orientation'] = future_count / total_temporal
            
        return temporal_context
        
    def _extract_spatial_context(self, doc) -> Dict[str, float]:
        """Extract spatial context information."""
        
        spatial_context = {}
        
        # Location entities
        location_entities = [ent for ent in doc.ents if ent.label_ in ['GPE', 'LOC', 'FAC']]
        if location_entities:
            spatial_context['has_location_references'] = 1.0
            spatial_context['location_entity_count'] = len(location_entities)
        
        # Spatial keywords
        spatial_words = {'here', 'there', 'where', 'location', 'place', 'region', 'area'}
        spatial_count = sum(1 for token in doc if token.lemma_.lower() in spatial_words)
        
        if spatial_count > 0:
            spatial_context['spatial_reference_density'] = spatial_count / len(doc)
            
        return spatial_context
        
    def _extract_relationships(self, doc, entities: List = None) -> List[EntityRelationship]:
        """Extract relationships between entities."""
        
        relationships = []
        
        if not entities or len(entities) < 2:
            return relationships
            
        # Simple relationship extraction based on proximity and patterns
        entity_positions = {}
        for entity in entities:
            # Find entity positions in text
            entity_text = entity.text if hasattr(entity, 'text') else str(entity)
            for token in doc:
                if entity_text.lower() in token.text.lower():
                    entity_positions[entity_text] = token.i
                    break
        
        # Find relationships between nearby entities
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                entity1_text = entity1.text if hasattr(entity1, 'text') else str(entity1)
                entity2_text = entity2.text if hasattr(entity2, 'text') else str(entity2)
                
                # Check if entities are mentioned close to each other
                pos1 = entity_positions.get(entity1_text)
                pos2 = entity_positions.get(entity2_text)
                
                if pos1 is not None and pos2 is not None:
                    distance = abs(pos1 - pos2)
                    if distance <= 5:  # Entities within 5 tokens
                        relationship = EntityRelationship(
                            entity1=entity1_text,
                            entity2=entity2_text,
                            relationship_type="proximity",
                            strength=max(0.1, 1.0 - (distance / 10)),
                            context="nearby_mention"
                        )
                        relationships.append(relationship)
        
        return relationships
        
    def _extract_key_concepts(self, doc) -> List[str]:
        """Extract key concepts from the text."""
        
        # Get important noun phrases and entities
        key_concepts = []
        
        # Extract noun chunks
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 3 and not chunk.root.is_stop:
                key_concepts.append(chunk.text.lower())
        
        # Extract important single nouns
        for token in doc:
            if (token.pos_ == 'NOUN' and 
                not token.is_stop and 
                len(token.text) > 4 and
                token.text.lower() not in key_concepts):
                key_concepts.append(token.text.lower())
        
        # Limit and deduplicate
        return list(set(key_concepts))[:10]
        
    def _identify_discourse_markers(self, doc) -> List[str]:
        """Identify discourse markers in the text."""
        
        markers = []
        
        for token in doc:
            token_lower = token.lemma_.lower()
            
            for marker_type, marker_set in self.discourse_markers.items():
                if token_lower in marker_set:
                    markers.append(f"{marker_type}:{token.text}")
                    
        # Also check for multi-word markers
        text_lower = doc.text.lower()
        multi_word_markers = [
            'above all', 'most importantly', 'in contrast', 'on the other hand',
            'in addition', 'as well', 'as a result', 'in conclusion', 'to sum up'
        ]
        
        for marker in multi_word_markers:
            if marker in text_lower:
                for marker_type, marker_set in self.discourse_markers.items():
                    if marker in marker_set:
                        markers.append(f"{marker_type}:{marker}")
                        break
        
        return markers
        
    def _find_emphasis_indicators(self, doc, emphasized_words: List[str] = None) -> List[str]:
        """Find linguistic indicators of emphasis."""
        
        indicators = []
        
        # Capitalization emphasis  
        for token in doc:
            if token.text.isupper() and len(token.text) > 2:
                indicators.append(f"capitalization:{token.text}")
                
        # Repetition emphasis
        word_counts = Counter(token.lemma_.lower() for token in doc if not token.is_punct)
        for word, count in word_counts.items():
            if count > 2 and len(word) > 3:
                indicators.append(f"repetition:{word}")
                
        # Intensifiers
        intensifiers = {'very', 'extremely', 'absolutely', 'completely', 'totally'}
        for token in doc:
            if token.lemma_.lower() in intensifiers:
                indicators.append(f"intensifier:{token.text}")
                
        # Emphasized words from external analysis
        if emphasized_words:
            for word in emphasized_words:
                indicators.append(f"acoustic_emphasis:{word}")
                
        return indicators
        
    def _calculate_coherence(self, doc) -> float:
        """Calculate text coherence score."""
        
        # Simple coherence based on entity repetition and discourse markers
        coherence_score = 0.0
        
        # Entity repetition (shows topic consistency)
        entities = [ent.text for ent in doc.ents]
        if entities:
            entity_repetition = (len(entities) - len(set(entities))) / len(entities)
            coherence_score += entity_repetition * 0.5
            
        # Discourse marker presence
        marker_count = 0
        for token in doc:
            for marker_set in self.discourse_markers.values():
                if token.lemma_.lower() in marker_set:
                    marker_count += 1
                    break
                    
        if len(doc) > 0:
            marker_density = marker_count / len(doc)
            coherence_score += min(0.5, marker_density * 10)
            
        return min(1.0, coherence_score)
        
    def _calculate_complexity(self, doc) -> float:
        """Calculate text complexity score."""
        
        complexity_score = 0.0
        
        if len(doc) == 0:
            return 0.0
            
        # Average sentence length
        sentences = list(doc.sents)
        if sentences:
            avg_sentence_length = len(doc) / len(sentences)
            complexity_score += min(0.3, avg_sentence_length / 50)
            
        # Average word length
        words = [token for token in doc if not token.is_punct and not token.is_space]
        if words:
            avg_word_length = sum(len(token.text) for token in words) / len(words)
            complexity_score += min(0.3, avg_word_length / 15)
            
        # Syntactic complexity (dependency depth)
        max_depth = 0
        for token in doc:
            depth = self._get_dependency_depth(token)
            max_depth = max(max_depth, depth)
            
        complexity_score += min(0.4, max_depth / 10)
        
        return min(1.0, complexity_score)
        
    def _get_dependency_depth(self, token) -> int:
        """Get dependency tree depth for a token."""
        depth = 0
        current = token
        
        while current.head != current and depth < 20:  # Prevent infinite loops
            current = current.head
            depth += 1
            
        return depth
        
    def get_analysis_summary(self, analysis: SemanticAnalysis) -> Dict:
        """Get a summary of semantic analysis results."""
        
        return {
            'text_length': len(analysis.text),
            'main_topics': analysis.semantic_context.topics,
            'sentiment': analysis.semantic_context.sentiment,
            'sentiment_score': analysis.semantic_context.sentiment_score,
            'tone': analysis.semantic_context.tone,
            'formality_level': analysis.semantic_context.formality,
            'urgency_level': analysis.semantic_context.urgency,
            'entity_relationships': len(analysis.entity_relationships),
            'key_concepts': len(analysis.key_concepts),
            'discourse_markers': len(analysis.discourse_markers),
            'emphasis_indicators': len(analysis.emphasis_indicators),
            'coherence_score': analysis.coherence_score,
            'complexity_score': analysis.complexity_score,
            'has_temporal_context': bool(analysis.semantic_context.temporal_context),
            'has_spatial_context': bool(analysis.semantic_context.spatial_context)
        } 