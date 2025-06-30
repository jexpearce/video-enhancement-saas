"""
Advanced Entity Recognition System

Provides sophisticated entity recognition using multiple sources:
- spaCy NER for general entities
- Regex patterns for specific entity types  
- Context-aware entity detection
- Image-optimized entity classification
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import spacy
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EntityType(Enum):
    """Enhanced entity types optimized for image retrieval."""
    # High image potential
    PERSON = "PERSON"           # People - photos available
    ORGANIZATION = "ORG"        # Companies - logos, buildings
    LOCATION = "LOCATION"       # Places - maps, landmarks  
    PRODUCT = "PRODUCT"         # Products - product images
    BRAND = "BRAND"             # Brands - logos, imagery
    EVENT = "EVENT"             # Events - photos, coverage
    
    # Medium image potential  
    CONCEPT = "CONCEPT"         # Abstract concepts
    TECHNOLOGY = "TECHNOLOGY"   # Tech terms
    CURRENCY = "CURRENCY"       # Money, financial
    
    # Lower image potential
    NUMBER = "NUMBER"           # Numeric values
    DATE = "DATE"               # Dates/times
    MISC = "MISC"               # Other entities

class ImagePotential(Enum):
    """Classification of how easily an entity can be visualized."""
    EXCELLENT = "excellent"     # Perfect for images (people, places, products)
    GOOD = "good"              # Good for images (events, organizations)  
    MODERATE = "moderate"      # Some visual potential (concepts, tech)
    POOR = "poor"              # Difficult to visualize (dates, numbers)

@dataclass
class EntityResult:
    """Comprehensive entity recognition result."""
    text: str                          # Original entity text
    entity_type: EntityType           # Classified entity type
    start_char: int                   # Start position in text
    end_char: int                     # End position in text
    confidence: float                 # Recognition confidence (0-1)
    image_potential: ImagePotential   # Visual searchability
    
    # Enhanced metadata
    canonical_name: str               # Standardized entity name
    aliases: List[str]                # Alternative names
    category: str                     # Detailed category
    context_score: float              # Importance in context
    
    # Source information
    recognition_sources: List[str]    # Which systems detected it
    spacy_label: Optional[str]        # Original spaCy label
    regex_pattern: Optional[str]      # Regex pattern if matched

class EntityRecognizer:
    """Advanced multi-source entity recognition system."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the entity recognizer."""
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.error(f"Failed to load spaCy model: {model_name}")
            raise
        
        # Initialize pattern matchers
        self._init_regex_patterns()
        self._init_entity_mappings()
        
        # Image potential mapping
        self.image_potential_map = {
            EntityType.PERSON: ImagePotential.EXCELLENT,
            EntityType.ORGANIZATION: ImagePotential.EXCELLENT, 
            EntityType.LOCATION: ImagePotential.EXCELLENT,
            EntityType.PRODUCT: ImagePotential.EXCELLENT,
            EntityType.BRAND: ImagePotential.EXCELLENT,
            EntityType.EVENT: ImagePotential.GOOD,
            EntityType.TECHNOLOGY: ImagePotential.MODERATE,
            EntityType.CONCEPT: ImagePotential.MODERATE,
            EntityType.CURRENCY: ImagePotential.MODERATE,
            EntityType.NUMBER: ImagePotential.POOR,
            EntityType.DATE: ImagePotential.POOR,
            EntityType.MISC: ImagePotential.POOR
        }
        
    def _init_regex_patterns(self):
        """Initialize regex patterns for specific entity types."""
        self.regex_patterns = {
            # Technology patterns
            'technology': [
                r'\b(?:AI|ML|IoT|5G|API|CPU|GPU|SaaS|AWS|Docker|React|Python)\b',
                r'\b(?:blockchain|cryptocurrency|bitcoin|ethereum)\b',
                r'\b(?:iPhone|iPad|Android|Windows|macOS|Linux)\b'
            ],
            
            # Brand patterns  
            'brand': [
                r'\b(?:Apple|Google|Microsoft|Amazon|Meta|Tesla|Netflix)\b',
                r'\b(?:Nike|Adidas|Samsung|Sony|LG|BMW|Mercedes)\b',
                r'\b(?:Coca-Cola|Pepsi|McDonald\'s|Starbucks)\b'
            ],
            
            # Currency patterns
            'currency': [
                r'\$[\d,]+(?:\.\d{2})?(?:\s?(?:million|billion|trillion))?',
                r'€[\d,]+(?:\.\d{2})?(?:\s?(?:million|billion|trillion))?',
                r'£[\d,]+(?:\.\d{2})?(?:\s?(?:million|billion|trillion))?',
                r'\b[\d,]+(?:\.\d{2})?\s?(?:dollars|euros|pounds|USD|EUR|GBP)\b'
            ],
            
            # Product patterns
            'product': [
                r'\b(?:iPhone\s?\d+|iPad\s?Pro|MacBook\s?(?:Pro|Air)?)\b',
                r'\b(?:Tesla\s?Model\s?[3SXY]|Cybertruck)\b',
                r'\b(?:PlayStation\s?\d+|Xbox\s?(?:Series\s?[XS]|One))\b'
            ]
        }
        
        # Compile regex patterns
        self.compiled_patterns = {}
        for category, patterns in self.regex_patterns.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def _init_entity_mappings(self):
        """Initialize mappings from spaCy labels to our entity types."""
        self.spacy_to_entity_type = {
            'PERSON': EntityType.PERSON,
            'ORG': EntityType.ORGANIZATION,
            'GPE': EntityType.LOCATION,        # Geopolitical entity
            'LOC': EntityType.LOCATION,        # Location
            'PRODUCT': EntityType.PRODUCT,
            'EVENT': EntityType.EVENT,
            'WORK_OF_ART': EntityType.CONCEPT,
            'LAW': EntityType.CONCEPT,
            'LANGUAGE': EntityType.CONCEPT,
            'NORP': EntityType.CONCEPT,        # Nationalities, religious groups
            'FAC': EntityType.LOCATION,        # Facilities
            'MONEY': EntityType.CURRENCY,
            'PERCENT': EntityType.NUMBER,
            'DATE': EntityType.DATE,
            'TIME': EntityType.DATE,
            'QUANTITY': EntityType.NUMBER,
            'ORDINAL': EntityType.NUMBER,
            'CARDINAL': EntityType.NUMBER
        }
        
        # Known entity aliases and canonical names
        self.canonical_names = {
            # People
            'biden': 'Joe Biden',
            'trump': 'Donald Trump', 
            'musk': 'Elon Musk',
            'bezos': 'Jeff Bezos',
            'gates': 'Bill Gates',
            
            # Places
            'usa': 'United States',
            'uk': 'United Kingdom',
            'uae': 'United Arab Emirates',
            'nyc': 'New York City',
            
            # Companies
            'meta': 'Meta Platforms',
            'alphabet': 'Google',
            'msft': 'Microsoft',
            'amzn': 'Amazon',
            'tsla': 'Tesla'
        }
    
    def recognize_entities(self, text: str, emphasized_words: List[str] = None) -> List[EntityResult]:
        """
        Recognize entities in text using multiple sources.
        
        Args:
            text: Input text to analyze
            emphasized_words: List of emphasized words to boost importance
            
        Returns:
            List of entity recognition results
        """
        try:
            # Get entities from multiple sources
            spacy_entities = self._extract_spacy_entities(text)
            regex_entities = self._extract_regex_entities(text)
            
            # Merge and deduplicate entities
            all_entities = self._merge_entities(spacy_entities + regex_entities)
            
            # Enhance entities with metadata
            enhanced_entities = []
            for entity in all_entities:
                enhanced = self._enhance_entity(entity, text, emphasized_words)
                enhanced_entities.append(enhanced)
            
            # Sort by importance (image potential + context score)
            enhanced_entities.sort(
                key=lambda e: (e.image_potential.value, e.context_score, e.confidence),
                reverse=True
            )
            
            return enhanced_entities
            
        except Exception as e:
            logger.error(f"Error in entity recognition: {e}")
            raise
    
    def _extract_spacy_entities(self, text: str) -> List[EntityResult]:
        """Extract entities using spaCy NER."""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            # Map spaCy label to our entity type
            entity_type = self.spacy_to_entity_type.get(ent.label_, EntityType.MISC)
            
            # Calculate confidence based on entity properties
            confidence = self._calculate_spacy_confidence(ent)
            
            entity = EntityResult(
                text=ent.text,
                entity_type=entity_type,
                start_char=ent.start_char,
                end_char=ent.end_char,
                confidence=confidence,
                image_potential=self.image_potential_map[entity_type],
                canonical_name=ent.text,
                aliases=list(),
                category=ent.label_,
                context_score=0.0,
                recognition_sources=['spacy'],
                spacy_label=ent.label_,
                regex_pattern=None
            )
            entities.append(entity)
        
        return entities
    
    def _extract_regex_entities(self, text: str) -> List[EntityResult]:
        """Extract entities using regex patterns."""
        entities = []
        
        for category, patterns in self.compiled_patterns.items():
            entity_type = self._category_to_entity_type(category)
            
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entity = EntityResult(
                        text=match.group(),
                        entity_type=entity_type,
                        start_char=match.start(),
                        end_char=match.end(),
                        confidence=0.8,  # High confidence for regex matches
                        image_potential=self.image_potential_map[entity_type],
                        canonical_name=match.group(),
                        aliases=list(),
                        category=category,
                        context_score=0.0,
                        recognition_sources=['regex'],
                        spacy_label=None,
                        regex_pattern=pattern.pattern
                    )
                    entities.append(entity)
        
        return entities
    
    def _category_to_entity_type(self, category: str) -> EntityType:
        """Map regex category to entity type."""
        mapping = {
            'technology': EntityType.TECHNOLOGY,
            'brand': EntityType.BRAND,
            'currency': EntityType.CURRENCY,
            'product': EntityType.PRODUCT
        }
        return mapping.get(category, EntityType.MISC)
    
    def _calculate_spacy_confidence(self, ent) -> float:
        """Calculate confidence score for spaCy entities."""
        # Base confidence from entity length and type
        base_confidence = min(1.0, 0.5 + (len(ent.text) / 20))
        
        # Boost confidence for known high-quality entity types
        high_confidence_types = {'PERSON', 'ORG', 'GPE', 'PRODUCT'}
        if ent.label_ in high_confidence_types:
            base_confidence = min(1.0, base_confidence + 0.2)
        
        # Check if entity is in canonical names (known entities)
        if ent.text.lower() in self.canonical_names:
            base_confidence = min(1.0, base_confidence + 0.3)
        
        return base_confidence
    
    def _merge_entities(self, entities: List[EntityResult]) -> List[EntityResult]:
        """Merge overlapping entities from different sources."""
        if not entities:
            return []
        
        # Sort by start position
        entities.sort(key=lambda e: e.start_char)
        
        merged = []
        current = entities[0]
        
        for entity in entities[1:]:
            # Check for overlap
            if entity.start_char < current.end_char:
                # Overlapping entities - keep the one with higher confidence
                if entity.confidence > current.confidence:
                    # Merge sources
                    entity.recognition_sources.extend(current.recognition_sources)
                    current = entity
                else:
                    # Keep current, but add source info
                    current.recognition_sources.extend(entity.recognition_sources)
            else:
                # No overlap - add current to results and update current
                merged.append(current)
                current = entity
        
        # Add the last entity
        merged.append(current)
        
        return merged
    
    def _enhance_entity(
        self, 
        entity: EntityResult, 
        full_text: str, 
        emphasized_words: List[str] = None
    ) -> EntityResult:
        """Enhance entity with additional metadata and context scoring."""
        
        # Get canonical name and aliases
        canonical_name = self.canonical_names.get(
            entity.text.lower(), 
            entity.text
        )
        entity.canonical_name = canonical_name
        
        # Calculate context score
        context_score = self._calculate_context_score(
            entity, full_text, emphasized_words
        )
        entity.context_score = context_score
        
        # Add aliases for known entities
        entity.aliases = self._get_entity_aliases(entity.text)
        
        return entity
    
    def _calculate_context_score(
        self, 
        entity: EntityResult, 
        full_text: str, 
        emphasized_words: List[str] = None
    ) -> float:
        """Calculate contextual importance score for entity."""
        score = 0.0
        
        # Base score from entity type and image potential
        type_scores = {
            EntityType.PERSON: 0.9,
            EntityType.ORGANIZATION: 0.8,
            EntityType.LOCATION: 0.8,
            EntityType.PRODUCT: 0.8,
            EntityType.BRAND: 0.7,
            EntityType.EVENT: 0.6,
            EntityType.TECHNOLOGY: 0.5,
            EntityType.CONCEPT: 0.4,
            EntityType.CURRENCY: 0.3,
            EntityType.NUMBER: 0.2,
            EntityType.DATE: 0.1,
            EntityType.MISC: 0.1
        }
        score += type_scores.get(entity.entity_type, 0.1)
        
        # Boost score if entity was emphasized
        if emphasized_words and entity.text.lower() in [w.lower() for w in emphasized_words]:
            score += 0.3
        
        # Position-based scoring (earlier entities often more important)
        text_position = entity.start_char / len(full_text) if full_text else 0
        position_bonus = max(0, 0.2 * (1 - text_position))
        score += position_bonus
        
        # Length-based scoring (longer entities often more specific)
        length_bonus = min(0.2, len(entity.text) / 50)
        score += length_bonus
        
        return min(1.0, score)
    
    def _get_entity_aliases(self, entity_text: str) -> List[str]:
        """Get known aliases for an entity."""
        # This could be expanded with a comprehensive alias database
        aliases_map = {
            'biden': ['Joe Biden', 'President Biden', 'POTUS'],
            'apple': ['Apple Inc.', 'AAPL', 'Apple Computer'],
            'google': ['Alphabet', 'GOOGL', 'Google Inc.'],
            'tesla': ['TSLA', 'Tesla Motors', 'Tesla Inc.'],
            'usa': ['United States', 'America', 'US', 'U.S.A.'],
            'uk': ['United Kingdom', 'Britain', 'Great Britain']
        }
        
        return aliases_map.get(entity_text.lower(), [])
    
    def get_image_optimized_entities(
        self, 
        entities: List[EntityResult], 
        min_image_potential: ImagePotential = ImagePotential.MODERATE
    ) -> List[EntityResult]:
        """Filter entities by image potential for visual search optimization."""
        
        potential_order = {
            ImagePotential.EXCELLENT: 4,
            ImagePotential.GOOD: 3,
            ImagePotential.MODERATE: 2,
            ImagePotential.POOR: 1
        }
        
        min_score = potential_order[min_image_potential]
        
        filtered = [
            entity for entity in entities 
            if potential_order[entity.image_potential] >= min_score
        ]
        
        # Sort by image potential and context score
        filtered.sort(
            key=lambda e: (
                potential_order[e.image_potential], 
                e.context_score, 
                e.confidence
            ),
            reverse=True
        )
        
        return filtered
    
    def get_statistics(self, entities: List[EntityResult]) -> Dict:
        """Get statistics about recognized entities."""
        if not entities:
            return {}
        
        # Count by type
        type_counts = {}
        for entity in entities:
            type_name = entity.entity_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        # Count by image potential
        potential_counts = {}
        for entity in entities:
            potential = entity.image_potential.value
            potential_counts[potential] = potential_counts.get(potential, 0) + 1
        
        # Calculate averages
        avg_confidence = np.mean([e.confidence for e in entities])
        avg_context_score = np.mean([e.context_score for e in entities])
        
        return {
            'total_entities': len(entities),
            'type_distribution': type_counts,
            'image_potential_distribution': potential_counts,
            'avg_confidence': float(avg_confidence),
            'avg_context_score': float(avg_context_score),
            'excellent_image_entities': len([
                e for e in entities 
                if e.image_potential == ImagePotential.EXCELLENT
            ])
        } 