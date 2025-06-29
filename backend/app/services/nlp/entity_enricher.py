"""
Entity Enricher for Enhanced Metadata and Knowledge

Enriches recognized entities with:
- External knowledge base information
- Wikipedia/Wikidata links
- Image tags and visual attributes
- Semantic relationships
- Popularity and relevance scores
"""

import logging
import json
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import requests
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KnowledgeLink:
    """External knowledge base link."""
    source: str                    # Knowledge base name (wikipedia, wikidata, etc.)
    url: str                      # Direct URL to entity page
    confidence: float             # Link confidence (0-1)
    description: str              # Brief description
    image_urls: List[str] = field(default_factory=list)  # Associated images
    
@dataclass 
class SemanticInfo:
    """Semantic information about an entity."""
    categories: List[str] = field(default_factory=list)     # Semantic categories
    related_entities: List[str] = field(default_factory=list)  # Related entity names
    synonyms: List[str] = field(default_factory=list)       # Alternative names
    image_tags: List[str] = field(default_factory=list)     # Visual search tags
    popularity_score: float = 0.0                           # Popularity/fame score
    recency_score: float = 0.0                              # How recent/current

@dataclass
class EnrichedEntity:
    """Entity enriched with external knowledge and metadata."""
    # Core entity information (from EntityResult)
    text: str
    entity_type: str
    confidence: float
    image_potential: str
    canonical_name: str
    
    # Enhanced enrichment data
    knowledge_links: List[KnowledgeLink] = field(default_factory=list)
    semantic_info: SemanticInfo = field(default_factory=SemanticInfo)
    enrichment_timestamp: datetime = field(default_factory=datetime.now)
    enrichment_sources: List[str] = field(default_factory=list)
    
    # Image optimization data
    visual_attributes: Dict[str, float] = field(default_factory=dict)
    search_queries: List[str] = field(default_factory=list)  # Optimized search terms
    image_quality_score: float = 0.0                        # Expected image quality

class EntityEnricher:
    """Advanced entity enrichment system."""
    
    def __init__(self):
        """Initialize the entity enricher."""
        
        # Knowledge base configurations
        self.knowledge_bases = {
            'wikipedia': {
                'api_url': 'https://en.wikipedia.org/api/rest_v1',
                'search_url': 'https://en.wikipedia.org/w/api.php',
                'enabled': True
            },
            'wikidata': {
                'api_url': 'https://www.wikidata.org/w/api.php',
                'enabled': True  
            }
        }
        
        # Image optimization mappings
        self._init_image_optimization_data()
        
        # Popularity databases (can be enhanced with real APIs)
        self._init_popularity_data()
        
        # Request session for efficiency
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'VideoEnhancementSaaS/1.0 (Educational Purpose)'
        })
    
    def _init_image_optimization_data(self):
        """Initialize image search optimization data."""
        
        # Visual attributes for different entity types
        self.visual_attributes = {
            'PERSON': {
                'portrait': 0.9,
                'professional': 0.8,
                'headshot': 0.7,
                'public_appearance': 0.6
            },
            'ORGANIZATION': {
                'logo': 0.9,
                'building': 0.7,
                'headquarters': 0.6,
                'products': 0.5
            },
            'LOCATION': {
                'landmark': 0.9,
                'skyline': 0.8,
                'map': 0.7,
                'flag': 0.6
            },
            'PRODUCT': {
                'product_shot': 0.9,
                'packaging': 0.7,
                'in_use': 0.6,
                'advertisement': 0.5
            },
            'BRAND': {
                'logo': 0.9,
                'advertisement': 0.7,
                'storefront': 0.6,
                'products': 0.5
            }
        }
        
        # Search query templates
        self.search_templates = {
            'PERSON': [
                '{name} portrait',
                '{name} headshot',
                '{name} official photo',
                '{name} professional'
            ],
            'ORGANIZATION': [
                '{name} logo',
                '{name} headquarters',
                '{name} building',
                '{name} official'
            ],
            'LOCATION': [
                '{name} landmark',
                '{name} skyline',
                '{name} aerial view',
                '{name} flag'
            ],
            'PRODUCT': [
                '{name} product',
                '{name} official image',
                '{name} high quality',
                '{name} professional'
            ],
            'BRAND': [
                '{name} logo',
                '{name} brand image',
                '{name} official',
                '{name} high resolution'
            ]
        }
    
    def _init_popularity_data(self):
        """Initialize popularity scoring data."""
        
        # Known high-profile entities (for demo - would be from real APIs)
        self.popularity_scores = {
            # Politicians
            'biden': 0.95, 'trump': 0.95, 'putin': 0.90, 'xi jinping': 0.85,
            'modi': 0.80, 'macron': 0.75, 'merkel': 0.80,
            
            # Business leaders  
            'elon musk': 0.95, 'jeff bezos': 0.90, 'bill gates': 0.85,
            'mark zuckerberg': 0.80, 'tim cook': 0.75,
            
            # Companies
            'apple': 0.95, 'google': 0.95, 'microsoft': 0.90, 'amazon': 0.90,
            'tesla': 0.85, 'meta': 0.80, 'netflix': 0.75,
            
            # Countries/Places
            'united states': 0.95, 'china': 0.90, 'russia': 0.85,
            'new york': 0.85, 'london': 0.80, 'paris': 0.80
        }
    
    def enrich_entities(self, entities: List) -> List[EnrichedEntity]:
        """
        Enrich a list of entities with external knowledge and metadata.
        
        Args:
            entities: List of EntityResult objects to enrich
            
        Returns:
            List of enriched entities with additional metadata
        """
        enriched_entities = []
        
        for entity in entities:
            try:
                enriched = self._enrich_single_entity(entity)
                enriched_entities.append(enriched)
            except Exception as e:
                logger.warning(f"Failed to enrich entity {entity.text}: {e}")
                # Create basic enriched entity even if enrichment fails
                enriched = self._create_basic_enriched_entity(entity)
                enriched_entities.append(enriched)
        
        # Sort by overall enrichment quality
        enriched_entities.sort(
            key=lambda e: (
                e.image_quality_score,
                e.semantic_info.popularity_score,
                e.confidence
            ),
            reverse=True
        )
        
        return enriched_entities
    
    def _enrich_single_entity(self, entity) -> EnrichedEntity:
        """Enrich a single entity with all available data."""
        
        # Create base enriched entity
        enriched = EnrichedEntity(
            text=entity.text,
            entity_type=entity.entity_type.value,
            confidence=entity.confidence,
            image_potential=entity.image_potential.value,
            canonical_name=entity.canonical_name
        )
        
        # Add knowledge base links
        knowledge_links = self._get_knowledge_links(entity)
        enriched.knowledge_links = knowledge_links
        
        # Add semantic information
        semantic_info = self._get_semantic_info(entity)
        enriched.semantic_info = semantic_info
        
        # Generate optimized search queries
        search_queries = self._generate_search_queries(entity)
        enriched.search_queries = search_queries
        
        # Calculate image quality score
        image_quality = self._calculate_image_quality_score(entity)
        enriched.image_quality_score = image_quality
        
        # Add visual attributes
        visual_attrs = self._get_visual_attributes(entity)
        enriched.visual_attributes = visual_attrs
        
        # Record enrichment sources
        enriched.enrichment_sources = ['internal_db', 'popularity_scoring', 'image_optimization']
        
        return enriched
    
    def _get_knowledge_links(self, entity) -> List[KnowledgeLink]:
        """Get knowledge base links for an entity."""
        links = []
        
        # Wikipedia search (simplified - in production would use real API)
        if self.knowledge_bases['wikipedia']['enabled']:
            wikipedia_link = self._search_wikipedia(entity.canonical_name)
            if wikipedia_link:
                links.append(wikipedia_link)
        
        return links
    
    def _search_wikipedia(self, entity_name: str) -> Optional[KnowledgeLink]:
        """Search Wikipedia for entity information."""
        try:
            # Simplified Wikipedia search - in production would use real API
            # For now, create a mock link for known entities
            
            search_term = entity_name.lower().replace(' ', '_')
            url = f"https://en.wikipedia.org/wiki/{search_term}"
            
            # Mock confidence based on entity recognition
            confidence = 0.8 if entity_name.lower() in self.popularity_scores else 0.6
            
            return KnowledgeLink(
                source='wikipedia',
                url=url,
                confidence=confidence,
                description=f"Wikipedia page for {entity_name}",
                image_urls=[f"https://commons.wikimedia.org/wiki/{search_term}"]
            )
            
        except Exception as e:
            logger.warning(f"Wikipedia search failed for {entity_name}: {e}")
            return None
    
    def _get_semantic_info(self, entity) -> SemanticInfo:
        """Get semantic information for an entity."""
        
        semantic_info = SemanticInfo()
        
        # Get popularity score
        popularity = self.popularity_scores.get(
            entity.canonical_name.lower(), 
            self._estimate_popularity(entity)
        )
        semantic_info.popularity_score = popularity
        
        # Calculate recency score (how current/recent the entity is)
        recency = self._calculate_recency_score(entity)
        semantic_info.recency_score = recency
        
        # Get semantic categories
        categories = self._get_semantic_categories(entity)
        semantic_info.categories = categories
        
        # Generate image tags for visual search
        image_tags = self._generate_image_tags(entity)
        semantic_info.image_tags = image_tags
        
        # Get related entities (simplified)
        related = self._get_related_entities(entity)
        semantic_info.related_entities = related
        
        return semantic_info
    
    def _estimate_popularity(self, entity) -> float:
        """Estimate popularity score for unknown entities."""
        
        # Base score from entity type
        type_popularity = {
            'PERSON': 0.5,
            'ORGANIZATION': 0.6,
            'LOCATION': 0.4,
            'PRODUCT': 0.5,
            'BRAND': 0.6,
            'EVENT': 0.4
        }
        
        base_score = type_popularity.get(entity.entity_type.value, 0.3)
        
        # Adjust based on entity properties
        if hasattr(entity, 'context_score'):
            base_score = min(1.0, base_score + (entity.context_score * 0.3))
        
        if hasattr(entity, 'confidence'):
            base_score = min(1.0, base_score + (entity.confidence * 0.2))
        
        return base_score
    
    def _calculate_recency_score(self, entity) -> float:
        """Calculate how recent/current an entity is."""
        
        # Recent terms and concepts (would be dynamically updated)
        recent_terms = {
            'ai', 'chatgpt', 'covid', 'ukraine', 'climate change',
            'metaverse', 'nft', 'cryptocurrency', 'web3', 'electric vehicle'
        }
        
        entity_lower = entity.canonical_name.lower()
        
        # Check for recent terms
        for term in recent_terms:
            if term in entity_lower:
                return 0.8
        
        # Tech companies and modern brands tend to be more recent
        if entity.entity_type.value in ['TECHNOLOGY', 'BRAND']:
            return 0.6
        
        # Default recency
        return 0.4
    
    def _get_semantic_categories(self, entity) -> List[str]:
        """Get semantic categories for an entity."""
        
        categories = []
        entity_type = entity.entity_type.value
        
        # Add base category
        categories.append(entity_type.lower())
        
        # Add specific categories based on type and name
        name_lower = entity.canonical_name.lower()
        
        if entity_type == 'PERSON':
            if any(term in name_lower for term in ['president', 'biden', 'trump']):
                categories.extend(['politician', 'government', 'leadership'])
            elif any(term in name_lower for term in ['musk', 'bezos', 'gates']):
                categories.extend(['business_leader', 'entrepreneur', 'technology'])
        
        elif entity_type == 'ORGANIZATION':
            if any(term in name_lower for term in ['apple', 'google', 'microsoft']):
                categories.extend(['technology', 'software', 'innovation'])
            elif any(term in name_lower for term in ['tesla', 'spacex']):
                categories.extend(['automotive', 'space', 'electric_vehicle'])
        
        elif entity_type == 'LOCATION':
            if any(term in name_lower for term in ['united states', 'china', 'russia']):
                categories.extend(['country', 'geopolitics', 'government'])
            elif any(term in name_lower for term in ['new york', 'london', 'paris']):
                categories.extend(['city', 'metropolitan', 'culture'])
        
        return categories
    
    def _generate_image_tags(self, entity) -> List[str]:
        """Generate image tags for visual search optimization."""
        
        tags = []
        entity_type = entity.entity_type.value
        name = entity.canonical_name.lower()
        
        # Add type-specific tags
        if entity_type == 'PERSON':
            tags.extend(['person', 'portrait', 'headshot', 'professional'])
            if 'president' in name or any(p in name for p in ['biden', 'trump']):
                tags.extend(['politician', 'government', 'official'])
        
        elif entity_type == 'ORGANIZATION':
            tags.extend(['company', 'logo', 'corporate', 'business'])
            if any(tech in name for tech in ['apple', 'google', 'microsoft']):
                tags.extend(['technology', 'tech_company', 'software'])
        
        elif entity_type == 'LOCATION':
            tags.extend(['place', 'geography', 'location'])
            if any(country in name for country in ['united states', 'china']):
                tags.extend(['country', 'flag', 'map', 'nation'])
        
        elif entity_type == 'PRODUCT':
            tags.extend(['product', 'item', 'commercial'])
        
        elif entity_type == 'BRAND':
            tags.extend(['brand', 'logo', 'trademark', 'corporate'])
        
        return tags
    
    def _get_related_entities(self, entity) -> List[str]:
        """Get related entities (simplified relationships)."""
        
        # Known relationships (would be from knowledge graph in production)
        relationships = {
            'biden': ['united states', 'president', 'democrats'],
            'trump': ['united states', 'president', 'republicans'], 
            'apple': ['tim cook', 'iphone', 'technology'],
            'google': ['alphabet', 'android', 'search'],
            'tesla': ['elon musk', 'electric vehicle', 'spacex'],
            'ukraine': ['russia', 'war', 'europe', 'zelensky']
        }
        
        entity_lower = entity.canonical_name.lower()
        return relationships.get(entity_lower, [])
    
    def _generate_search_queries(self, entity) -> List[str]:
        """Generate optimized search queries for images."""
        
        queries = []
        entity_type = entity.entity_type.value
        name = entity.canonical_name
        
        # Get templates for entity type
        templates = self.search_templates.get(entity_type, ['{name}'])
        
        # Generate queries from templates
        for template in templates:
            query = template.format(name=name)
            queries.append(query)
        
        # Add high-quality modifiers
        quality_modifiers = ['high resolution', 'professional', 'official', 'clear']
        for modifier in quality_modifiers[:2]:  # Limit to avoid too many queries
            queries.append(f"{name} {modifier}")
        
        return queries[:6]  # Limit to 6 queries per entity
    
    def _calculate_image_quality_score(self, entity) -> float:
        """Calculate expected image quality/availability score."""
        
        score = 0.0
        
        # Base score from image potential
        potential_scores = {
            'excellent': 0.9,
            'good': 0.7,
            'moderate': 0.5,
            'poor': 0.2
        }
        score += potential_scores.get(entity.image_potential.value, 0.3)
        
        # Boost for popular entities
        popularity = self.popularity_scores.get(entity.canonical_name.lower(), 0.5)
        score += popularity * 0.3
        
        # Boost for recent entities  
        if hasattr(entity, 'context_score'):
            score += entity.context_score * 0.2
        
        return min(1.0, score)
    
    def _get_visual_attributes(self, entity) -> Dict[str, float]:
        """Get visual attributes for image search."""
        
        entity_type = entity.entity_type.value
        return self.visual_attributes.get(entity_type, {})
    
    def _create_basic_enriched_entity(self, entity) -> EnrichedEntity:
        """Create a basic enriched entity when full enrichment fails."""
        
        return EnrichedEntity(
            text=entity.text,
            entity_type=entity.entity_type.value,
            confidence=entity.confidence,
            image_potential=entity.image_potential.value,
            canonical_name=entity.canonical_name,
            enrichment_sources=['basic_fallback']
        )
    
    def get_best_image_entities(
        self, 
        enriched_entities: List[EnrichedEntity], 
        max_count: int = 5
    ) -> List[EnrichedEntity]:
        """Get the best entities for image search."""
        
        # Sort by image optimization score
        sorted_entities = sorted(
            enriched_entities,
            key=lambda e: (
                e.image_quality_score,
                e.semantic_info.popularity_score,
                e.confidence
            ),
            reverse=True
        )
        
        return sorted_entities[:max_count]
    
    def get_enrichment_statistics(self, enriched_entities: List[EnrichedEntity]) -> Dict:
        """Get statistics about entity enrichment."""
        
        if not enriched_entities:
            return {}
        
        # Calculate averages
        avg_image_quality = np.mean([e.image_quality_score for e in enriched_entities])
        avg_popularity = np.mean([e.semantic_info.popularity_score for e in enriched_entities])
        
        # Count enrichment sources
        source_counts = {}
        for entity in enriched_entities:
            for source in entity.enrichment_sources:
                source_counts[source] = source_counts.get(source, 0) + 1
        
        # Count by image potential
        potential_counts = {}
        for entity in enriched_entities:
            potential = entity.image_potential
            potential_counts[potential] = potential_counts.get(potential, 0) + 1
        
        return {
            'total_entities': len(enriched_entities),
            'avg_image_quality_score': float(avg_image_quality),
            'avg_popularity_score': float(avg_popularity),
            'enrichment_sources': source_counts,
            'image_potential_distribution': potential_counts,
            'high_quality_entities': len([
                e for e in enriched_entities 
                if e.image_quality_score > 0.7
            ])
        } 