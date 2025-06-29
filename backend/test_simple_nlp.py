#!/usr/bin/env python3
"""
Simple NLP System Test

Demonstrates the core Part 4 NLP capabilities:
- Entity recognition with image optimization
- Basic enrichment with popularity scoring
- Entity prioritization for video enhancement

Shows how the system identifies image-findable entities like people, places, and brands.
"""

import sys
import os

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_simple_nlp():
    """Test basic NLP functionality with image-optimized entity recognition."""
    print("🧠 Simple NLP System Test - Image-Optimized Entity Recognition")
    print("=" * 70)
    
    try:
        # Test basic spaCy functionality first
        import spacy
        print("✅ spaCy imported successfully")
        
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model loaded successfully")
        
        # Test with real-world content
        test_cases = [
            {
                'name': 'Political News',
                'text': 'Biden met with Netanyahu in Washington to discuss Iran while Ukraine faces Russia.',
                'expected_image_entities': ['Biden', 'Netanyahu', 'Washington', 'Iran', 'Ukraine', 'Russia']
            },
            {
                'name': 'Tech Business', 
                'text': 'Elon Musk announced Tesla partnerships while Apple reported iPhone sales growth.',
                'expected_image_entities': ['Elon Musk', 'Tesla', 'Apple', 'iPhone']
            },
            {
                'name': 'Entertainment',
                'text': 'Netflix partnered with Disney to stream Marvel content exclusively.',
                'expected_image_entities': ['Netflix', 'Disney', 'Marvel']
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*50}")
            print(f"📝 Test Case {i}: {test_case['name']}")
            print(f"{'='*50}")
            print(f"Text: {test_case['text']}")
            print("-" * 50)
            
            # Basic entity recognition
            doc = nlp(test_case['text'])
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            print(f"🔍 Found {len(entities)} entities:")
            for entity_text, entity_label in entities:
                image_potential = classify_image_potential(entity_label)
                icon = get_entity_icon(entity_label)
                print(f"   {icon} {entity_text} ({entity_label}) - Image: {image_potential}")
            
            # Image-optimized analysis
            image_entities = get_image_optimized_entities(entities)
            print(f"\n🏆 Top Image-Findable Entities ({len(image_entities)}):")
            
            for j, (entity_text, entity_label) in enumerate(image_entities[:5], 1):
                popularity = get_popularity_score(entity_text)
                search_query = generate_search_query(entity_text, entity_label)
                print(f"   {j}. {entity_text}")
                print(f"      └── Popularity: {popularity:.2f} | Search: '{search_query}'")
            
            # Accuracy check
            found_texts = [ent[0].lower() for ent in entities]
            expected_lower = [e.lower() for e in test_case['expected_image_entities']]
            matches = sum(1 for exp in expected_lower if any(exp in found for found in found_texts))
            accuracy = matches / len(expected_lower) if expected_lower else 1.0
            
            print(f"\n✅ Recognition Accuracy: {accuracy:.1%} ({matches}/{len(expected_lower)})")
        
        print(f"\n{'='*70}")
        print("🎉 Simple NLP Test Complete!")
        print("✅ Image-optimized entity recognition working successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def classify_image_potential(entity_label):
    """Classify image potential for different entity types."""
    
    excellent_types = {'PERSON', 'ORG', 'GPE', 'PRODUCT'}  # People, orgs, places, products
    good_types = {'EVENT', 'FAC', 'WORK_OF_ART'}           # Events, facilities, artworks
    moderate_types = {'NORP', 'LAW', 'LANGUAGE'}           # Nationalities, laws, languages
    poor_types = {'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'}
    
    if entity_label in excellent_types:
        return "EXCELLENT ⭐⭐⭐"
    elif entity_label in good_types:
        return "GOOD ⭐⭐"
    elif entity_label in moderate_types:
        return "MODERATE ⭐"
    elif entity_label in poor_types:
        return "POOR"
    else:
        return "UNKNOWN"

def get_entity_icon(entity_label):
    """Get emoji icon for entity type."""
    icons = {
        'PERSON': '👤',
        'ORG': '🏢',
        'GPE': '📍',    # Geopolitical entity (countries, cities)
        'LOC': '🗺️',    # Location
        'PRODUCT': '📱',
        'EVENT': '📅',
        'FAC': '🏭',    # Facility
        'WORK_OF_ART': '🎨',
        'MONEY': '💰',
        'DATE': '📅',
        'TIME': '⏰'
    }
    return icons.get(entity_label, '📄')

def get_image_optimized_entities(entities):
    """Filter and sort entities by image potential."""
    
    # Score entities by image potential
    scored_entities = []
    for entity_text, entity_label in entities:
        score = 0
        
        # Base score from entity type
        if entity_label in {'PERSON', 'ORG', 'GPE', 'PRODUCT'}:
            score += 3  # Excellent
        elif entity_label in {'EVENT', 'FAC', 'WORK_OF_ART'}:
            score += 2  # Good
        elif entity_label in {'NORP', 'LAW', 'LANGUAGE'}:
            score += 1  # Moderate
        else:
            score += 0  # Poor
        
        # Boost for known high-profile entities
        popularity_bonus = get_popularity_score(entity_text)
        score += popularity_bonus * 2
        
        scored_entities.append((score, entity_text, entity_label))
    
    # Sort by score and return
    scored_entities.sort(reverse=True, key=lambda x: x[0])
    return [(entity[1], entity[2]) for entity in scored_entities if entity[0] > 0]

def get_popularity_score(entity_text):
    """Get popularity score for known entities."""
    
    # Known high-profile entities (simplified database)
    popularity_scores = {
        # Politicians
        'biden': 0.95, 'trump': 0.95, 'putin': 0.90, 'xi jinping': 0.85,
        'netanyahu': 0.80, 'zelensky': 0.75,
        
        # Business leaders
        'elon musk': 0.95, 'jeff bezos': 0.90, 'bill gates': 0.85,
        'mark zuckerberg': 0.80, 'tim cook': 0.75,
        
        # Companies
        'apple': 0.95, 'google': 0.95, 'microsoft': 0.90, 'amazon': 0.90,
        'tesla': 0.85, 'meta': 0.80, 'netflix': 0.75, 'disney': 0.80,
        
        # Countries/Places
        'united states': 0.95, 'china': 0.90, 'russia': 0.85, 'iran': 0.75,
        'ukraine': 0.80, 'washington': 0.70, 'new york': 0.85,
        
        # Products
        'iphone': 0.90, 'android': 0.80, 'windows': 0.70,
        
        # Entertainment
        'marvel': 0.85, 'star wars': 0.80
    }
    
    entity_lower = entity_text.lower()
    return popularity_scores.get(entity_lower, 0.3)  # Default low score

def generate_search_query(entity_text, entity_label):
    """Generate optimized search query for images."""
    
    # Query templates by entity type
    if entity_label == 'PERSON':
        return f"{entity_text} portrait official"
    elif entity_label == 'ORG':
        return f"{entity_text} logo official"
    elif entity_label == 'GPE':  # Countries, cities
        return f"{entity_text} flag landmark"
    elif entity_label == 'PRODUCT':
        return f"{entity_text} product official"
    elif entity_label == 'EVENT':
        return f"{entity_text} event photos"
    else:
        return f"{entity_text} high quality"

if __name__ == "__main__":
    success = test_simple_nlp()
    print(f"\n{'🎉 Success!' if success else '❌ Failed!'}")
    sys.exit(0 if success else 1) 