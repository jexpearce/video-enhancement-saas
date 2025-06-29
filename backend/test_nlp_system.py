#!/usr/bin/env python3
"""
Comprehensive NLP System Test

Tests the complete Part 4 NLP pipeline:
- Advanced Entity Recognition (multi-source, image-optimized)
- Entity Enrichment (knowledge linking, semantic info)
- Semantic Analysis (context, relationships, discourse)

Demonstrates how the system prioritizes image-findable entities
and provides rich metadata for video enhancement.
"""

import sys
import os
import json
from typing import List

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_nlp_system():
    """Test the complete NLP system with real-world content."""
    print("🧠 Testing Advanced NLP System for Video Enhancement")
    print("=" * 60)
    
    try:
        # Import all NLP components
        from backend.app.services.nlp import EntityRecognizer, EntityEnricher, SemanticAnalyzer
        from backend.app.services.emphasis import MultiModalEmphasisDetector
        
        # Initialize components
        print("🔧 Initializing NLP Components...")
        entity_recognizer = EntityRecognizer()
        entity_enricher = EntityEnricher()
        semantic_analyzer = SemanticAnalyzer()
        emphasis_detector = MultiModalEmphasisDetector(sample_rate=16000)
        print("✅ All components initialized successfully")
        
        # Test with multiple real-world scenarios
        test_cases = [
            {
                'name': 'Political News',
                'text': 'President Biden met with Netanyahu in Washington to discuss Iran sanctions while Ukraine continues its conflict with Russia.',
                'expected_entities': ['Biden', 'Netanyahu', 'Washington', 'Iran', 'Ukraine', 'Russia']
            },
            {
                'name': 'Tech Business',
                'text': 'Elon Musk announced that Tesla will expand manufacturing in China while Apple reported record iPhone sales.',
                'expected_entities': ['Elon Musk', 'Tesla', 'China', 'Apple', 'iPhone']
            },
            {
                'name': 'Entertainment',
                'text': 'Netflix announced a new partnership with Disney to stream Marvel movies exclusively on their platform.',
                'expected_entities': ['Netflix', 'Disney', 'Marvel']
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*60}")
            print(f"📝 Test Case {i}: {test_case['name']}")
            print(f"{'='*60}")
            print(f"Text: {test_case['text']}")
            print("-" * 60)
            
            # Run the complete NLP pipeline
            results = run_nlp_pipeline(
                test_case['text'],
                entity_recognizer,
                entity_enricher, 
                semantic_analyzer,
                emphasis_detector
            )
            
            # Display results
            display_results(results, test_case['expected_entities'])
            
        print(f"\n{'='*60}")
        print("🎉 NLP System Testing Complete!")
        print("✅ All components working together successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_nlp_pipeline(text, recognizer, enricher, analyzer, emphasis_detector):
    """Run the complete NLP pipeline on input text."""
    
    # Step 1: Multi-modal emphasis detection (using synthetic audio)
    print("🎯 Step 1: Multi-Modal Emphasis Detection")
    audio_data = create_synthetic_audio(text)
    transcription = create_transcription(text)
    
    emphasis_results = emphasis_detector.detect_emphasis(audio_data, transcription)
    emphasized_words = [r.word for r in emphasis_results if r.is_emphasized]
    
    print(f"   Emphasized words: {emphasized_words}")
    print(f"   Emphasis detection completed: {len(emphasis_results)} words analyzed")
    
    # Step 2: Advanced Entity Recognition
    print("\n🔍 Step 2: Advanced Entity Recognition")
    entities = recognizer.recognize_entities(text, emphasized_words)
    
    print(f"   Found {len(entities)} entities:")
    for entity in entities[:5]:  # Show top 5
        print(f"     • {entity.text} ({entity.entity_type.value}) - "
              f"Image: {entity.image_potential.value}, Conf: {entity.confidence:.2f}")
    
    # Step 3: Entity Enrichment
    print("\n🌟 Step 3: Entity Enrichment")
    enriched_entities = enricher.enrich_entities(entities)
    
    print(f"   Enriched {len(enriched_entities)} entities with metadata:")
    for entity in enriched_entities[:3]:  # Show top 3
        print(f"     • {entity.canonical_name}")
        print(f"       - Image quality: {entity.image_quality_score:.2f}")
        print(f"       - Popularity: {entity.semantic_info.popularity_score:.2f}")
        print(f"       - Search queries: {entity.search_queries[:2]}")
        print(f"       - Categories: {entity.semantic_info.categories[:3]}")
    
    # Step 4: Semantic Analysis
    print("\n🧠 Step 4: Semantic Analysis")
    semantic_analysis = analyzer.analyze_semantics(text, entities, emphasized_words)
    
    print(f"   Topics: {semantic_analysis.semantic_context.topics}")
    print(f"   Sentiment: {semantic_analysis.semantic_context.sentiment} "
          f"({semantic_analysis.semantic_context.sentiment_score:.2f})")
    print(f"   Tone: {semantic_analysis.semantic_context.tone}")
    print(f"   Relationships: {len(semantic_analysis.entity_relationships)}")
    print(f"   Key concepts: {semantic_analysis.key_concepts[:5]}")
    
    return {
        'text': text,
        'emphasis_results': emphasis_results,
        'emphasized_words': emphasized_words,
        'entities': entities,
        'enriched_entities': enriched_entities,
        'semantic_analysis': semantic_analysis,
        'entity_stats': recognizer.get_statistics(entities),
        'enrichment_stats': enricher.get_enrichment_statistics(enriched_entities),
        'semantic_summary': analyzer.get_analysis_summary(semantic_analysis)
    }

def display_results(results, expected_entities):
    """Display comprehensive results analysis."""
    
    print("\n📊 IMAGE-OPTIMIZED RESULTS ANALYSIS")
    print("-" * 40)
    
    # Best entities for image search
    best_entities = results['enriched_entities'][:5]
    print("🏆 Top Image-Findable Entities:")
    
    for i, entity in enumerate(best_entities, 1):
        icon = get_entity_icon(entity.entity_type)
        print(f"   {i}. {icon} {entity.canonical_name}")
        print(f"      └── Image Quality: {entity.image_quality_score:.2f} | "
              f"Popularity: {entity.semantic_info.popularity_score:.2f}")
        print(f"      └── Best Search: '{entity.search_queries[0] if entity.search_queries else entity.canonical_name}'")
    
    # Entity recognition accuracy
    found_entities = [e.text.lower() for e in results['entities']]
    expected_lower = [e.lower() for e in expected_entities]
    matches = sum(1 for expected in expected_lower if any(expected in found for found in found_entities))
    accuracy = matches / len(expected_entities) if expected_entities else 1.0
    
    print(f"\n✅ Entity Recognition Accuracy: {accuracy:.1%} ({matches}/{len(expected_entities)})")
    
    # Image potential distribution
    potential_dist = results['enrichment_stats']['image_potential_distribution']
    print(f"\n🖼️  Image Potential Distribution:")
    for potential, count in potential_dist.items():
        print(f"   {potential.title()}: {count} entities")
    
    # Semantic insights
    semantic = results['semantic_summary']
    print(f"\n🧠 Semantic Insights:")
    print(f"   Topics: {', '.join(semantic['main_topics'])}")
    print(f"   Sentiment: {semantic['sentiment']} ({semantic['sentiment_score']:+.2f})")
    print(f"   Complexity: {semantic['complexity_score']:.2f} | Coherence: {semantic['coherence_score']:.2f}")
    
    # Emphasis effectiveness
    emphasized_count = len(results['emphasized_words'])
    total_words = len(results['emphasis_results'])
    emphasis_rate = emphasized_count / total_words if total_words > 0 else 0
    
    print(f"\n🎯 Emphasis Detection:")
    print(f"   {emphasized_count}/{total_words} words emphasized ({emphasis_rate:.1%})")
    if results['emphasized_words']:
        print(f"   Emphasized: {', '.join(results['emphasized_words'][:5])}")

def get_entity_icon(entity_type):
    """Get emoji icon for entity type."""
    icons = {
        'PERSON': '👤',
        'ORGANIZATION': '🏢', 
        'LOCATION': '📍',
        'PRODUCT': '📱',
        'BRAND': '🏷️',
        'EVENT': '📅',
        'TECHNOLOGY': '💻',
        'CONCEPT': '💡',
        'CURRENCY': '💰'
    }
    return icons.get(entity_type, '📄')

def create_synthetic_audio(text):
    """Create synthetic audio for emphasis detection testing."""
    import numpy as np
    
    # Simple synthetic audio
    duration = len(text.split()) * 0.5  # 0.5 seconds per word
    sample_rate = 16000
    total_samples = int(duration * sample_rate)
    
    # Generate speech-like audio
    t = np.linspace(0, duration, total_samples)
    audio = 0.3 * np.sin(2 * np.pi * 150 * t)  # Base frequency
    audio += 0.1 * np.sin(2 * np.pi * 300 * t)  # Harmonic
    audio += np.random.normal(0, 0.05, total_samples)  # Noise
    
    return audio.astype(np.float32)

def create_transcription(text):
    """Create word-level transcription data."""
    words = text.split()
    transcription = {
        'text': text,
        'word_timestamps': []
    }
    
    current_time = 0.0
    for word in words:
        word_duration = 0.4 + (len(word) * 0.05)  # Duration based on word length
        
        transcription['word_timestamps'].append({
            'word': word.rstrip('.,!?'),  # Remove punctuation
            'start': current_time,
            'end': current_time + word_duration
        })
        
        current_time += word_duration + 0.1  # 0.1s pause between words
    
    return transcription

if __name__ == "__main__":
    success = test_nlp_system()
    print(f"\n{'🎉 Success!' if success else '❌ Failed!'}")
    sys.exit(0 if success else 1) 