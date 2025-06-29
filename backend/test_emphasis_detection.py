#!/usr/bin/env python3
"""
Test script for Multi-Modal Emphasis Detection

This script tests the complete emphasis detection pipeline including:
- Acoustic analysis
- Prosodic analysis  
- Linguistic analysis
- Multi-modal fusion
- Result validation and statistics
"""

import sys
import os
import numpy as np
import logging
from typing import Dict, List

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_emphasis_detection():
    """Test the complete emphasis detection pipeline."""
    print("🎬 Testing Multi-Modal Emphasis Detection System")
    print("=" * 60)
    
    try:
        # Import the emphasis detection components
        from backend.app.services.emphasis import MultiModalEmphasisDetector
        
        # Create synthetic test data
        test_data = create_test_data()
        
        # Initialize the detector
        detector = MultiModalEmphasisDetector(sample_rate=16000)
        print("✅ Successfully initialized MultiModalEmphasisDetector")
        
        # Test each component
        test_results = []
        
        for i, (audio_data, transcription_result, expected_emphasis) in enumerate(test_data):
            print(f"\n📝 Test Case {i+1}: {transcription_result['text']}")
            print("-" * 40)
            
            try:
                # Run emphasis detection
                results = detector.detect_emphasis(audio_data, transcription_result)
                
                # Validate results
                validation_result = validate_results(results, expected_emphasis)
                test_results.append(validation_result)
                
                # Print results
                print_results(results, validation_result)
                
                # Get statistics
                stats = detector.get_emphasis_statistics(results)
                print_statistics(stats)
                
            except Exception as e:
                logger.error(f"Test case {i+1} failed: {e}")
                test_results.append({'success': False, 'error': str(e)})
        
        # Print overall test summary
        print_test_summary(test_results)
        
        # Test configuration adjustments
        test_configuration_adjustments(detector)
        
        print("\n🎉 Emphasis detection testing completed!")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Make sure all dependencies are installed in the virtual environment")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

def create_test_data():
    """Create synthetic test data for emphasis detection."""
    test_cases = []
    
    # Test Case 1: Clear emphasis on important nouns
    audio1 = create_synthetic_audio(
        duration=5.0,
        emphasis_times=[(1.0, 1.5), (3.0, 3.8)],  # "AMAZING" and "BREAKTHROUGH"
        sample_rate=16000
    )
    transcription1 = {
        'text': "This is an AMAZING new BREAKTHROUGH in technology",
        'word_timestamps': [
            {'word': 'This', 'start': 0.0, 'end': 0.3},
            {'word': 'is', 'start': 0.3, 'end': 0.5},
            {'word': 'an', 'start': 0.5, 'end': 0.7},
            {'word': 'AMAZING', 'start': 1.0, 'end': 1.5},
            {'word': 'new', 'start': 2.0, 'end': 2.3},
            {'word': 'BREAKTHROUGH', 'start': 3.0, 'end': 3.8},
            {'word': 'in', 'start': 4.0, 'end': 4.2},
            {'word': 'technology', 'start': 4.2, 'end': 4.8}
        ]
    }
    expected1 = {'AMAZING': True, 'BREAKTHROUGH': True, 'technology': True}
    test_cases.append((audio1, transcription1, expected1))
    
    # Test Case 2: Numbers and proper nouns emphasis
    audio2 = create_synthetic_audio(
        duration=4.0,
        emphasis_times=[(0.5, 1.0), (2.0, 2.5)],  # "Apple" and "million"
        sample_rate=16000
    )
    transcription2 = {
        'text': "Apple sold fifty million devices last year",
        'word_timestamps': [
            {'word': 'Apple', 'start': 0.5, 'end': 1.0},
            {'word': 'sold', 'start': 1.2, 'end': 1.5},
            {'word': 'fifty', 'start': 1.7, 'end': 2.0},
            {'word': 'million', 'start': 2.0, 'end': 2.5},
            {'word': 'devices', 'start': 2.7, 'end': 3.1},
            {'word': 'last', 'start': 3.3, 'end': 3.5},
            {'word': 'year', 'start': 3.5, 'end': 3.8}
        ]
    }
    expected2 = {'Apple': True, 'million': True, 'fifty': True}
    test_cases.append((audio2, transcription2, expected2))
    
    # Test Case 3: Discourse markers and adjectives
    audio3 = create_synthetic_audio(
        duration=6.0,
        emphasis_times=[(0.8, 1.2), (3.5, 4.2)],  # "absolutely" and "incredible"
        sample_rate=16000
    )
    transcription3 = {
        'text': "This is absolutely the most incredible result we have seen",
        'word_timestamps': [
            {'word': 'This', 'start': 0.0, 'end': 0.2},
            {'word': 'is', 'start': 0.2, 'end': 0.4},
            {'word': 'absolutely', 'start': 0.8, 'end': 1.2},
            {'word': 'the', 'start': 1.5, 'end': 1.7},
            {'word': 'most', 'start': 1.9, 'end': 2.1},
            {'word': 'incredible', 'start': 3.5, 'end': 4.2},
            {'word': 'result', 'start': 4.5, 'end': 4.8},
            {'word': 'we', 'start': 5.0, 'end': 5.1},
            {'word': 'have', 'start': 5.1, 'end': 5.3},
            {'word': 'seen', 'start': 5.3, 'end': 5.6}
        ]
    }
    expected3 = {'absolutely': True, 'incredible': True, 'result': True}
    test_cases.append((audio3, transcription3, expected3))
    
    return test_cases

def create_synthetic_audio(duration: float, emphasis_times: List[tuple], sample_rate: int = 16000):
    """Create synthetic audio with emphasis at specified times."""
    total_samples = int(duration * sample_rate)
    audio = np.random.normal(0, 0.1, total_samples)  # Base noise
    
    # Add synthetic speech-like patterns
    t = np.linspace(0, duration, total_samples)
    
    # Base speech frequency (fundamental frequency around 150 Hz)
    base_freq = 150
    speech_signal = 0.3 * np.sin(2 * np.pi * base_freq * t)
    
    # Add formants (simplified)
    formant1 = 0.15 * np.sin(2 * np.pi * 800 * t)
    formant2 = 0.1 * np.sin(2 * np.pi * 1200 * t)
    
    audio += speech_signal + formant1 + formant2
    
    # Add emphasis (increased amplitude and frequency modulation)
    for start_time, end_time in emphasis_times:
        start_idx = int(start_time * sample_rate)
        end_idx = int(end_time * sample_rate)
        
        if start_idx < total_samples and end_idx <= total_samples:
            # Increase amplitude for emphasis
            emphasis_factor = 2.5
            audio[start_idx:end_idx] *= emphasis_factor
            
            # Add pitch variation for emphasis
            t_segment = np.linspace(start_time, end_time, end_idx - start_idx)
            pitch_mod = 0.5 * np.sin(2 * np.pi * 200 * t_segment)  # Pitch rise
            audio[start_idx:end_idx] += pitch_mod
    
    return audio.astype(np.float32)

def validate_results(results, expected_emphasis: Dict) -> Dict:
    """Validate emphasis detection results against expected outcomes."""
    validation = {
        'success': True,
        'correct_predictions': 0,
        'total_expected': len(expected_emphasis),
        'total_detected': sum(1 for r in results if r.is_emphasized),
        'details': {}
    }
    
    for result in results:
        word = result.word
        is_emphasized = result.is_emphasized
        expected = expected_emphasis.get(word, False)
        
        is_correct = (is_emphasized == expected)
        if is_correct:
            validation['correct_predictions'] += 1
        
        validation['details'][word] = {
            'detected': is_emphasized,
            'expected': expected,
            'correct': is_correct,
            'score': result.emphasis_score,
            'confidence': result.confidence
        }
    
    validation['accuracy'] = validation['correct_predictions'] / len(results) if results else 0
    
    return validation

def print_results(results, validation_result: Dict):
    """Print emphasis detection results."""
    print("Emphasis Detection Results:")
    for result in results:
        word = result.word
        details = validation_result['details'][word]
        status = "✅" if details['correct'] else "❌"
        emphasis_indicator = "🔴" if result.is_emphasized else "⚪"
        
        print(f"  {status} {emphasis_indicator} {word:15} | Score: {result.emphasis_score:.3f} | Conf: {result.confidence:.3f}")

def print_statistics(stats: Dict):
    """Print emphasis detection statistics."""
    print(f"\nStatistics:")
    print(f"  Total words: {stats.get('total_words', 0)}")
    print(f"  Emphasized: {stats.get('emphasized_words', 0)} ({stats.get('emphasis_percentage', 0):.1f}%)")
    print(f"  Avg score: {stats.get('average_emphasis_score', 0):.3f}")
    print(f"  Avg confidence: {stats.get('average_confidence', 0):.3f}")
    
    modality_scores = stats.get('modality_scores', {})
    print(f"  Modality averages:")
    for modality, scores in modality_scores.items():
        print(f"    {modality.capitalize()}: {scores.get('mean', 0):.3f} (±{scores.get('std', 0):.3f})")

def print_test_summary(test_results: List[Dict]):
    """Print overall test summary."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    successful_tests = sum(1 for r in test_results if r.get('success', False))
    total_tests = len(test_results)
    
    print(f"Tests passed: {successful_tests}/{total_tests}")
    
    if successful_tests > 0:
        accuracies = [r.get('accuracy', 0) for r in test_results if r.get('success', False)]
        avg_accuracy = np.mean(accuracies) if accuracies else 0
        print(f"Average accuracy: {avg_accuracy:.3f}")
        
        total_correct = sum(r.get('correct_predictions', 0) for r in test_results)
        total_words = sum(len(r.get('details', {})) for r in test_results)
        overall_accuracy = total_correct / total_words if total_words > 0 else 0
        print(f"Overall accuracy: {overall_accuracy:.3f}")
    
    # Print failed tests
    failed_tests = [r for r in test_results if not r.get('success', False)]
    if failed_tests:
        print(f"\nFailed tests:")
        for i, result in enumerate(failed_tests):
            print(f"  Test {i+1}: {result.get('error', 'Unknown error')}")

def test_configuration_adjustments(detector):
    """Test configuration adjustments."""
    print("\n" + "=" * 60)
    print("TESTING CONFIGURATION ADJUSTMENTS")
    print("=" * 60)
    
    # Test weight adjustments
    print("Testing fusion weight adjustments...")
    original_weights = detector.fusion_weights.copy()
    
    # Emphasize acoustic features
    detector.adjust_fusion_weights(0.6, 0.2, 0.2)
    print(f"Acoustic emphasis weights: {detector.fusion_weights}")
    
    # Emphasize linguistic features
    detector.adjust_fusion_weights(0.2, 0.2, 0.6)
    print(f"Linguistic emphasis weights: {detector.fusion_weights}")
    
    # Restore original weights
    detector.fusion_weights = original_weights
    print(f"Restored original weights: {detector.fusion_weights}")
    
    # Test threshold adjustments
    print("\nTesting threshold adjustments...")
    original_threshold = detector.emphasis_threshold
    
    detector.set_emphasis_threshold(0.4)
    print(f"Lower threshold: {detector.emphasis_threshold}")
    
    detector.set_emphasis_threshold(0.8)
    print(f"Higher threshold: {detector.emphasis_threshold}")
    
    detector.set_emphasis_threshold(original_threshold)
    print(f"Restored threshold: {detector.emphasis_threshold}")

if __name__ == "__main__":
    success = test_emphasis_detection()
    sys.exit(0 if success else 1) 