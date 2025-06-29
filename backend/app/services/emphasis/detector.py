"""
Multi-Modal Emphasis Detection

This module combines acoustic, prosodic, and linguistic analysis to provide
comprehensive emphasis detection using machine learning scoring and fusion techniques.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

from .acoustic_analyzer import AcousticAnalyzer
from .prosodic_analyzer import ProsodicAnalyzer
from .linguistic_analyzer import LinguisticAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class EmphasisResult:
    """Result of emphasis detection for a word."""
    word: str
    start: float
    end: float
    emphasis_score: float
    confidence: float
    is_emphasized: bool
    analysis_details: Dict
    
class MultiModalEmphasisDetector:
    """
    Multi-modal emphasis detector that combines multiple analysis methods.
    
    Integrates:
    - Acoustic analysis (volume, pitch, spectral features)
    - Prosodic analysis (rhythm, stress, timing)
    - Linguistic analysis (semantic importance, syntax)
    - Machine learning fusion and confidence scoring
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
        # Initialize analyzers
        self.acoustic_analyzer = AcousticAnalyzer(sample_rate)
        self.prosodic_analyzer = ProsodicAnalyzer(sample_rate)
        self.linguistic_analyzer = LinguisticAnalyzer()
        
        # Fusion weights (optimized for image-findable content)
        self.fusion_weights = {
            'acoustic': 0.25,     # Lower weight - volume doesn't indicate image relevance
            'prosodic': 0.25,     # Lower weight - rhythm doesn't indicate image relevance  
            'linguistic': 0.50    # Higher weight - entities/nouns are perfect for images
        }
        
        # Emphasis threshold
        self.emphasis_threshold = 0.6
        
        # Confidence calibration parameters
        self.confidence_alpha = 2.0  # Sharpening parameter
        self.confidence_beta = 0.1   # Minimum confidence
        
    def detect_emphasis(
        self, 
        audio_data: np.ndarray, 
        transcription_result: Dict
    ) -> List[EmphasisResult]:
        """
        Detect emphasized words using multi-modal analysis.
        
        Args:
            audio_data: Audio signal
            transcription_result: Result from transcription service with word timestamps
            
        Returns:
            List of EmphasisResult objects
        """
        try:
            word_timestamps = transcription_result['word_timestamps']
            full_text = transcription_result['text']
            
            if not word_timestamps:
                logger.warning("No word timestamps provided for emphasis detection")
                return []
            
            # Perform individual analyses
            acoustic_results = self._run_acoustic_analysis(audio_data, word_timestamps)
            prosodic_results = self._run_prosodic_analysis(audio_data, word_timestamps)
            linguistic_results = self._run_linguistic_analysis(full_text, word_timestamps)
            
            # Fuse results
            emphasis_results = self._fuse_analysis_results(
                acoustic_results, prosodic_results, linguistic_results
            )
            
            return emphasis_results
            
        except Exception as e:
            logger.error(f"Error in multi-modal emphasis detection: {e}")
            raise
    
    def _run_acoustic_analysis(
        self, 
        audio_data: np.ndarray, 
        word_timestamps: List[Dict]
    ) -> List[Dict]:
        """Run acoustic analysis."""
        try:
            return self.acoustic_analyzer.analyze_emphasis(audio_data, word_timestamps)
        except Exception as e:
            logger.error(f"Acoustic analysis failed: {e}")
            # Return fallback results
            return [{'word': w['word'], 'start': w['start'], 'end': w['end'], 
                    'emphasis_scores': {'total_score': 0.0}, 'is_emphasized': False}
                   for w in word_timestamps]
    
    def _run_prosodic_analysis(
        self, 
        audio_data: np.ndarray, 
        word_timestamps: List[Dict]
    ) -> List[Dict]:
        """Run prosodic analysis."""
        try:
            return self.prosodic_analyzer.analyze_prosody(audio_data, word_timestamps)
        except Exception as e:
            logger.error(f"Prosodic analysis failed: {e}")
            # Return fallback results
            return [{'word': w['word'], 'start': w['start'], 'end': w['end'], 
                    'prosodic_scores': {'total_prosodic_score': 0.0}, 
                    'is_prosodically_emphasized': False}
                   for w in word_timestamps]
    
    def _run_linguistic_analysis(
        self, 
        text: str, 
        word_timestamps: List[Dict]
    ) -> List[Dict]:
        """Run linguistic analysis."""
        try:
            return self.linguistic_analyzer.analyze_linguistic_emphasis(text, word_timestamps)
        except Exception as e:
            logger.error(f"Linguistic analysis failed: {e}")
            # Return fallback results
            return [{'word': w['word'], 'start': w['start'], 'end': w['end'], 
                    'linguistic_scores': {'total_linguistic_score': 0.0}, 
                    'is_linguistically_emphasized': False}
                   for w in word_timestamps]
    
    def _fuse_analysis_results(
        self, 
        acoustic_results: List[Dict],
        prosodic_results: List[Dict],
        linguistic_results: List[Dict]
    ) -> List[EmphasisResult]:
        """Fuse results from all analysis methods."""
        if not (len(acoustic_results) == len(prosodic_results) == len(linguistic_results)):
            logger.warning("Mismatch in number of analysis results")
            min_length = min(len(acoustic_results), len(prosodic_results), len(linguistic_results))
            acoustic_results = acoustic_results[:min_length]
            prosodic_results = prosodic_results[:min_length]
            linguistic_results = linguistic_results[:min_length]
        
        fused_results = []
        
        for i in range(len(acoustic_results)):
            acoustic = acoustic_results[i]
            prosodic = prosodic_results[i]
            linguistic = linguistic_results[i]
            
            # Extract scores
            acoustic_score = acoustic.get('emphasis_scores', {}).get('total_score', 0.0)
            prosodic_score = prosodic.get('prosodic_scores', {}).get('total_prosodic_score', 0.0)
            linguistic_score = linguistic.get('linguistic_scores', {}).get('total_linguistic_score', 0.0)
            
            # Normalize scores to ensure they're in [0, 1] range
            acoustic_score = max(0.0, min(1.0, acoustic_score))
            prosodic_score = max(0.0, min(1.0, prosodic_score))
            linguistic_score = max(0.0, min(1.0, linguistic_score))
            
            # Calculate fused emphasis score
            fused_score = self._calculate_fused_score(
                acoustic_score, prosodic_score, linguistic_score
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                acoustic_score, prosodic_score, linguistic_score, fused_score
            )
            
            # Determine emphasis
            is_emphasized = fused_score >= self.emphasis_threshold
            
            # Create detailed analysis
            analysis_details = {
                'acoustic': {
                    'score': acoustic_score,
                    'details': acoustic.get('emphasis_scores', {}),
                    'features': acoustic.get('acoustic_features', {})
                },
                'prosodic': {
                    'score': prosodic_score,
                    'details': prosodic.get('prosodic_scores', {}),
                    'duration': prosodic.get('duration', 0.0)
                },
                'linguistic': {
                    'score': linguistic_score,
                    'details': linguistic.get('linguistic_scores', {}),
                    'spacy_features': linguistic.get('spacy_features', {})
                },
                'fusion': {
                    'weights': self.fusion_weights,
                    'individual_scores': {
                        'acoustic': acoustic_score,
                        'prosodic': prosodic_score,
                        'linguistic': linguistic_score
                    }
                }
            }
            
            result = EmphasisResult(
                word=acoustic['word'],
                start=acoustic['start'],
                end=acoustic['end'],
                emphasis_score=fused_score,
                confidence=confidence,
                is_emphasized=is_emphasized,
                analysis_details=analysis_details
            )
            
            fused_results.append(result)
        
        return fused_results
    
    def _calculate_fused_score(
        self, 
        acoustic_score: float, 
        prosodic_score: float, 
        linguistic_score: float
    ) -> float:
        """Calculate fused emphasis score using weighted combination."""
        # Basic weighted average
        basic_score = (
            acoustic_score * self.fusion_weights['acoustic'] +
            prosodic_score * self.fusion_weights['prosodic'] +
            linguistic_score * self.fusion_weights['linguistic']
        )
        
        # Apply non-linear fusion for better discrimination
        # Use geometric mean for conservative fusion when scores disagree
        geometric_mean = (acoustic_score * prosodic_score * linguistic_score) ** (1/3)
        
        # Combine linear and geometric fusion
        alpha = 0.7  # Weight for linear fusion
        fused_score = alpha * basic_score + (1 - alpha) * geometric_mean
        
        # Apply sigmoid-like transformation for better calibration
        # This helps differentiate between medium and high emphasis
        calibrated_score = self._sigmoid_calibration(fused_score)
        
        return max(0.0, min(1.0, calibrated_score))
    
    def _sigmoid_calibration(self, score: float) -> float:
        """Apply sigmoid-like calibration to improve score distribution."""
        # Sigmoid transformation: f(x) = 1 / (1 + exp(-k*(x - 0.5)))
        k = 6  # Steepness parameter
        centered_score = score - 0.5
        calibrated = 1.0 / (1.0 + np.exp(-k * centered_score))
        return calibrated
    
    def _calculate_confidence(
        self, 
        acoustic_score: float, 
        prosodic_score: float, 
        linguistic_score: float,
        fused_score: float
    ) -> float:
        """Calculate confidence in the emphasis detection."""
        scores = [acoustic_score, prosodic_score, linguistic_score]
        
        # Measure agreement between different modalities
        score_variance = np.var(scores)
        agreement = 1.0 / (1.0 + score_variance * 10)  # High variance = low agreement
        
        # Measure certainty (distance from decision boundary)
        decision_certainty = abs(fused_score - self.emphasis_threshold)
        certainty = min(1.0, decision_certainty * 2.0)
        
        # Overall confidence combines agreement and certainty
        base_confidence = (agreement + certainty) / 2.0
        
        # Apply confidence calibration
        confidence = self.confidence_beta + (1 - self.confidence_beta) * (base_confidence ** self.confidence_alpha)
        
        return max(0.0, min(1.0, confidence))
    
    def get_emphasis_statistics(self, results: List[EmphasisResult]) -> Dict:
        """Calculate statistics about emphasis detection results."""
        if not results:
            return {}
        
        emphasized_words = [r for r in results if r.is_emphasized]
        
        stats = {
            'total_words': len(results),
            'emphasized_words': len(emphasized_words),
            'emphasis_percentage': len(emphasized_words) / len(results) * 100,
            'average_emphasis_score': np.mean([r.emphasis_score for r in results]),
            'average_confidence': np.mean([r.confidence for r in results]),
            'score_distribution': {
                'min': float(np.min([r.emphasis_score for r in results])),
                'max': float(np.max([r.emphasis_score for r in results])),
                'std': float(np.std([r.emphasis_score for r in results]))
            }
        }
        
        # Modality-specific statistics
        acoustic_scores = [r.analysis_details['acoustic']['score'] for r in results]
        prosodic_scores = [r.analysis_details['prosodic']['score'] for r in results]
        linguistic_scores = [r.analysis_details['linguistic']['score'] for r in results]
        
        stats['modality_scores'] = {
            'acoustic': {
                'mean': float(np.mean(acoustic_scores)),
                'std': float(np.std(acoustic_scores))
            },
            'prosodic': {
                'mean': float(np.mean(prosodic_scores)),
                'std': float(np.std(prosodic_scores))
            },
            'linguistic': {
                'mean': float(np.mean(linguistic_scores)),
                'std': float(np.std(linguistic_scores))
            }
        }
        
        return stats
    
    def adjust_fusion_weights(self, acoustic_weight: float, prosodic_weight: float, linguistic_weight: float):
        """Adjust fusion weights for different emphasis detection strategies."""
        total = acoustic_weight + prosodic_weight + linguistic_weight
        if total > 0:
            self.fusion_weights = {
                'acoustic': acoustic_weight / total,
                'prosodic': prosodic_weight / total,
                'linguistic': linguistic_weight / total
            }
    
    def set_emphasis_threshold(self, threshold: float):
        """Set the threshold for emphasis classification."""
        self.emphasis_threshold = max(0.0, min(1.0, threshold))
    
    def export_results_for_training(self, results: List[EmphasisResult]) -> List[Dict]:
        """Export results in format suitable for machine learning training."""
        training_data = []
        
        for result in results:
            feature_vector = {
                # Acoustic features
                'acoustic_score': result.analysis_details['acoustic']['score'],
                'volume_score': result.analysis_details['acoustic']['details'].get('volume_score', 0.0),
                'pitch_score': result.analysis_details['acoustic']['details'].get('pitch_score', 0.0),
                'spectral_score': result.analysis_details['acoustic']['details'].get('spectral_score', 0.0),
                
                # Prosodic features
                'prosodic_score': result.analysis_details['prosodic']['score'],
                'duration_score': result.analysis_details['prosodic']['details'].get('duration_score', 0.0),
                'pause_score': result.analysis_details['prosodic']['details'].get('pause_score', 0.0),
                'stress_score': result.analysis_details['prosodic']['details'].get('stress_score', 0.0),
                
                # Linguistic features
                'linguistic_score': result.analysis_details['linguistic']['score'],
                'pos_score': result.analysis_details['linguistic']['details'].get('pos_score', 0.0),
                'entity_score': result.analysis_details['linguistic']['details'].get('entity_score', 0.0),
                'keyword_score': result.analysis_details['linguistic']['details'].get('keyword_score', 0.0),
                
                # Metadata
                'word': result.word,
                'word_length': len(result.word),
                'word_duration': result.end - result.start,
                
                # Target (for supervised learning)
                'is_emphasized': result.is_emphasized,
                'emphasis_score': result.emphasis_score,
                'confidence': result.confidence
            }
            
            training_data.append(feature_vector)
        
        return training_data 