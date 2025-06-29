"""
Acoustic Analysis for Emphasis Detection

This module provides sophisticated acoustic analysis to detect emphasized speech
through volume changes, pitch variations, spectral characteristics, and temporal patterns.
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.stats import zscore
import logging

from ..audio.processor import AudioProcessor

logger = logging.getLogger(__name__)

class AcousticAnalyzer:
    """
    Analyzes acoustic features to detect speech emphasis.
    
    Uses multiple acoustic cues:
    - Volume/intensity changes
    - Fundamental frequency (pitch) variations
    - Spectral characteristics
    - Temporal dynamics
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.frame_length = 2048
        self.hop_length = 512
        self.n_mels = 128
        
        # Emphasis detection thresholds
        self.volume_threshold = 1.5  # Standard deviations above mean
        self.pitch_threshold = 1.2   # Standard deviations above mean
        self.spectral_threshold = 1.3
        
    def analyze_emphasis(
        self, 
        audio_data: np.ndarray, 
        word_timestamps: List[Dict]
    ) -> List[Dict]:
        """
        Analyze acoustic features to detect emphasized words.
        
        Args:
            audio_data: Audio signal
            word_timestamps: List of word timing information
            
        Returns:
            List of emphasis analysis results for each word
        """
        try:
            # Extract acoustic features
            features = self._extract_acoustic_features(audio_data)
            
            # Analyze each word
            emphasis_results = []
            for word_info in word_timestamps:
                word_analysis = self._analyze_word_emphasis(
                    features, word_info
                )
                emphasis_results.append(word_analysis)
                
            return emphasis_results
            
        except Exception as e:
            logger.error(f"Error in acoustic emphasis analysis: {e}")
            raise
    
    def _extract_acoustic_features(self, audio_data: np.ndarray) -> Dict:
        """Extract comprehensive acoustic features from audio."""
        features = {}
        
        # 1. Volume/Intensity Analysis
        features['rms'] = librosa.feature.rms(
            y=audio_data,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]
        
        # 2. Fundamental Frequency (Pitch) Analysis
        features['f0'] = self._extract_pitch(audio_data)
        
        # 3. Spectral Features
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=audio_data,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )[0]
        
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=audio_data,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )[0]
        
        features['spectral_contrast'] = librosa.feature.spectral_contrast(
            y=audio_data,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # 4. Mel-frequency Cepstral Coefficients (MFCCs)
        features['mfcc'] = librosa.feature.mfcc(
            y=audio_data,
            sr=self.sample_rate,
            n_mfcc=13,
            hop_length=self.hop_length
        )
        
        # 5. Zero Crossing Rate (speech activity indicator)
        features['zcr'] = librosa.feature.zero_crossing_rate(
            y=audio_data,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]
        
        # 6. Temporal dynamics
        features['tempo'] = self._extract_tempo_features(audio_data)
        
        return features
    
    def _extract_pitch(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract fundamental frequency using multiple methods."""
        try:
            # Method 1: Harmonic-percussive separation + pitch tracking
            y_harmonic, _ = librosa.effects.hpss(audio_data)
            
            # Method 2: PYIN algorithm for robust pitch extraction
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y_harmonic,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            # Fill unvoiced regions with interpolation
            f0_filled = np.copy(f0)
            unvoiced_indices = ~voiced_flag
            
            if np.any(voiced_flag):
                # Linear interpolation for short gaps
                voiced_indices = np.where(voiced_flag)[0]
                if len(voiced_indices) > 1:
                    f0_filled = np.interp(
                        np.arange(len(f0)),
                        voiced_indices,
                        f0[voiced_indices]
                    )
            
            return f0_filled
            
        except Exception as e:
            logger.warning(f"Pitch extraction failed: {e}")
            # Fallback to simple autocorrelation method
            return self._simple_pitch_extraction(audio_data)
    
    def _simple_pitch_extraction(self, audio_data: np.ndarray) -> np.ndarray:
        """Simple pitch extraction using autocorrelation."""
        hop_length = self.hop_length
        frame_length = self.frame_length
        
        n_frames = 1 + (len(audio_data) - frame_length) // hop_length
        f0 = np.zeros(n_frames)
        
        for i in range(n_frames):
            start = i * hop_length
            end = start + frame_length
            frame = audio_data[start:end]
            
            # Autocorrelation
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find pitch period
            min_period = int(self.sample_rate / 500)  # 500 Hz max
            max_period = int(self.sample_rate / 50)   # 50 Hz min
            
            if len(autocorr) > max_period:
                autocorr_region = autocorr[min_period:max_period]
                if len(autocorr_region) > 0:
                    period = np.argmax(autocorr_region) + min_period
                    f0[i] = self.sample_rate / period
        
        return f0
    
    def _extract_tempo_features(self, audio_data: np.ndarray) -> Dict:
        """Extract temporal dynamics and rhythm features."""
        try:
            # Onset strength
            onset_envelope = librosa.onset.onset_strength(
                y=audio_data,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            # Tempo estimation
            tempo, beats = librosa.beat.beat_track(
                onset_envelope=onset_envelope,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            return {
                'tempo': tempo,
                'onset_strength': onset_envelope,
                'beats': beats
            }
            
        except Exception as e:
            logger.warning(f"Tempo extraction failed: {e}")
            return {
                'tempo': 120.0,  # Default tempo
                'onset_strength': np.zeros(len(audio_data) // self.hop_length),
                'beats': np.array([])
            }
    
    def _analyze_word_emphasis(
        self, 
        features: Dict, 
        word_info: Dict
    ) -> Dict:
        """Analyze emphasis for a specific word."""
        start_time = word_info['start']
        end_time = word_info['end']
        word_text = word_info['word']
        
        # Convert time to frame indices
        start_frame = int(start_time * self.sample_rate / self.hop_length)
        end_frame = int(end_time * self.sample_rate / self.hop_length)
        
        # Ensure valid frame range
        max_frames = len(features['rms'])
        start_frame = max(0, min(start_frame, max_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, max_frames))
        
        # Extract features for this word
        word_features = self._extract_word_features(features, start_frame, end_frame)
        
        # Calculate emphasis scores
        emphasis_scores = self._calculate_emphasis_scores(word_features, features)
        
        return {
            'word': word_text,
            'start': start_time,
            'end': end_time,
            'acoustic_features': word_features,
            'emphasis_scores': emphasis_scores,
            'is_emphasized': emphasis_scores['total_score'] > 0.6
        }
    
    def _extract_word_features(
        self, 
        features: Dict, 
        start_frame: int, 
        end_frame: int
    ) -> Dict:
        """Extract acoustic features for a specific word segment."""
        word_features = {}
        
        # Volume/Intensity
        rms_segment = features['rms'][start_frame:end_frame]
        word_features['volume_mean'] = np.mean(rms_segment)
        word_features['volume_max'] = np.max(rms_segment)
        word_features['volume_std'] = np.std(rms_segment)
        
        # Pitch
        f0_segment = features['f0'][start_frame:end_frame]
        valid_f0 = f0_segment[f0_segment > 0]
        if len(valid_f0) > 0:
            word_features['pitch_mean'] = np.mean(valid_f0)
            word_features['pitch_max'] = np.max(valid_f0)
            word_features['pitch_range'] = np.max(valid_f0) - np.min(valid_f0)
            word_features['pitch_std'] = np.std(valid_f0)
        else:
            word_features['pitch_mean'] = 0
            word_features['pitch_max'] = 0
            word_features['pitch_range'] = 0
            word_features['pitch_std'] = 0
        
        # Spectral features
        spectral_centroid_segment = features['spectral_centroid'][start_frame:end_frame]
        word_features['spectral_centroid_mean'] = np.mean(spectral_centroid_segment)
        word_features['spectral_centroid_std'] = np.std(spectral_centroid_segment)
        
        spectral_rolloff_segment = features['spectral_rolloff'][start_frame:end_frame]
        word_features['spectral_rolloff_mean'] = np.mean(spectral_rolloff_segment)
        
        # Spectral contrast
        spectral_contrast_segment = features['spectral_contrast'][:, start_frame:end_frame]
        word_features['spectral_contrast_mean'] = np.mean(spectral_contrast_segment, axis=1)
        
        # MFCCs
        mfcc_segment = features['mfcc'][:, start_frame:end_frame]
        word_features['mfcc_mean'] = np.mean(mfcc_segment, axis=1)
        word_features['mfcc_std'] = np.std(mfcc_segment, axis=1)
        
        # Zero crossing rate
        zcr_segment = features['zcr'][start_frame:end_frame]
        word_features['zcr_mean'] = np.mean(zcr_segment)
        
        return word_features
    
    def _calculate_emphasis_scores(
        self, 
        word_features: Dict, 
        global_features: Dict
    ) -> Dict:
        """Calculate emphasis scores based on acoustic features."""
        scores = {}
        
        # 1. Volume emphasis score
        global_volume_mean = np.mean(global_features['rms'])
        global_volume_std = np.std(global_features['rms'])
        
        if global_volume_std > 0:
            volume_z_score = (word_features['volume_mean'] - global_volume_mean) / global_volume_std
            scores['volume_score'] = max(0, min(1, (volume_z_score - 0.5) / 2.0))
        else:
            scores['volume_score'] = 0
        
        # 2. Pitch emphasis score
        valid_f0 = global_features['f0'][global_features['f0'] > 0]
        if len(valid_f0) > 0 and word_features['pitch_mean'] > 0:
            global_pitch_mean = np.mean(valid_f0)
            global_pitch_std = np.std(valid_f0)
            
            if global_pitch_std > 0:
                pitch_z_score = (word_features['pitch_mean'] - global_pitch_mean) / global_pitch_std
                scores['pitch_score'] = max(0, min(1, (pitch_z_score - 0.3) / 1.5))
                
                # Pitch range emphasis
                pitch_range_score = min(1, word_features['pitch_range'] / (global_pitch_std * 2))
                scores['pitch_range_score'] = pitch_range_score
            else:
                scores['pitch_score'] = 0
                scores['pitch_range_score'] = 0
        else:
            scores['pitch_score'] = 0
            scores['pitch_range_score'] = 0
        
        # 3. Spectral emphasis score
        global_spectral_mean = np.mean(global_features['spectral_centroid'])
        global_spectral_std = np.std(global_features['spectral_centroid'])
        
        if global_spectral_std > 0:
            spectral_z_score = (word_features['spectral_centroid_mean'] - global_spectral_mean) / global_spectral_std
            scores['spectral_score'] = max(0, min(1, (spectral_z_score - 0.3) / 1.5))
        else:
            scores['spectral_score'] = 0
        
        # 4. Spectral contrast emphasis
        global_contrast_mean = np.mean(global_features['spectral_contrast'], axis=1)
        contrast_difference = np.abs(word_features['spectral_contrast_mean'] - global_contrast_mean)
        scores['contrast_score'] = min(1, np.mean(contrast_difference) / 500)
        
        # 5. Combined total score with weights
        weights = {
            'volume_score': 0.25,
            'pitch_score': 0.25,
            'pitch_range_score': 0.15,
            'spectral_score': 0.20,
            'contrast_score': 0.15
        }
        
        total_score = sum(scores.get(key, 0) * weight for key, weight in weights.items())
        scores['total_score'] = total_score
        
        return scores 