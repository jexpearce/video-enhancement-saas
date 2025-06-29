"""
Prosodic Analysis for Emphasis Detection

This module analyzes prosodic features including rhythm, stress patterns,
timing, and speech flow to identify emphasized speech segments.
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.signal import find_peaks
import logging

logger = logging.getLogger(__name__)

class ProsodicAnalyzer:
    """
    Analyzes prosodic features to detect speech emphasis.
    
    Focuses on:
    - Rhythm and timing patterns
    - Stress patterns and syllable emphasis
    - Pause analysis
    - Speech rate variations
    - Intonation patterns
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.hop_length = 512
        self.frame_length = 2048
        
        # Prosodic analysis parameters
        self.min_pause_duration = 0.1  # Minimum pause duration in seconds
        self.stress_window = 0.5       # Window for stress analysis
        self.rhythm_window = 2.0       # Window for rhythm analysis
        
    def analyze_prosody(
        self, 
        audio_data: np.ndarray, 
        word_timestamps: List[Dict]
    ) -> List[Dict]:
        """
        Analyze prosodic features to detect emphasized words.
        
        Args:
            audio_data: Audio signal
            word_timestamps: List of word timing information
            
        Returns:
            List of prosodic analysis results for each word
        """
        try:
            # Extract prosodic features
            prosodic_features = self._extract_prosodic_features(audio_data, word_timestamps)
            
            # Analyze each word for prosodic emphasis
            prosody_results = []
            for i, word_info in enumerate(word_timestamps):
                word_prosody = self._analyze_word_prosody(
                    word_info, word_timestamps, prosodic_features, i
                )
                prosody_results.append(word_prosody)
                
            return prosody_results
            
        except Exception as e:
            logger.error(f"Error in prosodic analysis: {e}")
            raise
    
    def _extract_prosodic_features(
        self, 
        audio_data: np.ndarray, 
        word_timestamps: List[Dict]
    ) -> Dict:
        """Extract comprehensive prosodic features."""
        features = {}
        
        # 1. Energy envelope for rhythm analysis
        features['energy'] = self._extract_energy_envelope(audio_data)
        
        # 2. Pitch contour for intonation analysis
        features['pitch_contour'] = self._extract_pitch_contour(audio_data)
        
        # 3. Speech rate analysis
        features['speech_rate'] = self._analyze_speech_rate(word_timestamps)
        
        # 4. Pause analysis
        features['pauses'] = self._analyze_pauses(word_timestamps)
        
        # 5. Rhythm analysis
        features['rhythm'] = self._analyze_rhythm(audio_data, word_timestamps)
        
        # 6. Stress patterns
        features['stress_patterns'] = self._analyze_stress_patterns(
            audio_data, word_timestamps
        )
        
        return features
    
    def _extract_energy_envelope(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract energy envelope for rhythm analysis."""
        # Short-time energy
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.010 * self.sample_rate)    # 10ms hop
        
        energy = []
        for i in range(0, len(audio_data) - frame_length, hop_length):
            frame = audio_data[i:i + frame_length]
            frame_energy = np.sum(frame ** 2)
            energy.append(frame_energy)
        
        return np.array(energy)
    
    def _extract_pitch_contour(self, audio_data: np.ndarray) -> Dict:
        """Extract detailed pitch contour for intonation analysis."""
        try:
            # High-resolution pitch tracking
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_data,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate,
                hop_length=self.hop_length // 2  # Higher resolution
            )
            
            # Smooth pitch contour
            valid_f0 = f0[voiced_flag]
            if len(valid_f0) > 0:
                # Interpolate missing values
                f0_smooth = np.copy(f0)
                voiced_indices = np.where(voiced_flag)[0]
                if len(voiced_indices) > 1:
                    f0_smooth = np.interp(
                        np.arange(len(f0)),
                        voiced_indices,
                        f0[voiced_indices]
                    )
                
                # Calculate pitch derivatives for contour analysis
                pitch_derivative = np.gradient(f0_smooth)
                pitch_acceleration = np.gradient(pitch_derivative)
                
                return {
                    'f0': f0_smooth,
                    'voiced_flag': voiced_flag,
                    'voiced_probs': voiced_probs,
                    'pitch_derivative': pitch_derivative,
                    'pitch_acceleration': pitch_acceleration
                }
            else:
                return {
                    'f0': f0,
                    'voiced_flag': voiced_flag,
                    'voiced_probs': voiced_probs,
                    'pitch_derivative': np.zeros_like(f0),
                    'pitch_acceleration': np.zeros_like(f0)
                }
                
        except Exception as e:
            logger.warning(f"Pitch contour extraction failed: {e}")
            return {
                'f0': np.zeros(len(audio_data) // (self.hop_length // 2)),
                'voiced_flag': np.zeros(len(audio_data) // (self.hop_length // 2), dtype=bool),
                'voiced_probs': np.zeros(len(audio_data) // (self.hop_length // 2)),
                'pitch_derivative': np.zeros(len(audio_data) // (self.hop_length // 2)),
                'pitch_acceleration': np.zeros(len(audio_data) // (self.hop_length // 2))
            }
    
    def _analyze_speech_rate(self, word_timestamps: List[Dict]) -> Dict:
        """Analyze speech rate patterns."""
        if len(word_timestamps) < 2:
            return {'global_rate': 0, 'local_rates': [], 'rate_changes': []}
        
        # Calculate global speech rate (words per minute)
        total_duration = word_timestamps[-1]['end'] - word_timestamps[0]['start']
        global_rate = len(word_timestamps) / (total_duration / 60) if total_duration > 0 else 0
        
        # Calculate local speech rates (sliding window)
        local_rates = []
        rate_changes = []
        
        window_size = 5  # 5-word window
        for i in range(len(word_timestamps) - window_size + 1):
            window_words = word_timestamps[i:i + window_size]
            window_duration = window_words[-1]['end'] - window_words[0]['start']
            
            if window_duration > 0:
                local_rate = window_size / (window_duration / 60)
                local_rates.append(local_rate)
                
                # Calculate rate change
                if i > 0:
                    rate_change = local_rate - local_rates[i-1]
                    rate_changes.append(rate_change)
        
        return {
            'global_rate': global_rate,
            'local_rates': local_rates,
            'rate_changes': rate_changes
        }
    
    def _analyze_pauses(self, word_timestamps: List[Dict]) -> Dict:
        """Analyze pause patterns between words."""
        pauses = []
        pause_positions = []
        
        for i in range(len(word_timestamps) - 1):
            current_end = word_timestamps[i]['end']
            next_start = word_timestamps[i + 1]['start']
            pause_duration = next_start - current_end
            
            if pause_duration > self.min_pause_duration:
                pauses.append(pause_duration)
                pause_positions.append(i)
        
        # Analyze pause statistics
        if pauses:
            pause_stats = {
                'mean_pause': np.mean(pauses),
                'std_pause': np.std(pauses),
                'max_pause': np.max(pauses),
                'pause_count': len(pauses),
                'pause_positions': pause_positions,
                'pause_durations': pauses
            }
        else:
            pause_stats = {
                'mean_pause': 0,
                'std_pause': 0,
                'max_pause': 0,
                'pause_count': 0,
                'pause_positions': [],
                'pause_durations': []
            }
        
        return pause_stats
    
    def _analyze_rhythm(
        self, 
        audio_data: np.ndarray, 
        word_timestamps: List[Dict]
    ) -> Dict:
        """Analyze rhythmic patterns in speech."""
        try:
            # Extract onset strength for rhythm analysis
            onset_envelope = librosa.onset.onset_strength(
                y=audio_data,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            # Find rhythmic periods
            tempo, beats = librosa.beat.beat_track(
                onset_envelope=onset_envelope,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            # Analyze rhythm regularity
            if len(beats) > 1:
                beat_intervals = np.diff(beats) * self.hop_length / self.sample_rate
                rhythm_regularity = 1.0 / (1.0 + np.std(beat_intervals))
            else:
                rhythm_regularity = 0.0
            
            # Analyze word-level rhythm
            word_durations = [w['end'] - w['start'] for w in word_timestamps]
            if word_durations:
                duration_regularity = 1.0 / (1.0 + np.std(word_durations))
                mean_word_duration = np.mean(word_durations)
            else:
                duration_regularity = 0.0
                mean_word_duration = 0.0
            
            return {
                'tempo': tempo,
                'beats': beats,
                'rhythm_regularity': rhythm_regularity,
                'duration_regularity': duration_regularity,
                'mean_word_duration': mean_word_duration,
                'onset_envelope': onset_envelope
            }
            
        except Exception as e:
            logger.warning(f"Rhythm analysis failed: {e}")
            return {
                'tempo': 120.0,
                'beats': np.array([]),
                'rhythm_regularity': 0.0,
                'duration_regularity': 0.0,
                'mean_word_duration': 0.0,
                'onset_envelope': np.zeros(len(audio_data) // self.hop_length)
            }
    
    def _analyze_stress_patterns(
        self, 
        audio_data: np.ndarray, 
        word_timestamps: List[Dict]
    ) -> Dict:
        """Analyze stress patterns within and between words."""
        stress_patterns = {}
        
        # Extract RMS energy for stress analysis
        rms = librosa.feature.rms(
            y=audio_data,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]
        
        # Time axis for RMS
        time_frames = librosa.frames_to_time(
            np.arange(len(rms)),
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        word_stress_scores = []
        
        for word_info in word_timestamps:
            start_time = word_info['start']
            end_time = word_info['end']
            
            # Find RMS frames within word boundaries
            word_mask = (time_frames >= start_time) & (time_frames <= end_time)
            word_rms = rms[word_mask]
            
            if len(word_rms) > 0:
                # Calculate stress indicators
                word_energy = np.mean(word_rms)
                energy_peak = np.max(word_rms)
                energy_variation = np.std(word_rms)
                
                # Stress score based on energy characteristics
                stress_score = (energy_peak + energy_variation) / 2
                word_stress_scores.append(stress_score)
            else:
                word_stress_scores.append(0.0)
        
        # Normalize stress scores
        if word_stress_scores:
            mean_stress = np.mean(word_stress_scores)
            std_stress = np.std(word_stress_scores)
            if std_stress > 0:
                normalized_stress = [(s - mean_stress) / std_stress for s in word_stress_scores]
            else:
                normalized_stress = word_stress_scores
        else:
            normalized_stress = []
        
        stress_patterns['word_stress_scores'] = word_stress_scores
        stress_patterns['normalized_stress'] = normalized_stress
        
        return stress_patterns
    
    def _analyze_word_prosody(
        self, 
        word_info: Dict, 
        all_words: List[Dict],
        prosodic_features: Dict,
        word_index: int
    ) -> Dict:
        """Analyze prosodic emphasis for a specific word."""
        start_time = word_info['start']
        end_time = word_info['end']
        word_text = word_info['word']
        word_duration = end_time - start_time
        
        # Initialize prosodic scores
        prosodic_scores = {}
        
        # 1. Duration emphasis (unusually long words)
        mean_duration = prosodic_features['rhythm']['mean_word_duration']
        if mean_duration > 0:
            duration_ratio = word_duration / mean_duration
            prosodic_scores['duration_score'] = min(1.0, max(0.0, (duration_ratio - 1.0) / 1.0))
        else:
            prosodic_scores['duration_score'] = 0.0
        
        # 2. Pause emphasis (words preceded or followed by pauses)
        pause_score = 0.0
        pause_positions = prosodic_features['pauses']['pause_positions']
        
        # Check if preceded by pause
        if word_index - 1 in pause_positions:
            pause_score += 0.5
        
        # Check if followed by pause
        if word_index in pause_positions:
            pause_score += 0.5
        
        prosodic_scores['pause_score'] = pause_score
        
        # 3. Speech rate emphasis (surrounded by rate changes)
        rate_changes = prosodic_features['speech_rate']['rate_changes']
        if word_index < len(rate_changes):
            # Significant rate change before this word
            if word_index > 0 and abs(rate_changes[word_index - 1]) > np.std(rate_changes):
                prosodic_scores['rate_change_score'] = 0.7
            else:
                prosodic_scores['rate_change_score'] = 0.0
        else:
            prosodic_scores['rate_change_score'] = 0.0
        
        # 4. Stress pattern emphasis
        stress_scores = prosodic_features['stress_patterns']['normalized_stress']
        if word_index < len(stress_scores):
            # Convert normalized stress to emphasis score
            stress_value = stress_scores[word_index]
            prosodic_scores['stress_score'] = max(0.0, min(1.0, (stress_value + 2) / 4))
        else:
            prosodic_scores['stress_score'] = 0.0
        
        # 5. Pitch contour emphasis
        prosodic_scores['pitch_contour_score'] = self._analyze_pitch_contour_emphasis(
            prosodic_features['pitch_contour'], start_time, end_time
        )
        
        # 6. Combined prosodic score
        weights = {
            'duration_score': 0.2,
            'pause_score': 0.25,
            'rate_change_score': 0.15,
            'stress_score': 0.25,
            'pitch_contour_score': 0.15
        }
        
        total_prosodic_score = sum(
            prosodic_scores.get(key, 0) * weight 
            for key, weight in weights.items()
        )
        
        prosodic_scores['total_prosodic_score'] = total_prosodic_score
        
        return {
            'word': word_text,
            'start': start_time,
            'end': end_time,
            'duration': word_duration,
            'prosodic_scores': prosodic_scores,
            'is_prosodically_emphasized': total_prosodic_score > 0.5
        }
    
    def _analyze_pitch_contour_emphasis(
        self, 
        pitch_contour: Dict, 
        start_time: float, 
        end_time: float
    ) -> float:
        """Analyze pitch contour for emphasis indicators."""
        try:
            # Convert time to frame indices
            hop_length = self.hop_length // 2  # High resolution
            start_frame = int(start_time * self.sample_rate / hop_length)
            end_frame = int(end_time * self.sample_rate / hop_length)
            
            # Extract pitch segment
            f0_segment = pitch_contour['f0'][start_frame:end_frame]
            derivative_segment = pitch_contour['pitch_derivative'][start_frame:end_frame]
            
            if len(f0_segment) == 0:
                return 0.0
            
            # Emphasis indicators in pitch contour
            emphasis_score = 0.0
            
            # 1. Large pitch movements (jumps up or down)
            if len(derivative_segment) > 0:
                max_derivative = np.max(np.abs(derivative_segment))
                emphasis_score += min(1.0, max_derivative / 50.0) * 0.4
            
            # 2. Pitch range within word
            valid_f0 = f0_segment[f0_segment > 0]
            if len(valid_f0) > 1:
                pitch_range = np.max(valid_f0) - np.min(valid_f0)
                emphasis_score += min(1.0, pitch_range / 100.0) * 0.3
            
            # 3. Pitch contour shape (rising, falling, complex)
            if len(valid_f0) > 2:
                # Simple contour classification
                start_f0 = valid_f0[0]
                end_f0 = valid_f0[-1]
                mid_f0 = np.mean(valid_f0[len(valid_f0)//4:3*len(valid_f0)//4])
                
                # Rising or falling contour with significant change
                f0_change = abs(end_f0 - start_f0)
                if f0_change > 20:  # 20 Hz change
                    emphasis_score += 0.3
            
            return min(1.0, emphasis_score)
            
        except Exception as e:
            logger.warning(f"Pitch contour emphasis analysis failed: {e}")
            return 0.0 