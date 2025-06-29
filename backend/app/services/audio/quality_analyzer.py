import librosa
import numpy as np
from typing import Tuple, Dict, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Audio quality metrics for processing decisions"""
    snr_db: float                    # Signal-to-noise ratio
    silence_ratio: float             # Ratio of silence to total audio
    clipping_detected: bool          # Audio clipping detection
    frequency_range: Tuple[float, float]  # Min/max frequency content
    recommended_enhancement_level: int    # 0-3 enhancement level
    spectral_centroid_mean: float    # Average spectral centroid
    zero_crossing_rate: float        # Speech activity indicator
    rms_energy: float               # Root mean square energy
    dynamic_range_db: float         # Dynamic range in dB

class AudioQualityAnalyzer:
    """Analyzes audio quality to guide processing decisions"""
    
    def __init__(self):
        self.sample_rate = 16000  # Target sample rate for Whisper
        self.frame_length = 2048
        self.hop_length = 512
        
    def analyze(self, audio_path: str) -> QualityMetrics:
        """
        Comprehensive audio quality analysis
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            QualityMetrics object with analysis results
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Calculate all quality metrics
            snr_db = self._calculate_snr(y)
            silence_ratio = self._calculate_silence_ratio(y)
            clipping_detected = self._detect_clipping(y)
            frequency_range = self._analyze_frequency_range(y, sr)
            spectral_centroid_mean = self._calculate_spectral_centroid(y, sr)
            zero_crossing_rate = self._calculate_zero_crossing_rate(y)
            rms_energy = self._calculate_rms_energy(y)
            dynamic_range_db = self._calculate_dynamic_range(y)
            
            # Determine recommended enhancement level
            enhancement_level = self._recommend_enhancement_level(
                snr_db, silence_ratio, clipping_detected, spectral_centroid_mean
            )
            
            return QualityMetrics(
                snr_db=snr_db,
                silence_ratio=silence_ratio,
                clipping_detected=clipping_detected,
                frequency_range=frequency_range,
                recommended_enhancement_level=enhancement_level,
                spectral_centroid_mean=spectral_centroid_mean,
                zero_crossing_rate=zero_crossing_rate,
                rms_energy=rms_energy,
                dynamic_range_db=dynamic_range_db
            )
            
        except Exception as e:
            logger.error(f"Audio quality analysis failed: {str(e)}")
            # Return default metrics if analysis fails
            return self._default_metrics()
    
    def _calculate_snr(self, y: np.ndarray) -> float:
        """Calculate signal-to-noise ratio"""
        # Use voice activity detection to separate signal from noise
        # Simple approach: top 60% energy frames = signal, bottom 20% = noise
        frame_energy = librosa.feature.rms(y=y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        
        signal_threshold = np.percentile(frame_energy, 70)
        noise_threshold = np.percentile(frame_energy, 20)
        
        signal_frames = frame_energy[frame_energy > signal_threshold]
        noise_frames = frame_energy[frame_energy < noise_threshold]
        
        if len(signal_frames) == 0 or len(noise_frames) == 0:
            return 10.0  # Default moderate SNR
            
        signal_power = np.mean(signal_frames ** 2)
        noise_power = np.mean(noise_frames ** 2)
        
        if noise_power == 0:
            return 40.0  # Very high SNR
            
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)
        
        return float(snr_db)
    
    def _calculate_silence_ratio(self, y: np.ndarray) -> float:
        """Calculate ratio of silence to total audio duration"""
        # Use energy-based voice activity detection
        frame_energy = librosa.feature.rms(y=y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        
        # Threshold for silence (adaptive based on audio statistics)
        energy_threshold = np.percentile(frame_energy, 15)  # Bottom 15% considered silence
        
        silence_frames = np.sum(frame_energy < energy_threshold)
        total_frames = len(frame_energy)
        
        return float(silence_frames / total_frames)
    
    def _detect_clipping(self, y: np.ndarray) -> bool:
        """Detect audio clipping (distortion from amplitude limiting)"""
        # Check for samples at or near maximum amplitude
        clipping_threshold = 0.95
        clipped_samples = np.sum(np.abs(y) > clipping_threshold)
        clipping_ratio = clipped_samples / len(y)
        
        return clipping_ratio > 0.001  # 0.1% clipping threshold
    
    def _analyze_frequency_range(self, y: np.ndarray, sr: int) -> Tuple[float, float]:
        """Analyze frequency content range"""
        # Compute power spectral density
        freqs, psd = librosa.core.spectrum._spectrogram(
            y=y, hop_length=self.hop_length, win_length=self.frame_length
        )
        
        # Convert to frequency bins
        freq_bins = librosa.frames_to_time(np.arange(len(freqs)), sr=sr, hop_length=self.hop_length)
        
        # Find frequency range containing 90% of energy
        total_energy = np.sum(psd)
        cumulative_energy = np.cumsum(np.sum(psd, axis=1))
        
        # Find 5th and 95th percentile frequencies
        min_freq_idx = np.where(cumulative_energy >= 0.05 * total_energy)[0]
        max_freq_idx = np.where(cumulative_energy >= 0.95 * total_energy)[0]
        
        if len(min_freq_idx) == 0:
            min_freq = 0.0
        else:
            min_freq = freq_bins[min_freq_idx[0]]
            
        if len(max_freq_idx) == 0:
            max_freq = sr / 2
        else:
            max_freq = freq_bins[max_freq_idx[0]]
        
        return (float(min_freq), float(max_freq))
    
    def _calculate_spectral_centroid(self, y: np.ndarray, sr: int) -> float:
        """Calculate mean spectral centroid (brightness indicator)"""
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        return float(np.mean(spectral_centroids))
    
    def _calculate_zero_crossing_rate(self, y: np.ndarray) -> float:
        """Calculate zero crossing rate (speech activity indicator)"""
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        return float(np.mean(zcr))
    
    def _calculate_rms_energy(self, y: np.ndarray) -> float:
        """Calculate root mean square energy"""
        rms = librosa.feature.rms(y=y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        return float(np.mean(rms))
    
    def _calculate_dynamic_range(self, y: np.ndarray) -> float:
        """Calculate dynamic range in dB"""
        if len(y) == 0:
            return 0.0
            
        max_amplitude = np.max(np.abs(y))
        min_amplitude = np.min(np.abs(y[np.abs(y) > 0]))  # Exclude perfect silence
        
        if min_amplitude == 0 or max_amplitude == 0:
            return 0.0
            
        dynamic_range = 20 * np.log10(max_amplitude / min_amplitude)
        return float(dynamic_range)
    
    def _recommend_enhancement_level(
        self, 
        snr_db: float, 
        silence_ratio: float, 
        clipping_detected: bool,
        spectral_centroid: float
    ) -> int:
        """
        Recommend enhancement level based on quality metrics
        
        Levels:
        0 - No enhancement needed (high quality)
        1 - Light enhancement (good quality, minor issues)
        2 - Moderate enhancement (medium quality, noticeable issues)
        3 - Heavy enhancement (poor quality, major issues)
        """
        score = 0
        
        # SNR scoring
        if snr_db < 5:
            score += 3
        elif snr_db < 10:
            score += 2
        elif snr_db < 15:
            score += 1
            
        # Silence ratio scoring
        if silence_ratio > 0.7:
            score += 1
        elif silence_ratio > 0.5:
            score += 0.5
            
        # Clipping penalty
        if clipping_detected:
            score += 2
            
        # Spectral centroid (brightness) scoring
        if spectral_centroid < 1000:  # Very low brightness (muffled)
            score += 1
        elif spectral_centroid > 4000:  # Very high brightness (harsh)
            score += 1
            
        # Convert score to enhancement level
        if score >= 4:
            return 3  # Heavy enhancement
        elif score >= 2:
            return 2  # Moderate enhancement
        elif score >= 1:
            return 1  # Light enhancement
        else:
            return 0  # No enhancement needed
    
    def _default_metrics(self) -> QualityMetrics:
        """Return default metrics when analysis fails"""
        return QualityMetrics(
            snr_db=10.0,
            silence_ratio=0.3,
            clipping_detected=False,
            frequency_range=(80.0, 8000.0),
            recommended_enhancement_level=1,
            spectral_centroid_mean=2000.0,
            zero_crossing_rate=0.1,
            rms_energy=0.1,
            dynamic_range_db=20.0
        ) 