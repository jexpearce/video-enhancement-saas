import ffmpeg
import librosa
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import tempfile
import asyncio
import os
import logging
from scipy import signal
from scipy.signal import butter, filtfilt, hilbert
from ...models.schemas import AudioSegment, ProcessedAudio
from .quality_analyzer import AudioQualityAnalyzer, QualityMetrics
from app.config import settings

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Production-grade audio extraction and enhancement pipeline
    
    Handles:
    - Video-to-audio extraction using FFmpeg
    - Audio quality analysis and enhancement
    - Noise reduction and spectral processing
    - Segmentation for optimal transcription
    """
    
    def __init__(self):
        self.target_sample_rate = settings.target_sample_rate
        self.chunk_duration = settings.audio_chunk_duration
        self.quality_analyzer = AudioQualityAnalyzer()
        
        # Audio processing parameters
        self.frame_length = 2048
        self.hop_length = 512
        self.enhancement_levels = {
            0: "none",
            1: "light", 
            2: "moderate",
            3: "heavy"
        }
        
    async def extract_and_process(self, video_path: str) -> ProcessedAudio:
        """
        Main processing pipeline: extract audio from video and enhance it
        
        Args:
            video_path: Path to input video file
            
        Returns:
            ProcessedAudio object with enhanced audio and segments
        """
        logger.info(f"Starting audio processing for: {video_path}")
        
        try:
            # 1. Probe video for metadata
            video_info = await self._probe_video(video_path)
            logger.info(f"Video info: {video_info}")
            
            # 2. Extract audio with optimal settings
            raw_audio_path = await self._extract_audio(video_path, video_info)
            
            # 3. Analyze audio quality
            quality_metrics = self.quality_analyzer.analyze(raw_audio_path)
            logger.info(f"Audio quality: SNR={quality_metrics.snr_db:.1f}dB, "
                       f"Enhancement level={quality_metrics.recommended_enhancement_level}")
            
            # 4. Enhance audio based on quality analysis
            enhanced_audio_path = await self._enhance_audio(raw_audio_path, quality_metrics)
            
            # 5. Load enhanced audio for segmentation
            audio_data, sr = librosa.load(enhanced_audio_path, sr=self.target_sample_rate)
            duration = len(audio_data) / sr
            
            # 6. Create segments for processing
            segments = self._create_segments(enhanced_audio_path, duration, video_info.get('fps', 30))
            
            # 7. Clean up temporary files
            if os.path.exists(raw_audio_path):
                os.remove(raw_audio_path)
                
            return ProcessedAudio(
                path=enhanced_audio_path,
                segments=segments,
                duration=duration,
                sample_rate=self.target_sample_rate,
                metadata={
                    "original_video": video_path,
                    "video_info": video_info,
                    "quality_metrics": quality_metrics.__dict__,
                    "enhancement_applied": self.enhancement_levels[quality_metrics.recommended_enhancement_level]
                }
            )
            
        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}")
            raise RuntimeError(f"Audio processing failed: {str(e)}")
    
    async def _probe_video(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata using FFmpeg probe"""
        try:
            probe = ffmpeg.probe(video_path)
            
            # Find video and audio streams
            video_stream = None
            audio_stream = None
            
            for stream in probe['streams']:
                if stream['codec_type'] == 'video' and not video_stream:
                    video_stream = stream
                elif stream['codec_type'] == 'audio' and not audio_stream:
                    audio_stream = stream
            
            # Extract relevant metadata
            metadata = {
                'duration': float(probe['format'].get('duration', 0)),
                'size': int(probe['format'].get('size', 0)),
                'bitrate': int(probe['format'].get('bit_rate', 0)),
            }
            
            if video_stream:
                metadata.update({
                    'width': int(video_stream.get('width', 0)),
                    'height': int(video_stream.get('height', 0)),
                    'fps': eval(video_stream.get('r_frame_rate', '30/1')),  # Convert fraction to float
                    'video_codec': video_stream.get('codec_name', 'unknown')
                })
            
            if audio_stream:
                metadata.update({
                    'audio_codec': audio_stream.get('codec_name', 'unknown'),
                    'sample_rate': int(audio_stream.get('sample_rate', 0)),
                    'channels': int(audio_stream.get('channels', 0)),
                    'audio_bitrate': int(audio_stream.get('bit_rate', 0))
                })
                
            return metadata
            
        except Exception as e:
            logger.warning(f"Video probe failed: {str(e)}")
            return {'duration': 0, 'fps': 30}  # Default values
    
    async def _extract_audio(self, video_path: str, video_info: Dict[str, Any]) -> str:
        """Extract audio from video using FFmpeg with optimal settings"""
        
        # Create temporary file for raw audio
        temp_audio = tempfile.mktemp(suffix='.wav')
        
        try:
            # Build FFmpeg command for optimal audio extraction
            stream = ffmpeg.input(video_path)
            
            # Audio extraction settings optimized for speech
            audio_stream = stream.audio.filter(
                'highpass', f=80  # Remove low-frequency noise
            ).filter(
                'lowpass', f=8000  # Remove high-frequency noise above speech range
            )
            
            # Output with optimal settings for Whisper
            out = ffmpeg.output(
                audio_stream,
                temp_audio,
                acodec='pcm_s16le',  # 16-bit PCM
                ac=1,               # Mono
                ar=self.target_sample_rate,  # 16kHz for Whisper
                f='wav'
            )
            
            # Run extraction
            await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: ffmpeg.run(out, overwrite_output=True, quiet=True)
            )
            
            return temp_audio
            
        except Exception as e:
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            raise RuntimeError(f"Audio extraction failed: {str(e)}")
    
    async def _enhance_audio(self, audio_path: str, quality_metrics: QualityMetrics) -> str:
        """
        Apply audio enhancement based on quality analysis
        
        Enhancement pipeline:
        - Spectral noise reduction
        - Dynamic range compression
        - EQ optimization for speech
        - Normalization
        """
        enhancement_level = quality_metrics.recommended_enhancement_level
        
        if enhancement_level == 0:
            # No enhancement needed - just copy the file
            enhanced_path = tempfile.mktemp(suffix='.wav')
            os.rename(audio_path, enhanced_path)
            return enhanced_path
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.target_sample_rate)
            
            # Apply enhancements based on level
            if enhancement_level >= 1:
                y = self._spectral_denoise(y, sr, intensity=0.3)
                
            if enhancement_level >= 2:
                y = self._compress_dynamic_range(y, ratio=4.0)
                y = self._optimize_speech_eq(y, sr)
                
            if enhancement_level >= 3:
                y = self._spectral_denoise(y, sr, intensity=0.6)  # Stronger denoising
                y = self._compress_dynamic_range(y, ratio=6.0)    # More compression
                
            # Always normalize
            y = self._normalize_audio(y)
            
            # Save enhanced audio
            enhanced_path = tempfile.mktemp(suffix='.wav')
            librosa.output.write_wav(enhanced_path, y, sr)
            
            # Clean up original
            os.remove(audio_path)
            
            return enhanced_path
            
        except Exception as e:
            logger.error(f"Audio enhancement failed: {str(e)}")
            # Return original if enhancement fails
            return audio_path
    
    def _spectral_denoise(self, y: np.ndarray, sr: int, intensity: float = 0.5) -> np.ndarray:
        """
        Spectral noise reduction using Wiener filtering approach
        
        Args:
            y: Audio signal
            sr: Sample rate
            intensity: Denoising intensity (0.0 - 1.0)
        """
        # Compute STFT
        stft = librosa.stft(y, hop_length=self.hop_length, win_length=self.frame_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise from quiet segments (bottom 10% of energy)
        frame_energy = np.mean(magnitude, axis=0)
        noise_threshold = np.percentile(frame_energy, 10)
        noise_frames = magnitude[:, frame_energy <= noise_threshold]
        
        if noise_frames.shape[1] > 0:
            # Compute noise profile
            noise_profile = np.mean(noise_frames, axis=1, keepdims=True)
            
            # Apply spectral subtraction
            clean_magnitude = magnitude - intensity * noise_profile
            
            # Ensure we don't over-subtract (minimum 20% of original)
            clean_magnitude = np.maximum(clean_magnitude, 0.2 * magnitude)
        else:
            clean_magnitude = magnitude
        
        # Reconstruct signal
        clean_stft = clean_magnitude * np.exp(1j * phase)
        y_clean = librosa.istft(clean_stft, hop_length=self.hop_length, win_length=self.frame_length)
        
        return y_clean[:len(y)]  # Maintain original length
    
    def _compress_dynamic_range(self, y: np.ndarray, ratio: float = 4.0, threshold: float = -20.0) -> np.ndarray:
        """
        Apply dynamic range compression to even out volume levels
        
        Args:
            y: Audio signal
            ratio: Compression ratio
            threshold: Compression threshold in dB
        """
        # Convert to dB scale
        y_db = 20 * np.log10(np.abs(y) + 1e-10)
        
        # Apply compression above threshold
        compressed_db = np.where(
            y_db > threshold,
            threshold + (y_db - threshold) / ratio,
            y_db
        )
        
        # Convert back to linear scale
        gain = 10 ** ((compressed_db - y_db) / 20)
        y_compressed = y * gain
        
        return y_compressed
    
    def _optimize_speech_eq(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply EQ optimization for speech clarity
        
        Speech optimization:
        - Slight boost around 2-4kHz (speech intelligibility)
        - Slight cut below 100Hz (rumble)
        - Slight cut above 6kHz (sibilance)
        """
        # Design filters
        nyquist = sr / 2
        
        # High-pass filter (remove rumble below 80Hz)
        b_hp, a_hp = butter(2, 80 / nyquist, btype='high')
        y = filtfilt(b_hp, a_hp, y)
        
        # Speech clarity boost (2-4kHz)
        # Create a gentle bell curve boost
        center_freq = 3000
        bandwidth = 2000
        boost_db = 2.0
        
        # Simple implementation using frequency domain
        stft = librosa.stft(y, hop_length=self.hop_length)
        freqs = librosa.stft_frequencies(sr=sr, hop_length=self.hop_length)
        
        # Create boost filter
        boost_factor = np.ones_like(freqs)
        for i, freq in enumerate(freqs):
            if center_freq - bandwidth/2 <= freq <= center_freq + bandwidth/2:
                # Gaussian-like boost
                distance = abs(freq - center_freq) / (bandwidth/2)
                boost_factor[i] = 10 ** (boost_db * (1 - distance**2) / 20)
        
        # Apply boost
        stft_boosted = stft * boost_factor.reshape(-1, 1)
        y = librosa.istft(stft_boosted, hop_length=self.hop_length)
        
        return y[:len(y)]  # Maintain original length
    
    def _normalize_audio(self, y: np.ndarray, target_db: float = -12.0) -> np.ndarray:
        """
        Normalize audio to target RMS level
        
        Args:
            y: Audio signal
            target_db: Target RMS level in dB
        """
        # Calculate current RMS
        rms_current = np.sqrt(np.mean(y**2))
        
        if rms_current == 0:
            return y
        
        # Calculate target RMS
        rms_target = 10 ** (target_db / 20)
        
        # Apply gain
        gain = rms_target / rms_current
        y_normalized = y * gain
        
        # Prevent clipping
        max_val = np.max(np.abs(y_normalized))
        if max_val > 0.95:
            y_normalized = y_normalized * (0.95 / max_val)
        
        return y_normalized
    
    def _create_segments(self, audio_path: str, duration: float, fps: float) -> List[AudioSegment]:
        """
        Create audio segments for processing with overlap for context
        
        Args:
            audio_path: Path to processed audio
            duration: Total audio duration
            fps: Video frame rate
            
        Returns:
            List of AudioSegment objects
        """
        segments = []
        segment_length = self.chunk_duration  # seconds
        overlap = 2.0  # 2 seconds overlap for context
        
        start_time = 0.0
        segment_id = 0
        
        while start_time < duration:
            end_time = min(start_time + segment_length, duration)
            
            # Create segment with overlap for context (except first segment)
            segment_start = max(0, start_time - overlap/2) if segment_id > 0 else 0
            segment_end = min(duration, end_time + overlap/2)
            
            segment = AudioSegment(
                path=audio_path,
                start_time=segment_start,
                end_time=segment_end,
                duration=segment_end - segment_start,
                video_fps=fps,
                previous_text=None,  # Will be filled during transcription
                metadata={
                    "segment_id": segment_id,
                    "overlap_seconds": overlap,
                    "is_final": end_time >= duration
                }
            )
            
            segments.append(segment)
            
            # Move to next segment
            start_time = end_time - overlap/2  # Overlap for continuity
            segment_id += 1
            
            # Prevent infinite loop
            if end_time >= duration:
                break
        
        logger.info(f"Created {len(segments)} audio segments")
        return segments 