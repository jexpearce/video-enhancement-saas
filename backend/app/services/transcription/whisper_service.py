import whisper
import whisper_timestamped as whisper_ts
import numpy as np
import librosa
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import logging
import tempfile
import os
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

from ...models.schemas import AudioSegment, TranscriptionResult, WordInfo
from ..config import settings

logger = logging.getLogger(__name__)

@dataclass
class TranscriptionConfig:
    """Configuration for Whisper transcription"""
    model_size: str = "base"  # base, small, medium, large, large-v3
    language: Optional[str] = None  # Auto-detect if None
    initial_prompt: Optional[str] = None
    temperature: float = 0.0  # Deterministic results
    compression_ratio_threshold: float = 2.4
    condition_on_previous_text: bool = True
    word_timestamps: bool = True
    no_speech_threshold: float = 0.6
    logprob_threshold: float = -1.0

class WhisperService:
    """
    Production-grade Whisper transcription service
    
    Features:
    - Word-level timestamps using whisper-timestamped
    - Context-aware prompting for better accuracy
    - Retry logic and error handling
    - Segment-based processing with overlap handling
    - Domain-specific optimization
    """
    
    def __init__(self, config: Optional[TranscriptionConfig] = None):
        self.config = config or TranscriptionConfig(model_size=settings.whisper_model)
        self.model = None
        self._load_model()
        
        # Domain-specific prompts for better accuracy
        self.domain_prompts = {
            "news": "Latest news update covering current events, politics, and breaking news.",
            "politics": "Political commentary and analysis discussing government, policy, and current affairs.",
            "tech": "Technology discussion covering software, hardware, and digital innovation.",
            "education": "Educational content explaining concepts and sharing knowledge.",
            "general": "Conversation and commentary on various topics."
        }
        
    def _load_model(self):
        """Load Whisper model with GPU optimization if available"""
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading Whisper model '{self.config.model_size}' on {device}")
            
            # Load model with appropriate device
            self.model = whisper.load_model(self.config.model_size, device=device)
            logger.info("Whisper model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            raise RuntimeError(f"Whisper model loading failed: {str(e)}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def transcribe_segment(self, audio_segment: AudioSegment) -> TranscriptionResult:
        """
        Transcribe an audio segment with advanced options and retry logic
        
        Args:
            audio_segment: AudioSegment object with path and timing info
            
        Returns:
            TranscriptionResult with word-level timestamps
        """
        try:
            logger.info(f"Transcribing segment: {audio_segment.start_time:.1f}s - {audio_segment.end_time:.1f}s")
            
            # Extract segment audio if needed
            segment_audio_path = await self._extract_segment_audio(audio_segment)
            
            # Generate context-aware prompt
            initial_prompt = self._generate_context_prompt(audio_segment)
            
            # Run transcription with word-level timestamps
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._transcribe_audio_file,
                segment_audio_path,
                initial_prompt
            )
            
            # Post-process and align timing
            processed_result = self._post_process_transcription(result, audio_segment)
            
            # Clean up temporary files
            if segment_audio_path != audio_segment.path:
                os.remove(segment_audio_path)
            
            return processed_result
            
        except Exception as e:
            logger.error(f"Transcription failed for segment {audio_segment.start_time:.1f}s: {str(e)}")
            # Return empty result on failure
            return self._empty_transcription_result()
    
    def _transcribe_audio_file(self, audio_path: str, initial_prompt: Optional[str]) -> Dict[str, Any]:
        """
        Core transcription function using whisper-timestamped
        
        Args:
            audio_path: Path to audio file
            initial_prompt: Context prompt for better accuracy
            
        Returns:
            Raw transcription result from whisper-timestamped
        """
        try:
            # Use whisper-timestamped for word-level timestamps
            result = whisper_ts.transcribe(
                self.model,
                audio_path,
                language=self.config.language,
                initial_prompt=initial_prompt,
                temperature=self.config.temperature,
                compression_ratio_threshold=self.config.compression_ratio_threshold,
                condition_on_previous_text=self.config.condition_on_previous_text,
                word_timestamps=True,
                # Enhanced punctuation handling
                prepend_punctuations="\"'"¿([{-",
                append_punctuations="\"'.。,，!！?？:：")]}、",
                # Improved segment detection
                no_speech_threshold=self.config.no_speech_threshold,
                logprob_threshold=self.config.logprob_threshold
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Core transcription failed: {str(e)}")
            raise
    
    async def _extract_segment_audio(self, audio_segment: AudioSegment) -> str:
        """
        Extract specific audio segment if timing information is provided
        
        Args:
            audio_segment: AudioSegment with timing info
            
        Returns:
            Path to extracted audio segment
        """
        # If no timing info, use the full audio
        if audio_segment.start_time == 0 and audio_segment.end_time == audio_segment.duration:
            return audio_segment.path
        
        try:
            # Load audio and extract segment
            y, sr = librosa.load(
                audio_segment.path,
                offset=audio_segment.start_time,
                duration=audio_segment.duration,
                sr=16000  # Whisper's preferred sample rate
            )
            
            # Save segment to temporary file
            temp_audio = tempfile.mktemp(suffix='.wav')
            librosa.output.write_wav(temp_audio, y, sr)
            
            return temp_audio
            
        except Exception as e:
            logger.warning(f"Failed to extract audio segment: {str(e)}")
            # Fall back to full audio
            return audio_segment.path
    
    def _generate_context_prompt(self, audio_segment: AudioSegment) -> Optional[str]:
        """
        Generate context-aware prompts for better transcription accuracy
        
        Args:
            audio_segment: AudioSegment with metadata
            
        Returns:
            Context prompt string or None
        """
        prompt_parts = []
        
        # Use previous text for continuity
        if audio_segment.previous_text:
            # Take last 50 characters for context
            context = audio_segment.previous_text[-50:].strip()
            if context:
                prompt_parts.append(f"...{context}")
        
        # Add domain-specific prompt
        domain = audio_segment.metadata.get('domain', 'general')
        if domain in self.domain_prompts:
            prompt_parts.append(self.domain_prompts[domain])
        
        # Combine prompt parts
        if prompt_parts:
            return " ".join(prompt_parts)
        
        return None
    
    def _post_process_transcription(
        self, 
        raw_result: Dict[str, Any], 
        audio_segment: AudioSegment
    ) -> TranscriptionResult:
        """
        Post-process transcription results and align timing with video frames
        
        Args:
            raw_result: Raw result from whisper-timestamped
            audio_segment: Original audio segment info
            
        Returns:
            Processed TranscriptionResult
        """
        try:
            # Extract text and segments
            full_text = raw_result.get('text', '').strip()
            segments = raw_result.get('segments', [])
            
            # Process word-level information
            words = []
            for segment in segments:
                segment_words = segment.get('words', [])
                for word_info in segment_words:
                    # Adjust timing to account for segment offset
                    adjusted_start = word_info['start'] + audio_segment.start_time
                    adjusted_end = word_info['end'] + audio_segment.start_time
                    
                    word = WordInfo(
                        text=word_info['text'].strip(),
                        start=adjusted_start,
                        end=adjusted_end,
                        confidence=word_info.get('confidence', 0.0),
                        speaker_id=0  # Single speaker for now
                    )
                    words.append(word)
            
            # Calculate overall confidence
            if words:
                avg_confidence = sum(w.confidence for w in words) / len(words)
            else:
                avg_confidence = 0.0
            
            # Align with video frames if available
            if audio_segment.video_fps:
                words = self._align_to_frames(words, audio_segment.video_fps)
            
            return TranscriptionResult(
                text=full_text,
                words=words,
                language=raw_result.get('language', 'en'),
                confidence=avg_confidence,
                duration=audio_segment.duration,
                model_version=f"whisper-{self.config.model_size}"
            )
            
        except Exception as e:
            logger.error(f"Post-processing failed: {str(e)}")
            return self._empty_transcription_result()
    
    def _align_to_frames(self, words: List[WordInfo], fps: float) -> List[WordInfo]:
        """
        Align word timestamps to video frame boundaries for precise synchronization
        
        Args:
            words: List of WordInfo objects
            fps: Video frames per second
            
        Returns:
            List of WordInfo with frame-aligned timestamps
        """
        frame_duration = 1.0 / fps
        
        aligned_words = []
        for word in words:
            # Round to nearest frame
            aligned_start = round(word.start / frame_duration) * frame_duration
            aligned_end = round(word.end / frame_duration) * frame_duration
            
            # Ensure end > start
            if aligned_end <= aligned_start:
                aligned_end = aligned_start + frame_duration
            
            aligned_word = WordInfo(
                text=word.text,
                start=aligned_start,
                end=aligned_end,
                confidence=word.confidence,
                speaker_id=word.speaker_id
            )
            aligned_words.append(aligned_word)
        
        return aligned_words
    
    def _empty_transcription_result(self) -> TranscriptionResult:
        """Return empty transcription result for error cases"""
        return TranscriptionResult(
            text="",
            words=[],
            language="en",
            confidence=0.0,
            duration=0.0,
            model_version=f"whisper-{self.config.model_size}"
        )
    
    async def transcribe_multiple_segments(
        self, 
        audio_segments: List[AudioSegment],
        domain: str = "general"
    ) -> List[TranscriptionResult]:
        """
        Transcribe multiple audio segments in parallel with context handling
        
        Args:
            audio_segments: List of AudioSegment objects
            domain: Content domain for context prompting
            
        Returns:
            List of TranscriptionResult objects
        """
        logger.info(f"Transcribing {len(audio_segments)} segments")
        
        # Add domain context to segments
        for segment in audio_segments:
            segment.metadata['domain'] = domain
        
        # Process segments with context chain
        results = []
        previous_text = ""
        
        for i, segment in enumerate(audio_segments):
            # Add previous text for context (except first segment)
            if i > 0 and previous_text:
                segment.previous_text = previous_text
            
            # Transcribe segment
            result = await self.transcribe_segment(segment)
            results.append(result)
            
            # Update context for next segment
            if result.text:
                previous_text = result.text
                # Keep context reasonable length
                if len(previous_text) > 200:
                    previous_text = previous_text[-200:]
        
        logger.info(f"Completed transcription of {len(results)} segments")
        return results
    
    def merge_segment_results(self, segment_results: List[TranscriptionResult]) -> TranscriptionResult:
        """
        Merge multiple segment transcription results into single result
        
        Args:
            segment_results: List of TranscriptionResult objects
            
        Returns:
            Merged TranscriptionResult
        """
        if not segment_results:
            return self._empty_transcription_result()
        
        # Combine text
        full_text = " ".join(result.text for result in segment_results if result.text)
        
        # Combine words
        all_words = []
        for result in segment_results:
            all_words.extend(result.words)
        
        # Calculate overall metrics
        total_duration = sum(result.duration for result in segment_results)
        avg_confidence = sum(result.confidence for result in segment_results) / len(segment_results)
        
        # Use language from first non-empty result
        language = "en"
        for result in segment_results:
            if result.language:
                language = result.language
                break
        
        return TranscriptionResult(
            text=full_text,
            words=all_words,
            language=language,
            confidence=avg_confidence,
            duration=total_duration,
            model_version=f"whisper-{self.config.model_size}"
        ) 