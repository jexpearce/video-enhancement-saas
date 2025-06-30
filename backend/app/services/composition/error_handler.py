"""
Comprehensive Error Handling for Video Composition

Provides structured error recovery, detailed logging, and user-friendly error messages
for the video composition pipeline.
"""

import logging
import traceback
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification."""
    INPUT_VALIDATION = "input_validation"
    RESOURCE_ACCESS = "resource_access"
    PROCESSING_ERROR = "processing_error"
    NETWORK_ERROR = "network_error"
    STORAGE_ERROR = "storage_error"
    FFMPEG_ERROR = "ffmpeg_error"
    TIMEOUT_ERROR = "timeout_error"
    SYSTEM_ERROR = "system_error"

@dataclass
class CompositionError:
    """Structured error information."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    user_message: str
    details: Dict[str, Any]
    timestamp: datetime
    job_id: Optional[str] = None
    stack_trace: Optional[str] = None
    recovery_suggestion: Optional[str] = None

class CompositionErrorHandler:
    """
    Centralized error handling for video composition operations.
    
    Features:
    - Structured error classification and logging
    - User-friendly error messages
    - Recovery strategy suggestions
    - Error analytics and reporting
    """
    
    def __init__(self):
        """Initialize error handler."""
        self.error_history: List[CompositionError] = []
        self.error_counts = {category: 0 for category in ErrorCategory}
        
        # User-friendly error messages
        self.user_messages = {
            ErrorCategory.INPUT_VALIDATION: "There was an issue with the video or settings provided. Please check your input and try again.",
            ErrorCategory.RESOURCE_ACCESS: "We couldn't access some required resources. This is usually temporary - please try again in a few minutes.",
            ErrorCategory.PROCESSING_ERROR: "There was an issue processing your video. Our team has been notified and will investigate.",
            ErrorCategory.NETWORK_ERROR: "Network connectivity issues prevented processing. Please check your connection and try again.",
            ErrorCategory.STORAGE_ERROR: "There was an issue saving your video. Please try again or contact support if the problem persists.",
            ErrorCategory.FFMPEG_ERROR: "Video encoding encountered an issue. Please try again with a different video format if the problem continues.",
            ErrorCategory.TIMEOUT_ERROR: "Processing took longer than expected. Please try again - shorter videos typically process faster.",
            ErrorCategory.SYSTEM_ERROR: "A system error occurred. Our team has been notified and will investigate promptly."
        }
        
        # Recovery strategies
        self.recovery_strategies = {
            ErrorCategory.INPUT_VALIDATION: [
                "Validate input parameters",
                "Check video format compatibility",
                "Verify file size limits"
            ],
            ErrorCategory.RESOURCE_ACCESS: [
                "Retry with exponential backoff",
                "Check service availability",
                "Use fallback resources"
            ],
            ErrorCategory.PROCESSING_ERROR: [
                "Retry with reduced quality settings",
                "Skip problematic segments",
                "Use alternative processing method"
            ],
            ErrorCategory.NETWORK_ERROR: [
                "Retry request",
                "Check network connectivity",
                "Use cached resources if available"
            ],
            ErrorCategory.STORAGE_ERROR: [
                "Retry upload",
                "Try alternative storage location",
                "Clear temporary files and retry"
            ],
            ErrorCategory.FFMPEG_ERROR: [
                "Retry with different encoding settings",
                "Check FFmpeg installation",
                "Use fallback encoding parameters"
            ],
            ErrorCategory.TIMEOUT_ERROR: [
                "Increase timeout limits",
                "Process in smaller chunks",
                "Optimize processing parameters"
            ],
            ErrorCategory.SYSTEM_ERROR: [
                "Restart processing service",
                "Check system resources",
                "Contact system administrator"
            ]
        }
    
    def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        job_id: Optional[str] = None
    ) -> CompositionError:
        """
        Handle and classify an error with context.
        
        Args:
            error: The exception that occurred
            context: Context information about the error
            job_id: Optional job identifier
            
        Returns:
            CompositionError: Structured error information
        """
        
        # Generate unique error ID
        error_id = f"comp_err_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(error) % 10000}"
        
        # Classify error
        category = self._classify_error(error, context)
        severity = self._determine_severity(error, category, context)
        
        # Create structured error
        composition_error = CompositionError(
            error_id=error_id,
            category=category,
            severity=severity,
            message=str(error),
            user_message=self._get_user_message(category, error),
            details={
                'error_type': type(error).__name__,
                'context': context,
                'args': getattr(error, 'args', [])
            },
            timestamp=datetime.now(),
            job_id=job_id,
            stack_trace=traceback.format_exc(),
            recovery_suggestion=self._get_recovery_suggestion(category)
        )
        
        # Log error
        self._log_error(composition_error)
        
        # Update statistics
        self.error_counts[category] += 1
        self.error_history.append(composition_error)
        
        # Keep history manageable
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-500:]
        
        return composition_error
    
    def _classify_error(self, error: Exception, context: Dict[str, Any]) -> ErrorCategory:
        """Classify error into appropriate category."""
        
        error_message = str(error).lower()
        error_type = type(error).__name__
        
        # Network-related errors
        if any(keyword in error_message for keyword in ['connection', 'network', 'timeout', 'dns', 'unreachable']):
            return ErrorCategory.NETWORK_ERROR
        
        # FFmpeg-specific errors
        if any(keyword in error_message for keyword in ['ffmpeg', 'codec', 'encoding', 'decoding', 'filter']):
            return ErrorCategory.FFMPEG_ERROR
        
        # Storage errors
        if any(keyword in error_message for keyword in ['s3', 'storage', 'upload', 'download', 'bucket']):
            return ErrorCategory.STORAGE_ERROR
        
        # Timeout errors
        if any(keyword in error_message for keyword in ['timeout', 'timed out', 'deadline']):
            return ErrorCategory.TIMEOUT_ERROR
        
        # Input validation
        if error_type in ['ValueError', 'TypeError', 'ValidationError']:
            return ErrorCategory.INPUT_VALIDATION
        
        # File/resource access
        if error_type in ['FileNotFoundError', 'PermissionError', 'OSError']:
            return ErrorCategory.RESOURCE_ACCESS
        
        # Processing errors
        if any(keyword in error_message for keyword in ['processing', 'composition', 'animation', 'image']):
            return ErrorCategory.PROCESSING_ERROR
        
        # Default to system error
        return ErrorCategory.SYSTEM_ERROR
    
    def _determine_severity(
        self,
        error: Exception,
        category: ErrorCategory,
        context: Dict[str, Any]
    ) -> ErrorSeverity:
        """Determine error severity based on type and context."""
        
        # Critical errors that break the entire system
        if category == ErrorCategory.SYSTEM_ERROR:
            return ErrorSeverity.CRITICAL
        
        # High severity for core functionality failures
        if category in [ErrorCategory.FFMPEG_ERROR, ErrorCategory.PROCESSING_ERROR]:
            return ErrorSeverity.HIGH
        
        # Medium severity for recoverable issues
        if category in [ErrorCategory.STORAGE_ERROR, ErrorCategory.TIMEOUT_ERROR]:
            return ErrorSeverity.MEDIUM
        
        # Low severity for validation and network issues
        return ErrorSeverity.LOW
    
    def _get_user_message(self, category: ErrorCategory, error: Exception) -> str:
        """Get user-friendly error message."""
        
        base_message = self.user_messages.get(category, "An unexpected error occurred. Please try again.")
        
        # Add specific context for certain errors
        if "timeout" in str(error).lower():
            return f"{base_message} (Processing timeout after extended period)"
        elif "not found" in str(error).lower():
            return f"{base_message} (Required resource not available)"
        elif "permission" in str(error).lower():
            return f"{base_message} (Access permission issue)"
        
        return base_message
    
    def _get_recovery_suggestion(self, category: ErrorCategory) -> str:
        """Get recovery strategy suggestion."""
        
        strategies = self.recovery_strategies.get(category, ["Contact support"])
        return "; ".join(strategies[:2])  # Return top 2 strategies
    
    def _log_error(self, error: CompositionError) -> None:
        """Log error with appropriate level."""
        
        log_message = f"[{error.error_id}] {error.category.value}: {error.message}"
        
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, extra={
                'error_id': error.error_id,
                'job_id': error.job_id,
                'category': error.category.value,
                'details': error.details
            })
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(log_message, extra={
                'error_id': error.error_id,
                'job_id': error.job_id,
                'category': error.category.value
            })
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message, extra={
                'error_id': error.error_id,
                'job_id': error.job_id
            })
        else:
            logger.info(log_message)
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        
        recent_errors = [e for e in self.error_history if 
                        (datetime.now() - e.timestamp).total_seconds() < 3600]  # Last hour
        
        return {
            'total_errors': len(self.error_history),
            'recent_errors': len(recent_errors),
            'error_counts_by_category': dict(self.error_counts),
            'error_rates_by_severity': {
                severity.value: len([e for e in recent_errors if e.severity == severity])
                for severity in ErrorSeverity
            },
            'most_common_category': max(self.error_counts, key=self.error_counts.get).value if self.error_counts else None
        }
    
    def get_recovery_recommendations(self, job_id: str) -> List[str]:
        """Get recovery recommendations for a specific job."""
        
        job_errors = [e for e in self.error_history if e.job_id == job_id]
        
        if not job_errors:
            return ["No specific errors found for this job"]
        
        # Get most recent error
        latest_error = max(job_errors, key=lambda e: e.timestamp)
        
        recommendations = self.recovery_strategies.get(
            latest_error.category,
            ["Contact support for assistance"]
        )
        
        return recommendations
    
    def clear_error_history(self) -> None:
        """Clear error history (for maintenance)."""
        
        self.error_history.clear()
        self.error_counts = {category: 0 for category in ErrorCategory}
        logger.info("Error history cleared")

# Global error handler instance
composition_error_handler = CompositionErrorHandler() 