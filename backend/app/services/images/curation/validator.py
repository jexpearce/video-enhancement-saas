"""
Legal Compliance Validator - Phase 2 Days 23-24

Advanced validation system for image legal compliance with:
- NSFW content detection
- Copyright/watermark detection  
- Brand safety validation
- Prohibited content database checking
"""

import logging
import hashlib
import io
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio

import requests

try:
    import numpy as np
    import cv2
    from PIL import Image
    VISION_LIBS_AVAILABLE = True
except ImportError:
    VISION_LIBS_AVAILABLE = False
    np = None
    cv2 = None
    Image = None

from ..providers.base import ImageResult, ImageLicense

logger = logging.getLogger(__name__)

@dataclass
class ValidationIssue:
    """Individual validation issue found in an image."""
    severity: str  # 'critical', 'high', 'medium', 'low'
    issue_type: str  # 'nsfw', 'copyright', 'watermark', 'prohibited', 'quality'
    message: str
    confidence: float
    details: Dict

@dataclass
class ValidationResult:
    """Result of comprehensive image validation."""
    image: ImageResult
    is_valid: bool
    issues: List[ValidationIssue]
    validation_timestamp: datetime
    validation_version: str
    safety_score: float  # 0.0 = unsafe, 1.0 = safe
    
class LegalComplianceValidator:
    """Advanced legal compliance and safety validator."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the validator.
        
        Args:
            config: Configuration for validation settings
        """
        self.config = config or {}
        
        # Validation thresholds
        self.nsfw_threshold = self.config.get('nsfw_threshold', 0.3)
        self.copyright_threshold = self.config.get('copyright_threshold', 0.7)
        self.watermark_threshold = self.config.get('watermark_threshold', 0.6)
        
        # Initialize detection systems
        self._init_detection_systems()
        
        # Brand safety keywords
        self.unsafe_keywords = {
            'violence', 'weapon', 'gun', 'knife', 'blood', 'fight',
            'drug', 'alcohol', 'smoking', 'cigarette', 'beer', 'wine',
            'adult', 'sexy', 'nude', 'bikini', 'lingerie',
            'controversial', 'political', 'protest', 'riot',
            'accident', 'crash', 'disaster', 'death', 'funeral'
        }
    
    def _init_detection_systems(self):
        """Initialize detection systems."""
        try:
            # Prohibited content hashes (simulated)
            self.prohibited_hashes = {
                'abc123def456',
                'xyz789ghi012'
            }
            
            # Initialize computer vision detectors if available
            if VISION_LIBS_AVAILABLE and cv2:
                self.sift_detector = cv2.SIFT_create()
                self.vision_available = True
            else:
                self.sift_detector = None
                self.vision_available = False
            
            logger.info("Validation systems initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize detection systems: {e}")
            self.vision_available = False
    
    async def validate_image_batch(self, images: List[ImageResult]) -> List[ValidationResult]:
        """
        Validate a batch of images for legal compliance.
        
        Args:
            images: List of ImageResult objects to validate
            
        Returns:
            List of ValidationResult objects
        """
        try:
            # Process images in parallel for efficiency
            validation_tasks = [
                self._validate_single_image(image) for image in images
            ]
            
            results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            # Handle any exceptions
            validated_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Validation failed for image {images[i].image_url}: {result}")
                    # Create a failed validation result
                    validated_results.append(ValidationResult(
                        image=images[i],
                        is_valid=False,
                        issues=[ValidationIssue(
                            severity='critical',
                            issue_type='validation_error',
                            message=f"Validation failed: {str(result)}",
                            confidence=1.0,
                            details={}
                        )],
                        validation_timestamp=datetime.utcnow(),
                        validation_version='1.0',
                        safety_score=0.0
                    ))
                else:
                    validated_results.append(result)
            
            logger.info(f"Validated {len(validated_results)} images")
            return validated_results
            
        except Exception as e:
            logger.error(f"Error in batch validation: {e}")
            return []
    
    async def _validate_single_image(self, image: ImageResult) -> ValidationResult:
        """Validate a single image for all compliance criteria."""
        issues = []
        
        try:
            # Download image for analysis
            image_data = await self._download_image(image.thumbnail_url)
            image_hash = hashlib.sha256(image_data).hexdigest()
            
            # 1. Check against prohibited content database
            prohibited_issue = self._check_prohibited_content(image_hash)
            if prohibited_issue:
                issues.append(prohibited_issue)
            
            # 2. Verify license validity
            license_issue = await self._verify_license_compliance(image)
            if license_issue:
                issues.append(license_issue)
            
            # 3. Check image availability
            availability_issue = await self._check_image_availability(image)
            if availability_issue:
                issues.append(availability_issue)
            
            # 4. NSFW content detection
            nsfw_issue = await self._detect_nsfw_content(image_data)
            if nsfw_issue:
                issues.append(nsfw_issue)
            
            # 5. Copyright and watermark detection
            copyright_issue = await self._detect_copyright_content(image_data)
            if copyright_issue:
                issues.append(copyright_issue)
            
            # 6. Brand safety validation
            brand_safety_issue = self._check_brand_safety(image)
            if brand_safety_issue:
                issues.append(brand_safety_issue)
            
            # 7. Image quality validation
            quality_issue = await self._validate_image_quality(image_data, image)
            if quality_issue:
                issues.append(quality_issue)
            
            # Calculate overall safety score
            safety_score = self._calculate_safety_score(issues)
            
            # Determine if image is valid (no critical or high severity issues)
            is_valid = not any(
                issue.severity in ['critical', 'high'] for issue in issues
            )
            
            return ValidationResult(
                image=image,
                is_valid=is_valid,
                issues=issues,
                validation_timestamp=datetime.utcnow(),
                validation_version='1.0',
                safety_score=safety_score
            )
            
        except Exception as e:
            logger.error(f"Error validating image {image.image_url}: {e}")
            return ValidationResult(
                image=image,
                is_valid=False,
                issues=[ValidationIssue(
                    severity='critical',
                    issue_type='validation_error',
                    message=f"Validation error: {str(e)}",
                    confidence=1.0,
                    details={'error': str(e)}
                )],
                validation_timestamp=datetime.utcnow(),
                validation_version='1.0',
                safety_score=0.0
            )
    
    def _check_prohibited_content(self, image_hash: str) -> Optional[ValidationIssue]:
        """Check if image hash matches prohibited content database."""
        if image_hash in self.prohibited_hashes:
            return ValidationIssue(
                severity='critical',
                issue_type='prohibited',
                message='Image matches prohibited content database',
                confidence=1.0,
                details={'hash': image_hash}
            )
        return None
    
    async def _verify_license_compliance(self, image: ImageResult) -> Optional[ValidationIssue]:
        """Verify image license is valid for commercial use."""
        # Check if license allows commercial use
        non_commercial_licenses = {
            ImageLicense.EDITORIAL_ONLY,
            ImageLicense.UNKNOWN
        }
        
        if image.license in non_commercial_licenses:
            return ValidationIssue(
                severity='high',
                issue_type='invalid_license',
                message=f'License {image.license.value} not suitable for commercial use',
                confidence=1.0,
                details={'license': image.license.value}
            )
        
        # Additional license verification could be done here
        # (e.g., checking if Creative Commons attribution is properly provided)
        
        return None
    
    async def _check_image_availability(self, image: ImageResult) -> Optional[ValidationIssue]:
        """Check if image URL is accessible."""
        try:
            response = requests.head(image.image_url, timeout=10)
            if response.status_code != 200:
                return ValidationIssue(
                    severity='high',
                    issue_type='unavailable',
                    message=f'Image URL returned status {response.status_code}',
                    confidence=1.0,
                    details={'status_code': response.status_code}
                )
        except Exception as e:
            return ValidationIssue(
                severity='high',
                issue_type='unavailable',
                message=f'Image URL is not accessible: {str(e)}',
                confidence=1.0,
                details={'error': str(e)}
            )
        
        return None
    
    async def _detect_nsfw_content(self, image_data: bytes) -> Optional[ValidationIssue]:
        """Detect NSFW content in image."""
        try:
            # Convert to OpenCV format
            nparr = np.frombuffer(image_data, np.uint8)
            cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Basic NSFW detection using skin color detection
            nsfw_score = self._detect_skin_content(cv_image)
            
            if nsfw_score > self.nsfw_threshold:
                severity = 'critical' if nsfw_score > 0.7 else 'medium'
                return ValidationIssue(
                    severity=severity,
                    issue_type='nsfw',
                    message=f'Potential NSFW content detected (score: {nsfw_score:.2f})',
                    confidence=nsfw_score,
                    details={'nsfw_score': nsfw_score}
                )
        
        except Exception as e:
            logger.warning(f"NSFW detection failed: {e}")
        
        return None
    
    def _detect_skin_content(self, image: np.ndarray) -> float:
        """Basic skin color detection for NSFW screening."""
        try:
            # Convert to HSV for better skin detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define skin color range in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # Create mask for skin color
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Calculate percentage of skin-colored pixels
            skin_pixels = np.sum(skin_mask > 0)
            total_pixels = image.shape[0] * image.shape[1]
            skin_percentage = skin_pixels / total_pixels
            
            return min(1.0, skin_percentage * 2)  # Scale up for sensitivity
            
        except Exception as e:
            logger.warning(f"Skin detection failed: {e}")
            return 0.0
    
    async def _detect_copyright_content(self, image_data: bytes) -> Optional[ValidationIssue]:
        """Detect potential copyright issues (watermarks, logos)."""
        try:
            # Convert to OpenCV format
            nparr = np.frombuffer(image_data, np.uint8)
            cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Detect potential watermarks
            watermark_score = self._detect_watermarks(gray)
            
            # Detect potential logos
            logo_score = self._detect_logos(gray)
            
            max_score = max(watermark_score, logo_score)
            
            if max_score > self.copyright_threshold:
                return ValidationIssue(
                    severity='high',
                    issue_type='copyright',
                    message=f'Potential copyrighted content detected (score: {max_score:.2f})',
                    confidence=max_score,
                    details={
                        'watermark_score': watermark_score,
                        'logo_score': logo_score
                    }
                )
        
        except Exception as e:
            logger.warning(f"Copyright detection failed: {e}")
        
        return None
    
    def _detect_watermarks(self, gray_image: np.ndarray) -> float:
        """Detect potential watermarks in image."""
        try:
            # Look for semi-transparent overlays (common in watermarks)
            # Use edge detection to find potential watermark boundaries
            edges = cv2.Canny(gray_image, 50, 150)
            
            # Look for repeating patterns (common in watermarks)
            # Use template matching for common watermark locations
            h, w = gray_image.shape
            
            # Check corners and center for watermark-like features
            regions = [
                gray_image[0:h//4, 0:w//4],  # Top-left
                gray_image[0:h//4, -w//4:],  # Top-right
                gray_image[-h//4:, 0:w//4],  # Bottom-left
                gray_image[-h//4:, -w//4:],  # Bottom-right
                gray_image[h//3:2*h//3, w//3:2*w//3]  # Center
            ]
            
            watermark_score = 0.0
            for region in regions:
                if region.size > 0:
                    # Calculate standard deviation (watermarks often have low contrast)
                    std_dev = np.std(region)
                    if std_dev < 20:  # Low contrast region
                        watermark_score += 0.2
            
            return min(1.0, watermark_score)
            
        except Exception as e:
            logger.warning(f"Watermark detection failed: {e}")
            return 0.0
    
    def _detect_logos(self, gray_image: np.ndarray) -> float:
        """Detect potential logos in image."""
        try:
            # Use SIFT to detect distinctive features (logos often have unique features)
            keypoints = self.sift_detector.detect(gray_image, None)
            
            # Calculate logo likelihood based on feature density
            feature_density = len(keypoints) / (gray_image.shape[0] * gray_image.shape[1])
            
            # Look for high-contrast regions (logos are often high contrast)
            _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check for rectangular regions (common logo shapes)
            rectangular_regions = 0
            for contour in contours:
                if cv2.contourArea(contour) > 1000:  # Significant size
                    # Approximate contour to polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) == 4:  # Rectangular
                        rectangular_regions += 1
            
            logo_score = min(1.0, (feature_density * 1000) + (rectangular_regions * 0.1))
            
            return logo_score
            
        except Exception as e:
            logger.warning(f"Logo detection failed: {e}")
            return 0.0
    
    def _check_brand_safety(self, image: ImageResult) -> Optional[ValidationIssue]:
        """Check brand safety using keywords and metadata."""
        unsafe_content_score = 0.0
        found_keywords = []
        
        # Check title and description for unsafe keywords
        text_to_check = [
            image.title or '',
            image.description or '',
            str(image.metadata.get('tags', ''))
        ]
        
        for text in text_to_check:
            text_lower = text.lower()
            for keyword in self.unsafe_keywords:
                if keyword in text_lower:
                    unsafe_content_score += 0.1
                    found_keywords.append(keyword)
        
        if unsafe_content_score > 0.3:
            return ValidationIssue(
                severity='medium',
                issue_type='brand_safety',
                message=f'Potential brand safety issue (found keywords: {", ".join(found_keywords)})',
                confidence=min(1.0, unsafe_content_score),
                details={
                    'unsafe_keywords': found_keywords,
                    'safety_score': unsafe_content_score
                }
            )
        
        return None
    
    async def _validate_image_quality(self, image_data: bytes, image: ImageResult) -> Optional[ValidationIssue]:
        """Validate basic image quality requirements."""
        try:
            # Check file size (too small might indicate low quality)
            if len(image_data) < 10000:  # Less than 10KB
                return ValidationIssue(
                    severity='medium',
                    issue_type='quality',
                    message='Image file size too small, may indicate low quality',
                    confidence=0.8,
                    details={'file_size_bytes': len(image_data)}
                )
            
            # Check dimensions
            if image.width < 200 or image.height < 200:
                return ValidationIssue(
                    severity='medium',
                    issue_type='quality',
                    message=f'Image dimensions too small ({image.width}x{image.height})',
                    confidence=1.0,
                    details={
                        'width': image.width,
                        'height': image.height
                    }
                )
            
            # Check aspect ratio (extreme ratios might be problematic)
            aspect_ratio = image.width / image.height if image.height > 0 else 1
            if aspect_ratio > 5 or aspect_ratio < 0.2:
                return ValidationIssue(
                    severity='low',
                    issue_type='quality',
                    message=f'Unusual aspect ratio: {aspect_ratio:.2f}',
                    confidence=0.6,
                    details={'aspect_ratio': aspect_ratio}
                )
        
        except Exception as e:
            logger.warning(f"Quality validation failed: {e}")
        
        return None
    
    def _calculate_safety_score(self, issues: List[ValidationIssue]) -> float:
        """Calculate overall safety score based on issues found."""
        if not issues:
            return 1.0
        
        # Weight issues by severity
        severity_weights = {
            'critical': 1.0,
            'high': 0.7,
            'medium': 0.4,
            'low': 0.2
        }
        
        total_deduction = 0.0
        for issue in issues:
            weight = severity_weights.get(issue.severity, 0.5)
            total_deduction += weight * issue.confidence
        
        # Calculate final score (0.0 = unsafe, 1.0 = safe)
        safety_score = max(0.0, 1.0 - (total_deduction / 2.0))
        return safety_score
    
    async def _download_image(self, url: str) -> bytes:
        """Download image data for analysis."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Failed to download image from {url}: {e}")
            raise
    
    def get_validation_statistics(self, results: List[ValidationResult]) -> Dict:
        """Get statistics about validation results."""
        if not results:
            return {}
        
        total_images = len(results)
        valid_images = sum(1 for r in results if r.is_valid)
        
        # Count issues by type
        issue_counts = {}
        severity_counts = {}
        
        for result in results:
            for issue in result.issues:
                issue_counts[issue.issue_type] = issue_counts.get(issue.issue_type, 0) + 1
                severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
        
        # Calculate average safety score
        avg_safety_score = np.mean([r.safety_score for r in results])
        
        return {
            'total_images': total_images,
            'valid_images': valid_images,
            'invalid_images': total_images - valid_images,
            'validation_rate': valid_images / total_images if total_images > 0 else 0,
            'average_safety_score': avg_safety_score,
            'issue_counts': issue_counts,
            'severity_counts': severity_counts,
            'validation_version': '1.0'
        } 