"""
Wikimedia Commons Integration - Phase 2

Advanced Wikimedia Commons image provider with:
- Comprehensive metadata extraction
- Accurate license detection
- Quality filtering
- Category-based relevance scoring
"""

import aiohttp
from typing import List, Dict, Optional
import urllib.parse
import logging
import asyncio
import re
from datetime import datetime

from .base import (
    ImageProvider, 
    ImageResult, 
    ImageLicense, 
    SearchRequest,
    RateLimitError,
    LicenseValidationError,
    ImageUnavailableError
)

logger = logging.getLogger(__name__)

class WikimediaProvider(ImageProvider):
    """Wikimedia Commons image provider with advanced features"""
    
    def __init__(self, user_agent: str = "VideoEnhancementSaaS/1.0"):
        self.base_url = "https://commons.wikimedia.org/w/api.php"
        self.headers = {"User-Agent": user_agent}
        self.session: Optional[aiohttp.ClientSession] = None
        
        # License mappings for accurate detection
        self.license_mapping = {
            'cc0': ImageLicense.CREATIVE_COMMONS_ZERO,
            'cc-0': ImageLicense.CREATIVE_COMMONS_ZERO,
            'cc-by': ImageLicense.CREATIVE_COMMONS_BY,
            'cc-by-sa': ImageLicense.CREATIVE_COMMONS_BY_SA,
            'public domain': ImageLicense.PUBLIC_DOMAIN,
            'pd': ImageLicense.PUBLIC_DOMAIN,
            'pd-old': ImageLicense.PUBLIC_DOMAIN,
            'pd-art': ImageLicense.PUBLIC_DOMAIN,
            'pd-us': ImageLicense.PUBLIC_DOMAIN
        }
        
        # Quality indicators from categories
        self.quality_categories = {
            'featured': 1.0,
            'quality': 0.9, 
            'valued': 0.8,
            'good': 0.7
        }
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            
    @property
    def name(self) -> str:
        return "wikimedia"
        
    @property
    def rate_limit(self) -> Dict[str, int]:
        return {
            "requests_per_second": 10,
            "requests_per_hour": 3600,
            "concurrent_requests": 5
        }
        
    async def search(self, query: str, count: int = 10) -> List[ImageResult]:
        """
        Search Wikimedia Commons with advanced filtering.
        
        Args:
            query: Search query string
            count: Maximum number of results
            
        Returns:
            List of ImageResult objects
        """
        if not self.session:
            async with self:
                return await self._perform_search(query, count)
        else:
            return await self._perform_search(query, count)
            
    async def _perform_search(self, query: str, count: int) -> List[ImageResult]:
        """Perform the actual search with error handling"""
        try:
            # 1. Search for files with advanced parameters
            search_params = {
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": f"filetype:bitmap|drawing {query}",
                "gsrnamespace": "6",  # File namespace
                "gsrlimit": min(count * 3, 50),  # Get extra for filtering
                "prop": "imageinfo|categories|info",
                "iiprop": "url|size|mime|extmetadata|canonicaltitle|timestamp",
                "iiurlwidth": 1024,  # Get reasonable thumbnail
                "iiurlheight": 768,
                "inprop": "url"
            }
            
            if not self.session:
                raise RuntimeError("Session not initialized")
            async with self.session.get(self.base_url, params=search_params) as response:
                if response.status == 429:
                    raise RateLimitError("Wikimedia API rate limit exceeded")
                    
                response.raise_for_status()
                data = await response.json()
                
            # 2. Process results with comprehensive metadata extraction
            images = []
            if 'query' in data and 'pages' in data['query']:
                for page in data['query']['pages'].values():
                    try:
                        image = await self._process_page_result(page, query)
                        if image:
                            images.append(image)
                    except Exception as e:
                        logger.warning(f"Failed to process Wikimedia page: {e}")
                        continue
                        
            # 3. Sort by quality and relevance, return top N with safe type conversion
            def safe_combined_score(x: ImageResult) -> float:
                try:
                    quality = float(x.quality_score) if x.quality_score is not None else 0.0
                    relevance = float(x.relevance_score) if x.relevance_score is not None else 0.0
                    return (quality + relevance) / 2
                except (ValueError, TypeError):
                    return 0.0
            
            images.sort(key=safe_combined_score, reverse=True)
            return images[:count]
            
        except aiohttp.ClientError as e:
            logger.error(f"Wikimedia API error: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in Wikimedia search: {e}")
            return []
            
    async def _process_page_result(self, page: Dict, query: str) -> Optional[ImageResult]:
        """Process a single page result into ImageResult"""
        
        if 'imageinfo' not in page or not page['imageinfo']:
            return None
            
        image_info = page['imageinfo'][0]
        metadata = image_info.get('extmetadata', {})
        
        # Skip non-image files
        if not image_info.get('mime', '').startswith('image/'):
            return None
            
        # Extract comprehensive metadata
        title = page.get('title', '').replace('File:', '')
        description = self._extract_description(metadata)
        author_info = self._extract_author_info(metadata)
        license_info = self._extract_license_info(metadata)
        
        # Skip if license cannot be determined or is problematic
        if license_info['license'] == ImageLicense.UNKNOWN:
            return None
            
        # Calculate quality score
        quality_score = self._calculate_quality_score(image_info, metadata, page)
        
        # Skip very low quality images
        if quality_score < 0.2:
            return None
            
        # Calculate relevance score
        relevance_score = self._calculate_relevance_score(
            title, description, query, page.get('categories', [])
        )
        
        # Create ImageResult
        try:
            return ImageResult(
                provider="wikimedia",
                image_url=image_info['url'],
                thumbnail_url=image_info.get('thumburl', image_info['url']),
                title=title,
                description=description,
                author=author_info['name'],
                author_url=author_info['url'],
                license=license_info['license'],
                license_url=license_info['url'],
                width=image_info['width'],
                height=image_info['height'],
                relevance_score=relevance_score,
                quality_score=quality_score,
                file_size=image_info.get('size'),
                mime_type=image_info.get('mime'),
                created_at=self._parse_timestamp(image_info.get('timestamp')),
                metadata={
                    'page_id': page['pageid'],
                    'categories': [cat['title'] for cat in page.get('categories', [])],
                    'commons_url': f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(page['title'])}",
                    'original_metadata': metadata
                }
            )
        except ValueError as e:
            logger.warning(f"Invalid image data from Wikimedia: {e}")
            return None
            
    def _extract_description(self, metadata: Dict) -> Optional[str]:
        """Extract image description from metadata"""
        # Try multiple description fields
        for field in ['ImageDescription', 'ObjectName', 'Caption']:
            if field in metadata and 'value' in metadata[field]:
                desc = metadata[field]['value']
                # Clean HTML tags and excessive markup
                desc = re.sub(r'<[^>]+>', '', desc)
                desc = re.sub(r'\{\{[^}]+\}\}', '', desc)
                desc = desc.strip()
                if desc and len(desc) > 10:
                    return desc[:500]  # Limit length
        return None
        
    def _extract_author_info(self, metadata: Dict) -> Dict[str, Optional[str]]:
        """Extract author information from metadata"""
        author_name = None
        author_url = None
        
        # Try to extract author name
        for field in ['Artist', 'Credit', 'Author']:
            if field in metadata and 'value' in metadata[field]:
                author_raw = metadata[field]['value']
                # Clean markup
                author_clean = re.sub(r'<[^>]+>', '', author_raw)
                author_clean = re.sub(r'\[\[[^]]+\]\]', '', author_clean)
                author_clean = author_clean.strip()
                if author_clean:
                    author_name = author_clean[:100]  # Limit length
                    break
                    
        # Try to extract author URL
        if 'Artist' in metadata and 'source' in metadata['Artist']:
            source = metadata['Artist']['source']
            url_match = re.search(r'https?://[^\s\]]+', source)
            if url_match:
                author_url = url_match.group()
                
        return {'name': author_name, 'url': author_url}
        
    def _extract_license_info(self, metadata: Dict) -> Dict:
        """Extract and normalize license information"""
        license_type = ImageLicense.UNKNOWN
        license_url = None
        
        # Check license short name first
        if 'LicenseShortName' in metadata and 'value' in metadata['LicenseShortName']:
            license_text = metadata['LicenseShortName']['value'].lower()
            
            for key, license_enum in self.license_mapping.items():
                if key in license_text:
                    license_type = license_enum
                    break
                    
        # Get license URL
        if 'LicenseUrl' in metadata and 'value' in metadata['LicenseUrl']:
            license_url = metadata['LicenseUrl']['value']
            
        # If no license found in short name, check copyright status
        if license_type == ImageLicense.UNKNOWN and 'Copyrighted' in metadata:
            copyright_status = metadata['Copyrighted']['value'].lower()
            if 'false' in copyright_status or 'public domain' in copyright_status:
                license_type = ImageLicense.PUBLIC_DOMAIN
                
        return {
            'license': license_type,
            'url': license_url
        }
        
    def _calculate_quality_score(self, image_info: Dict, metadata: Dict, page: Dict) -> float:
        """Calculate comprehensive quality score"""
        
        # Start with base quality from dimensions
        quality = self.calculate_base_quality_score(
            image_info['width'],
            image_info['height'],
            {'file_size': image_info.get('size')}
        )
        
        # Bonus for quality categories
        categories = [cat['title'].lower() for cat in page.get('categories', [])]
        for cat_keyword, bonus in self.quality_categories.items():
            if any(cat_keyword in cat for cat in categories):
                quality = min(1.0, quality + 0.2)
                break
                
        # Penalty for problematic categories
        problematic_keywords = ['deletion', 'copyright', 'dispute', 'low quality']
        for keyword in problematic_keywords:
            if any(keyword in cat for cat in categories):
                quality *= 0.5
                break
                
        # Bonus for having good metadata
        if metadata.get('ImageDescription'):
            quality = min(1.0, quality + 0.1)
        if metadata.get('Artist'):
            quality = min(1.0, quality + 0.05)
            
        return quality
        
    def _calculate_relevance_score(self, title: str, description: Optional[str], 
                                 query: str, categories: List[Dict]) -> float:
        """Calculate relevance score based on text matching"""
        
        score = 0.0
        query_lower = query.lower()
        query_terms = query_lower.split()
        
        # Title matching (highest weight)
        title_lower = title.lower()
        title_matches = sum(1 for term in query_terms if term in title_lower)
        score += (title_matches / len(query_terms)) * 0.5
        
        # Description matching
        if description:
            desc_lower = description.lower()
            desc_matches = sum(1 for term in query_terms if term in desc_lower)
            score += (desc_matches / len(query_terms)) * 0.3
            
        # Category matching
        category_text = ' '.join([cat.get('title', '').lower() for cat in categories])
        cat_matches = sum(1 for term in query_terms if term in category_text)
        score += (cat_matches / len(query_terms)) * 0.2
        
        return min(1.0, score)
        
    def _parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[datetime]:
        """Parse Wikimedia timestamp to datetime"""
        if not timestamp_str:
            return None
            
        try:
            # Wikimedia format: 2023-01-15T10:30:00Z
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return None
            
    async def validate_license(self, image: ImageResult) -> bool:
        """Validate that the image license is correctly identified"""
        
        # For Wikimedia, we trust our license extraction
        # In production, you might want to re-verify critical images
        return image.license != ImageLicense.UNKNOWN
        
    async def check_availability(self, image_url: str) -> bool:
        """Check if image URL is accessible"""
        
        try:
            if not self.session:
                async with aiohttp.ClientSession() as session:
                    async with session.head(image_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        return response.status == 200
            else:
                async with self.session.head(image_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    return response.status == 200
        except Exception:
            return False 