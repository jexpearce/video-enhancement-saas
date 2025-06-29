"""
Style System Data Models - Optimized for TikTok/Instagram Creators.

Defines trendy, eye-catching visual styles with automatic selection
and manual override capabilities.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json


class Platform(Enum):
    """Target social media platforms."""
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram" 
    YOUTUBE_SHORTS = "youtube_shorts"
    SNAPCHAT = "snapchat"
    UNIVERSAL = "universal"


class ContentType(Enum):
    """Video content categories for style matching."""
    DANCE = "dance"
    COMEDY = "comedy"
    LIFESTYLE = "lifestyle"
    EDUCATION = "education"
    BUSINESS = "business"
    TRAVEL = "travel"
    FOOD = "food"
    FITNESS = "fitness"
    FASHION = "fashion"
    TECH = "tech"
    MUSIC = "music"
    GAMING = "gaming"


class AnimationType(Enum):
    """Animation effects for image transitions."""
    # Trendy TikTok/Instagram animations
    BOUNCE_IN = "bounce_in"           # Bouncy entrance
    SLIDE_UP = "slide_up"             # Slide from bottom  
    ZOOM_BLAST = "zoom_blast"         # Explosive zoom
    NEON_FLASH = "neon_flash"         # Neon glow effect
    GLITCH_POP = "glitch_pop"         # Glitch transition
    FLOAT_DRIFT = "float_drift"       # Floating motion
    SPIN_REVEAL = "spin_reveal"       # Spinning entrance
    PULSE_BEAT = "pulse_beat"         # Pulsing to music
    WAVE_RIPPLE = "wave_ripple"       # Ripple effect
    PARTICLE_BURST = "particle_burst" # Particle explosion
    
    # Classic effects
    FADE_IN = "fade_in"
    SCALE_UP = "scale_up"
    ROTATE_IN = "rotate_in"


class PositionStyle(Enum):
    """Image positioning styles."""
    TOP_RIGHT = "top_right"           # Classic corner
    TOP_LEFT = "top_left" 
    BOTTOM_RIGHT = "bottom_right"
    BOTTOM_LEFT = "bottom_left"
    CENTER_OVERLAY = "center_overlay" # Full center focus
    SIDE_PANEL = "side_panel"         # Side strip
    FLOATING = "floating"             # Dynamic positioning
    FULLSCREEN_BEHIND = "fullscreen_behind" # Background layer


@dataclass
class ColorScheme:
    """Trendy color schemes for social media."""
    
    name: str
    primary: str              # Main brand color
    secondary: str            # Accent color  
    background: str           # Background color
    text_primary: str         # Main text color
    text_secondary: str       # Secondary text
    accent_glow: str          # Neon/glow effects
    gradient_start: str       # Gradient beginning
    gradient_end: str         # Gradient end
    
    # Engagement-boosting colors
    highlight: str            # Attention grabber
    shadow: str               # Drop shadows
    
    @classmethod
    def create_neon_vibe(cls) -> 'ColorScheme':
        """Neon/cyber aesthetic - very popular on TikTok."""
        return cls(
            name="Neon Vibe",
            primary="#FF0080",           # Hot pink
            secondary="#00FFFF",         # Cyan
            background="#0A0A0A",        # Almost black
            text_primary="#FFFFFF",      # White
            text_secondary="#CCCCCC",    # Light gray
            accent_glow="#FF0080",       # Pink glow
            gradient_start="#FF0080",    # Pink to cyan
            gradient_end="#00FFFF",
            highlight="#FFFF00",         # Yellow highlight
            shadow="#000000"
        )
    
    @classmethod
    def create_sunset_gradient(cls) -> 'ColorScheme':
        """Instagram-style sunset gradients."""
        return cls(
            name="Sunset Gradient",
            primary="#FF6B6B",           # Coral
            secondary="#4ECDC4",         # Teal
            background="#FFE66D",        # Warm yellow
            text_primary="#2C3E50",      # Dark blue
            text_secondary="#34495E",    # Slate
            accent_glow="#FF9F43",       # Orange glow
            gradient_start="#FF9F43",    # Orange to pink
            gradient_end="#FF6B6B",
            highlight="#F39C12",         # Bright orange
            shadow="#2C3E50"
        )
    
    @classmethod
    def create_pastel_dream(cls) -> 'ColorScheme':
        """Soft pastel aesthetic for lifestyle content."""
        return cls(
            name="Pastel Dream",
            primary="#FF99CC",           # Soft pink
            secondary="#99CCFF",         # Sky blue
            background="#F8F9FA",        # Almost white
            text_primary="#2C3E50",      # Dark blue
            text_secondary="#6C757D",    # Gray
            accent_glow="#FFB3E6",       # Pink glow
            gradient_start="#FF99CC",    # Pink to blue
            gradient_end="#99CCFF",
            highlight="#FFD700",         # Gold
            shadow="#E9ECEF"
        )


@dataclass 
class TypographyStyle:
    """Typography settings for text overlays."""
    
    font_family: str
    font_size: int              # Base size in pixels
    font_weight: str            # "normal", "bold", "black"
    text_transform: str         # "none", "uppercase", "lowercase"
    letter_spacing: float       # Letter spacing in pixels
    line_height: float          # Line height multiplier
    
    # TikTok/Instagram specific
    has_outline: bool           # Text outline for readability
    outline_color: str          # Outline color
    outline_width: int          # Outline thickness
    has_shadow: bool            # Drop shadow
    shadow_offset: Tuple[int, int] # (x, y) shadow offset
    shadow_blur: int            # Shadow blur radius
    
    @classmethod
    def create_bold_impact(cls) -> 'TypographyStyle':
        """Bold, attention-grabbing text for viral content."""
        return cls(
            font_family="Montserrat Black",
            font_size=32,
            font_weight="black", 
            text_transform="uppercase",
            letter_spacing=2.0,
            line_height=1.2,
            has_outline=True,
            outline_color="#000000",
            outline_width=2,
            has_shadow=True,
            shadow_offset=(2, 2),
            shadow_blur=4
        )
    
    @classmethod
    def create_trendy_casual(cls) -> 'TypographyStyle':
        """Modern, casual font for lifestyle content."""
        return cls(
            font_family="Poppins",
            font_size=24,
            font_weight="semibold",
            text_transform="none",
            letter_spacing=0.5,
            line_height=1.4,
            has_outline=False,
            outline_color="#FFFFFF",
            outline_width=1,
            has_shadow=True,
            shadow_offset=(1, 1),
            shadow_blur=2
        )


@dataclass
class AnimationEffect:
    """Animation effect definition."""
    
    animation_type: AnimationType
    duration: float             # Animation duration in seconds
    delay: float                # Delay before animation starts
    easing: str                 # CSS easing function
    
    # Advanced properties
    scale_from: float           # Starting scale (1.0 = normal)
    scale_to: float             # Ending scale
    rotation_degrees: float     # Rotation amount
    opacity_from: float         # Starting opacity (0.0-1.0)
    opacity_to: float           # Ending opacity
    
    # Special effects
    has_glow: bool              # Glowing effect
    glow_color: str             # Glow color
    has_particles: bool         # Particle effects
    particle_count: int         # Number of particles
    
    @classmethod
    def create_viral_entrance(cls) -> 'AnimationEffect':
        """High-energy entrance animation for viral content."""
        return cls(
            animation_type=AnimationType.ZOOM_BLAST,
            duration=0.6,
            delay=0.0,
            easing="cubic-bezier(0.68, -0.55, 0.265, 1.55)",
            scale_from=0.1,
            scale_to=1.0,
            rotation_degrees=15,
            opacity_from=0.0,
            opacity_to=1.0,
            has_glow=True,
            glow_color="#FF0080",
            has_particles=True,
            particle_count=20
        )
    
    @classmethod
    def create_smooth_slide(cls) -> 'AnimationEffect':
        """Smooth, professional slide animation."""
        return cls(
            animation_type=AnimationType.SLIDE_UP,
            duration=0.8,
            delay=0.2,
            easing="ease-out",
            scale_from=1.0,
            scale_to=1.0,
            rotation_degrees=0,
            opacity_from=0.0,
            opacity_to=1.0,
            has_glow=False,
            glow_color="#FFFFFF",
            has_particles=False,
            particle_count=0
        )


@dataclass
class StyleTemplate:
    """Complete visual style template."""
    
    # Metadata
    template_id: str
    name: str
    description: str
    creator: str                # Template creator
    popularity_score: float     # How popular/trending it is
    
    # Target audience
    target_platforms: List[Platform]
    content_types: List[ContentType]
    target_demographics: List[str] # ["gen_z", "millennials", etc.]
    
    # Visual components
    color_scheme: ColorScheme
    typography: TypographyStyle
    position_style: PositionStyle
    entrance_animation: AnimationEffect
    exit_animation: AnimationEffect
    
    # Layout settings
    image_size_ratio: float     # Image size relative to screen (0.0-1.0)
    border_radius: int          # Rounded corners in pixels
    border_width: int           # Border thickness
    has_background_blur: bool   # Blur video behind image
    background_opacity: float   # Background opacity when blurred
    
    # Advanced effects
    has_ken_burns: bool         # Slow zoom/pan effect
    ken_burns_intensity: float  # How much zoom/pan
    has_parallax: bool          # Parallax scrolling effect
    parallax_speed: float       # Parallax movement speed
    
    # Engagement features
    has_pulse_to_beat: bool     # Pulse with music beat
    has_emoji_reactions: bool   # Add trending emojis
    emoji_set: List[str]        # Emojis to use
    
    def get_engagement_score(self) -> float:
        """Calculate how engaging this template is likely to be."""
        score = self.popularity_score
        
        # Boost for trendy features
        if self.has_pulse_to_beat:
            score += 0.2
        if self.entrance_animation.has_particles:
            score += 0.15
        if self.has_emoji_reactions:
            score += 0.1
        if self.color_scheme.name in ["Neon Vibe", "Sunset Gradient"]:
            score += 0.15
            
        return min(score, 1.0)


@dataclass
class VisualStyle:
    """Applied visual style for a specific video."""
    
    template: StyleTemplate
    video_id: str
    applied_at: str             # Timestamp
    
    # Customizations from base template
    custom_colors: Optional[ColorScheme] = None
    custom_position: Optional[PositionStyle] = None
    custom_animations: Optional[List[AnimationEffect]] = None
    
    # Creator preferences
    intensity_level: float = 1.0    # 0.0-1.0, how intense effects are
    speed_multiplier: float = 1.0   # Animation speed adjustment
    
    # Performance tracking
    engagement_metrics: Dict[str, float] = field(default_factory=dict)
    viewer_feedback: List[str] = field(default_factory=list)
    
    def apply_customizations(self) -> StyleTemplate:
        """Apply customizations to base template."""
        customized = StyleTemplate(
            template_id=f"{self.template.template_id}_custom",
            name=f"{self.template.name} (Custom)",
            description=self.template.description,
            creator=self.template.creator,
            popularity_score=self.template.popularity_score,
            target_platforms=self.template.target_platforms,
            content_types=self.template.content_types,
            target_demographics=self.template.target_demographics,
            color_scheme=self.custom_colors or self.template.color_scheme,
            typography=self.template.typography,
            position_style=self.custom_position or self.template.position_style,
            entrance_animation=self.template.entrance_animation,
            exit_animation=self.template.exit_animation,
            image_size_ratio=self.template.image_size_ratio,
            border_radius=self.template.border_radius,
            border_width=self.template.border_width,
            has_background_blur=self.template.has_background_blur,
            background_opacity=self.template.background_opacity,
            has_ken_burns=self.template.has_ken_burns,
            ken_burns_intensity=self.template.ken_burns_intensity * self.intensity_level,
            has_parallax=self.template.has_parallax,
            parallax_speed=self.template.parallax_speed * self.speed_multiplier,
            has_pulse_to_beat=self.template.has_pulse_to_beat,
            has_emoji_reactions=self.template.has_emoji_reactions,
            emoji_set=self.template.emoji_set
        )
        
        return customized


@dataclass
class StyleConfig:
    """Configuration for style system."""
    
    # Auto-selection settings
    enable_auto_selection: bool = True
    auto_selection_confidence_threshold: float = 0.7
    
    # Creator override settings
    allow_manual_override: bool = True
    save_creator_preferences: bool = True
    
    # Performance settings
    max_templates_to_analyze: int = 20
    cache_style_decisions: bool = True
    
    # A/B testing
    enable_ab_testing: bool = True
    test_percentage: float = 0.1
    
    # Trending features
    update_popularity_scores: bool = True
    trending_boost_factor: float = 1.5
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'enable_auto_selection': self.enable_auto_selection,
            'auto_selection_confidence_threshold': self.auto_selection_confidence_threshold,
            'allow_manual_override': self.allow_manual_override,
            'save_creator_preferences': self.save_creator_preferences,
            'max_templates_to_analyze': self.max_templates_to_analyze,
            'cache_style_decisions': self.cache_style_decisions,
            'enable_ab_testing': self.enable_ab_testing,
            'test_percentage': self.test_percentage,
            'update_popularity_scores': self.update_popularity_scores,
            'trending_boost_factor': self.trending_boost_factor
        }


# Predefined trendy templates for TikTok/Instagram creators
VIRAL_TEMPLATES = {
    "neon_cyber_2024": StyleTemplate(
        template_id="neon_cyber_2024",
        name="Neon Cyber 🌟", 
        description="Ultra-trendy neon aesthetic that's crushing it on TikTok",
        creator="viral_templates",
        popularity_score=0.95,
        target_platforms=[Platform.TIKTOK, Platform.INSTAGRAM],
        content_types=[ContentType.DANCE, ContentType.MUSIC, ContentType.GAMING],
        target_demographics=["gen_z", "young_millennials"],
        color_scheme=ColorScheme.create_neon_vibe(),
        typography=TypographyStyle.create_bold_impact(),
        position_style=PositionStyle.FLOATING,
        entrance_animation=AnimationEffect.create_viral_entrance(),
        exit_animation=AnimationEffect(
            animation_type=AnimationType.GLITCH_POP,
            duration=0.4,
            delay=0.0,
            easing="ease-in",
            scale_from=1.0,
            scale_to=0.0,
            rotation_degrees=-15,
            opacity_from=1.0,
            opacity_to=0.0,
            has_glow=True,
            glow_color="#00FFFF",
            has_particles=True,
            particle_count=15
        ),
        image_size_ratio=0.4,
        border_radius=20,
        border_width=3,
        has_background_blur=True,
        background_opacity=0.3,
        has_ken_burns=True,
        ken_burns_intensity=0.8,
        has_parallax=True,
        parallax_speed=0.5,
        has_pulse_to_beat=True,
        has_emoji_reactions=True,
        emoji_set=["⚡", "🔥", "✨", "💎", "🌟"]
    ),
    
    "sunset_aesthetic": StyleTemplate(
        template_id="sunset_aesthetic",
        name="Sunset Aesthetic 🌅",
        description="Instagram-worthy sunset vibes for lifestyle content", 
        creator="viral_templates",
        popularity_score=0.88,
        target_platforms=[Platform.INSTAGRAM, Platform.TIKTOK],
        content_types=[ContentType.LIFESTYLE, ContentType.TRAVEL, ContentType.FASHION],
        target_demographics=["millennials", "gen_z_female"],
        color_scheme=ColorScheme.create_sunset_gradient(),
        typography=TypographyStyle.create_trendy_casual(),
        position_style=PositionStyle.TOP_RIGHT,
        entrance_animation=AnimationEffect.create_smooth_slide(),
        exit_animation=AnimationEffect(
            animation_type=AnimationType.FADE_IN,
            duration=0.5,
            delay=0.0,
            easing="ease-out",
            scale_from=1.0,
            scale_to=0.9,
            rotation_degrees=0,
            opacity_from=1.0,
            opacity_to=0.0,
            has_glow=False,
            glow_color="#FFFFFF",
            has_particles=False,
            particle_count=0
        ),
        image_size_ratio=0.35,
        border_radius=15,
        border_width=0,
        has_background_blur=False,
        background_opacity=1.0,
        has_ken_burns=True,
        ken_burns_intensity=0.3,
        has_parallax=False,
        parallax_speed=0.0,
        has_pulse_to_beat=False,
        has_emoji_reactions=True,
        emoji_set=["🌅", "✨", "💕", "🦋", "🌸"]
    ),
    
    "minimal_clean": StyleTemplate(
        template_id="minimal_clean",
        name="Clean Minimal ⚪",
        description="Professional minimal style for business and education",
        creator="viral_templates", 
        popularity_score=0.75,
        target_platforms=[Platform.INSTAGRAM, Platform.YOUTUBE_SHORTS],
        content_types=[ContentType.EDUCATION, ContentType.BUSINESS, ContentType.TECH],
        target_demographics=["millennials", "gen_x", "professionals"],
        color_scheme=ColorScheme.create_pastel_dream(),
        typography=TypographyStyle.create_trendy_casual(),
        position_style=PositionStyle.SIDE_PANEL,
        entrance_animation=AnimationEffect.create_smooth_slide(),
        exit_animation=AnimationEffect.create_smooth_slide(),
        image_size_ratio=0.3,
        border_radius=8,
        border_width=1,
        has_background_blur=True,
        background_opacity=0.1,
        has_ken_burns=False,
        ken_burns_intensity=0.0,
        has_parallax=False,
        parallax_speed=0.0,
        has_pulse_to_beat=False,
        has_emoji_reactions=False,
        emoji_set=[]
    )
} 