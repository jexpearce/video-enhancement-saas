"""
Animation Easing Functions

Provides various easing functions for smooth and engaging animations.
Includes standard easing curves plus specialized effects for social media content.
"""

import math
from enum import Enum
from typing import Callable


class EasingType(Enum):
    """Standard easing function types."""
    
    # Basic easing
    LINEAR = "linear"
    
    # Quadratic easing
    EASE_IN_QUAD = "ease_in_quad"
    EASE_OUT_QUAD = "ease_out_quad"
    EASE_IN_OUT_QUAD = "ease_in_out_quad"
    
    # Cubic easing
    EASE_IN_CUBIC = "ease_in_cubic"
    EASE_OUT_CUBIC = "ease_out_cubic"
    EASE_IN_OUT_CUBIC = "ease_in_out_cubic"
    
    # Quartic easing
    EASE_IN_QUART = "ease_in_quart"
    EASE_OUT_QUART = "ease_out_quart"
    EASE_IN_OUT_QUART = "ease_in_out_quart"
    
    # Quintic easing
    EASE_IN_QUINT = "ease_in_quint"
    EASE_OUT_QUINT = "ease_out_quint"
    EASE_IN_OUT_QUINT = "ease_in_out_quint"
    
    # Sinusoidal easing
    EASE_IN_SINE = "ease_in_sine"
    EASE_OUT_SINE = "ease_out_sine"
    EASE_IN_OUT_SINE = "ease_in_out_sine"
    
    # Exponential easing
    EASE_IN_EXPO = "ease_in_expo"
    EASE_OUT_EXPO = "ease_out_expo"
    EASE_IN_OUT_EXPO = "ease_in_out_expo"
    
    # Circular easing
    EASE_IN_CIRC = "ease_in_circ"
    EASE_OUT_CIRC = "ease_out_circ"
    EASE_IN_OUT_CIRC = "ease_in_out_circ"
    
    # Back easing (overshoots)
    EASE_IN_BACK = "ease_in_back"
    EASE_OUT_BACK = "ease_out_back"
    EASE_IN_OUT_BACK = "ease_in_out_back"
    
    # Elastic easing
    EASE_IN_ELASTIC = "ease_in_elastic"
    EASE_OUT_ELASTIC = "ease_out_elastic"
    EASE_IN_OUT_ELASTIC = "ease_in_out_elastic"
    
    # Bounce easing
    EASE_IN_BOUNCE = "ease_in_bounce"
    EASE_OUT_BOUNCE = "ease_out_bounce"
    EASE_IN_OUT_BOUNCE = "ease_in_out_bounce"


class EasingFunction:
    """
    Provides easing functions for smooth animations.
    
    All easing functions take a value t in range [0, 1] and return
    the eased value also in range [0, 1].
    """
    
    def __init__(self):
        """Initialize easing function collection."""
        self._functions = {
            EasingType.LINEAR: self.linear,
            
            # Quadratic
            EasingType.EASE_IN_QUAD: self.ease_in_quad,
            EasingType.EASE_OUT_QUAD: self.ease_out_quad,
            EasingType.EASE_IN_OUT_QUAD: self.ease_in_out_quad,
            
            # Cubic
            EasingType.EASE_IN_CUBIC: self.ease_in_cubic,
            EasingType.EASE_OUT_CUBIC: self.ease_out_cubic,
            EasingType.EASE_IN_OUT_CUBIC: self.ease_in_out_cubic,
            
            # Quartic
            EasingType.EASE_IN_QUART: self.ease_in_quart,
            EasingType.EASE_OUT_QUART: self.ease_out_quart,
            EasingType.EASE_IN_OUT_QUART: self.ease_in_out_quart,
            
            # Quintic
            EasingType.EASE_IN_QUINT: self.ease_in_quint,
            EasingType.EASE_OUT_QUINT: self.ease_out_quint,
            EasingType.EASE_IN_OUT_QUINT: self.ease_in_out_quint,
            
            # Sinusoidal
            EasingType.EASE_IN_SINE: self.ease_in_sine,
            EasingType.EASE_OUT_SINE: self.ease_out_sine,
            EasingType.EASE_IN_OUT_SINE: self.ease_in_out_sine,
            
            # Exponential
            EasingType.EASE_IN_EXPO: self.ease_in_expo,
            EasingType.EASE_OUT_EXPO: self.ease_out_expo,
            EasingType.EASE_IN_OUT_EXPO: self.ease_in_out_expo,
            
            # Circular
            EasingType.EASE_IN_CIRC: self.ease_in_circ,
            EasingType.EASE_OUT_CIRC: self.ease_out_circ,
            EasingType.EASE_IN_OUT_CIRC: self.ease_in_out_circ,
            
            # Back
            EasingType.EASE_IN_BACK: self.ease_in_back,
            EasingType.EASE_OUT_BACK: self.ease_out_back,
            EasingType.EASE_IN_OUT_BACK: self.ease_in_out_back,
            
            # Elastic
            EasingType.EASE_IN_ELASTIC: self.ease_in_elastic,
            EasingType.EASE_OUT_ELASTIC: self.ease_out_elastic,
            EasingType.EASE_IN_OUT_ELASTIC: self.ease_in_out_elastic,
            
            # Bounce
            EasingType.EASE_IN_BOUNCE: self.ease_in_bounce,
            EasingType.EASE_OUT_BOUNCE: self.ease_out_bounce,
            EasingType.EASE_IN_OUT_BOUNCE: self.ease_in_out_bounce,
        }
    
    def get_function(self, easing_type: EasingType) -> Callable[[float], float]:
        """Get easing function by type."""
        return self._functions.get(easing_type, self.linear)
    
    def apply(self, easing_type: EasingType, t: float) -> float:
        """Apply easing function to input value."""
        # Clamp t to [0, 1]
        t = max(0.0, min(1.0, t))
        
        function = self.get_function(easing_type)
        return function(t)
    
    # Basic easing functions
    def linear(self, t: float) -> float:
        """Linear interpolation (no easing)."""
        return t
    
    # Quadratic easing
    def ease_in_quad(self, t: float) -> float:
        """Quadratic ease in."""
        return t * t
    
    def ease_out_quad(self, t: float) -> float:
        """Quadratic ease out."""
        return t * (2 - t)
    
    def ease_in_out_quad(self, t: float) -> float:
        """Quadratic ease in-out."""
        if t < 0.5:
            return 2 * t * t
        return -1 + (4 - 2 * t) * t
    
    # Cubic easing
    def ease_in_cubic(self, t: float) -> float:
        """Cubic ease in."""
        return t * t * t
    
    def ease_out_cubic(self, t: float) -> float:
        """Cubic ease out."""
        return (t - 1) * (t - 1) * (t - 1) + 1
    
    def ease_in_out_cubic(self, t: float) -> float:
        """Cubic ease in-out."""
        if t < 0.5:
            return 4 * t * t * t
        return (t - 1) * (2 * t - 2) * (2 * t - 2) + 1
    
    # Quartic easing
    def ease_in_quart(self, t: float) -> float:
        """Quartic ease in."""
        return t * t * t * t
    
    def ease_out_quart(self, t: float) -> float:
        """Quartic ease out."""
        return 1 - (t - 1) * (t - 1) * (t - 1) * (t - 1)
    
    def ease_in_out_quart(self, t: float) -> float:
        """Quartic ease in-out."""
        if t < 0.5:
            return 8 * t * t * t * t
        return 1 - 8 * (t - 1) * (t - 1) * (t - 1) * (t - 1)
    
    # Quintic easing
    def ease_in_quint(self, t: float) -> float:
        """Quintic ease in."""
        return t * t * t * t * t
    
    def ease_out_quint(self, t: float) -> float:
        """Quintic ease out."""
        return 1 + (t - 1) * (t - 1) * (t - 1) * (t - 1) * (t - 1)
    
    def ease_in_out_quint(self, t: float) -> float:
        """Quintic ease in-out."""
        if t < 0.5:
            return 16 * t * t * t * t * t
        return 1 + 16 * (t - 1) * (t - 1) * (t - 1) * (t - 1) * (t - 1)
    
    # Sinusoidal easing
    def ease_in_sine(self, t: float) -> float:
        """Sinusoidal ease in."""
        return 1 - math.cos(t * math.pi / 2)
    
    def ease_out_sine(self, t: float) -> float:
        """Sinusoidal ease out."""
        return math.sin(t * math.pi / 2)
    
    def ease_in_out_sine(self, t: float) -> float:
        """Sinusoidal ease in-out."""
        return -(math.cos(math.pi * t) - 1) / 2
    
    # Exponential easing
    def ease_in_expo(self, t: float) -> float:
        """Exponential ease in."""
        return 0 if t == 0 else math.pow(2, 10 * (t - 1))
    
    def ease_out_expo(self, t: float) -> float:
        """Exponential ease out."""
        return 1 if t == 1 else 1 - math.pow(2, -10 * t)
    
    def ease_in_out_expo(self, t: float) -> float:
        """Exponential ease in-out."""
        if t == 0:
            return 0
        if t == 1:
            return 1
        if t < 0.5:
            return math.pow(2, 20 * t - 10) / 2
        return (2 - math.pow(2, -20 * t + 10)) / 2
    
    # Circular easing
    def ease_in_circ(self, t: float) -> float:
        """Circular ease in."""
        return 1 - math.sqrt(1 - t * t)
    
    def ease_out_circ(self, t: float) -> float:
        """Circular ease out."""
        return math.sqrt(1 - (t - 1) * (t - 1))
    
    def ease_in_out_circ(self, t: float) -> float:
        """Circular ease in-out."""
        if t < 0.5:
            return (1 - math.sqrt(1 - 4 * t * t)) / 2
        return (math.sqrt(1 - (-2 * t + 2) * (-2 * t + 2)) + 1) / 2
    
    # Back easing (overshoots)
    def ease_in_back(self, t: float) -> float:
        """Back ease in (slight overshoot)."""
        c1 = 1.70158
        c3 = c1 + 1
        return c3 * t * t * t - c1 * t * t
    
    def ease_out_back(self, t: float) -> float:
        """Back ease out (slight overshoot)."""
        c1 = 1.70158
        c3 = c1 + 1
        return 1 + c3 * math.pow(t - 1, 3) + c1 * math.pow(t - 1, 2)
    
    def ease_in_out_back(self, t: float) -> float:
        """Back ease in-out (slight overshoot)."""
        c1 = 1.70158
        c2 = c1 * 1.525
        
        if t < 0.5:
            return (math.pow(2 * t, 2) * ((c2 + 1) * 2 * t - c2)) / 2
        return (math.pow(2 * t - 2, 2) * ((c2 + 1) * (t * 2 - 2) + c2) + 2) / 2
    
    # Elastic easing
    def ease_in_elastic(self, t: float) -> float:
        """Elastic ease in."""
        if t == 0:
            return 0
        if t == 1:
            return 1
        
        c4 = (2 * math.pi) / 3
        return -math.pow(2, 10 * t - 10) * math.sin((t * 10 - 10.75) * c4)
    
    def ease_out_elastic(self, t: float) -> float:
        """Elastic ease out."""
        if t == 0:
            return 0
        if t == 1:
            return 1
        
        c4 = (2 * math.pi) / 3
        return math.pow(2, -10 * t) * math.sin((t * 10 - 0.75) * c4) + 1
    
    def ease_in_out_elastic(self, t: float) -> float:
        """Elastic ease in-out."""
        if t == 0:
            return 0
        if t == 1:
            return 1
        
        c5 = (2 * math.pi) / 4.5
        
        if t < 0.5:
            return -(math.pow(2, 20 * t - 10) * math.sin((20 * t - 11.125) * c5)) / 2
        return (math.pow(2, -20 * t + 10) * math.sin((20 * t - 11.125) * c5)) / 2 + 1
    
    # Bounce easing
    def ease_in_bounce(self, t: float) -> float:
        """Bounce ease in."""
        return 1 - self.ease_out_bounce(1 - t)
    
    def ease_out_bounce(self, t: float) -> float:
        """Bounce ease out."""
        n1 = 7.5625
        d1 = 2.75
        
        if t < 1 / d1:
            return n1 * t * t
        elif t < 2 / d1:
            return n1 * (t - 1.5 / d1) * (t - 1.5 / d1) + 0.75
        elif t < 2.5 / d1:
            return n1 * (t - 2.25 / d1) * (t - 2.25 / d1) + 0.9375
        else:
            return n1 * (t - 2.625 / d1) * (t - 2.625 / d1) + 0.984375
    
    def ease_in_out_bounce(self, t: float) -> float:
        """Bounce ease in-out."""
        if t < 0.5:
            return (1 - self.ease_out_bounce(1 - 2 * t)) / 2
        return (1 + self.ease_out_bounce(2 * t - 1)) / 2
    
    # Social media specific easing
    def tiktok_bounce(self, t: float) -> float:
        """TikTok-style bounce effect."""
        # More aggressive bounce for viral content
        return self.ease_out_bounce(t) * 1.1
    
    def instagram_smooth(self, t: float) -> float:
        """Instagram-style smooth transition."""
        # Polished, smooth easing for Instagram content
        return self.ease_in_out_cubic(t)
    
    def viral_pop(self, t: float) -> float:
        """Viral content pop effect."""
        # Quick snap with slight overshoot for viral content
        if t < 0.8:
            return self.ease_out_back(t / 0.8) * 1.05
        else:
            # Quick settle
            return 1.05 - 0.05 * self.ease_out_quad((t - 0.8) / 0.2) 