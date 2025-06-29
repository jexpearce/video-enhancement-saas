"""
Animation Timeline Management

Manages the sequence of animation events and provides utilities for
timeline optimization, conflict resolution, and event synchronization.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class AnimationEvent:
    """Represents a single animation event in the timeline."""
    
    type: str                    # Type of animation (image_entry, image_exit, etc.)
    target_id: str              # ID of the element being animated
    start_time: float           # Start time in seconds
    duration: float             # Duration in seconds
    properties: Dict[str, Any]  # Animation properties (easing, from/to states, etc.)
    priority: float = 0.5       # Priority for conflict resolution (0-1)
    
    @property
    def end_time(self) -> float:
        """Calculate end time of the animation."""
        return self.start_time + self.duration
    
    def overlaps_with(self, other: 'AnimationEvent') -> bool:
        """Check if this event overlaps with another event."""
        return (self.start_time < other.end_time and 
                self.end_time > other.start_time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'type': self.type,
            'target_id': self.target_id,
            'start_time': self.start_time,
            'duration': self.duration,
            'properties': self.properties,
            'priority': self.priority
        }


class AnimationTimeline:
    """
    Manages a complete animation timeline with events, optimization,
    and conflict resolution capabilities.
    """
    
    def __init__(self, duration: float = 0.0):
        """Initialize timeline with total duration."""
        self.duration = duration
        self.events: List[AnimationEvent] = []
        self._cached_events_by_time = {}
        self._needs_cache_refresh = True
    
    def add_event(self, event: AnimationEvent):
        """Add a single animation event to the timeline."""
        self.events.append(event)
        self._needs_cache_refresh = True
        
        # Update timeline duration if needed
        if event.end_time > self.duration:
            self.duration = event.end_time
    
    def add_events(self, events: List[AnimationEvent]):
        """Add multiple animation events to the timeline."""
        self.events.extend(events)
        self._needs_cache_refresh = True
        
        # Update timeline duration
        if events:
            max_end_time = max(event.end_time for event in events)
            if max_end_time > self.duration:
                self.duration = max_end_time
    
    def get_events_at_time(
        self, 
        time: float, 
        tolerance: float = 0.1,
        event_type: Optional[str] = None
    ) -> List[AnimationEvent]:
        """Get all events occurring at a specific time."""
        
        matching_events = []
        
        for event in self.events:
            # Check if event is active at the specified time
            if (event.start_time <= time <= event.end_time or 
                abs(event.start_time - time) <= tolerance):
                
                # Filter by event type if specified
                if event_type is None or event.type == event_type:
                    matching_events.append(event)
        
        return matching_events
    
    def get_events_by_target(self, target_id: str) -> List[AnimationEvent]:
        """Get all events for a specific target element."""
        return [event for event in self.events if event.target_id == target_id]
    
    def get_events_by_type(self, event_type: str) -> List[AnimationEvent]:
        """Get all events of a specific type."""
        return [event for event in self.events if event.type == event_type]
    
    def resolve_conflicts(self):
        """Resolve conflicts between overlapping animations."""
        
        # Group events by target
        target_groups = {}
        for event in self.events:
            if event.target_id not in target_groups:
                target_groups[event.target_id] = []
            target_groups[event.target_id].append(event)
        
        # Resolve conflicts within each target group
        resolved_events = []
        
        for target_id, target_events in target_groups.items():
            # Sort by start time
            target_events.sort(key=lambda e: e.start_time)
            
            # Check for overlaps and resolve
            non_conflicting = []
            
            for event in target_events:
                conflicts = [e for e in non_conflicting if event.overlaps_with(e)]
                
                if conflicts:
                    # Resolve conflict by priority
                    highest_priority_event = max(
                        [event] + conflicts, 
                        key=lambda e: e.priority
                    )
                    
                    # Remove lower priority events
                    non_conflicting = [e for e in non_conflicting if e not in conflicts]
                    
                    # Add highest priority event if it's the new one
                    if highest_priority_event == event:
                        non_conflicting.append(event)
                else:
                    non_conflicting.append(event)
            
            resolved_events.extend(non_conflicting)
        
        self.events = resolved_events
        self._needs_cache_refresh = True
    
    def merge_similar_events(self):
        """Merge similar consecutive events to optimize performance."""
        
        if len(self.events) < 2:
            return
        
        # Sort events by start time
        self.events.sort(key=lambda e: e.start_time)
        
        merged_events = []
        current_event = self.events[0]
        
        for next_event in self.events[1:]:
            # Check if events can be merged
            if self._can_merge_events(current_event, next_event):
                # Merge events
                current_event = self._merge_events(current_event, next_event)
            else:
                # Can't merge, add current and move to next
                merged_events.append(current_event)
                current_event = next_event
        
        # Add the last event
        merged_events.append(current_event)
        
        self.events = merged_events
        self._needs_cache_refresh = True
    
    def _can_merge_events(self, event1: AnimationEvent, event2: AnimationEvent) -> bool:
        """Check if two events can be merged."""
        
        return (
            event1.target_id == event2.target_id and
            event1.type == event2.type and
            abs(event1.end_time - event2.start_time) < 0.1 and  # Very close timing
            event1.properties.get('animation') == event2.properties.get('animation')
        )
    
    def _merge_events(self, event1: AnimationEvent, event2: AnimationEvent) -> AnimationEvent:
        """Merge two compatible events into one."""
        
        return AnimationEvent(
            type=event1.type,
            target_id=event1.target_id,
            start_time=event1.start_time,
            duration=event2.end_time - event1.start_time,
            properties=event1.properties.copy(),  # Use first event's properties
            priority=max(event1.priority, event2.priority)
        )
    
    def optimize(self):
        """Perform complete timeline optimization."""
        
        # 1. Resolve conflicts
        self.resolve_conflicts()
        
        # 2. Merge similar events
        self.merge_similar_events()
        
        # 3. Sort by start time for efficient playback
        self.events.sort(key=lambda e: e.start_time)
        
        self._needs_cache_refresh = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert timeline to dictionary for serialization."""
        
        return {
            'duration': self.duration,
            'events': [event.to_dict() for event in self.events],
        }
    
    def clear(self):
        """Clear all events from the timeline."""
        self.events.clear()
        self.duration = 0.0
        self._needs_cache_refresh = True
    
    def __len__(self) -> int:
        """Return the number of events in the timeline."""
        return len(self.events)
    
    def __iter__(self):
        """Iterate over events in the timeline."""
        return iter(self.events)
    
    def __repr__(self) -> str:
        """String representation of the timeline."""
        return f"AnimationTimeline(duration={self.duration:.2f}s, events={len(self.events)})" 