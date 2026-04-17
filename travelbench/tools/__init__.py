"""
Tool modules for the travel benchmark framework.
This module imports all available tool classes and handles their centralized registration.
"""

# Import sandbox_tool_registry for centralized registration
from ..core.tools import sandbox_tool_registry

from . import travel_search_flights
from . import travel_search_trains
from . import weather_current_conditions
from . import weather_forecast_days
from . import map_compute_routes
from . import map_search_places
from . import map_search_along_route
from . import map_search_central_places
from . import map_search_ranking_list
from . import web_search

# Import the tool list for runtime filtering
from .tool_list import TOOL_NAMES

# Re-export specific tool instances for direct access
from .travel_search_flights import search_flights_tool
from .travel_search_trains import search_trains_tool
from .weather_current_conditions import weather_current_tool
from .weather_forecast_days import weather_forecast_tool
from .map_compute_routes import map_compute_routes_tool
from .map_search_places import map_search_places_tool
from .map_search_along_route import map_search_along_route_tool
from .map_search_central_places import map_search_central_places_tool
from .map_search_ranking_list import map_search_ranking_list_tool
from .web_search import web_search_tool

# Centralized tool registration - register all tools here
def register_all_tools():
    """Register all tools with the sandbox tool registry."""
    tools_to_register = [
        search_flights_tool,
        search_trains_tool,
        weather_current_tool,
        weather_forecast_tool,
        map_compute_routes_tool,
        map_search_places_tool,
        map_search_along_route_tool,
        map_search_central_places_tool,
        map_search_ranking_list_tool,
        web_search_tool,
    ]
    
    for tool in tools_to_register:
        try:
            sandbox_tool_registry.register(tool)
        except ValueError as e:
            if "already registered" not in str(e):
                raise e  # Re-raise if it's not a duplicate registration error

# Automatically register all tools when this module is imported
register_all_tools()

# Define what gets exported with 'from .tools import *'
__all__ = [
    'TOOL_NAMES',
    'search_flights_tool',
    'search_trains_tool',
    'weather_current_tool',
    'weather_forecast_tool',
    'map_compute_routes_tool',
    'map_search_places_tool',
    'map_search_along_route_tool',
    'map_search_central_places_tool',
    'map_search_ranking_list_tool',
    'web_search_tool',
]


