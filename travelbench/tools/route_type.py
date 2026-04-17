from typing import Optional, Tuple

class RouteStrategyMapper:
    """Maps route preferences to internal route type codes."""
    
    # Base mapping table: (traffic_aware, route_modifier) -> route_type
    STRATEGY_MAP = {        False: { # traffic_aware=False (默认)
            "": 38, # 时间优先 (默认)
            "prefer_highways": 34, # 高速优先
            "avoid_highways": 35, # 避开高速
            "avoid_tolls": 36, # 避开收费
            "prefer_main_roads": 37, # 大路优先
            "avoid_highways+avoid_tolls": 42, # 避开收费+避开高速
        },
        True: { # traffic_aware=True
            "": 45,
            "prefer_highways": 39,
            "avoid_highways": 40,
            "avoid_tolls": 41,
            "prefer_main_roads": 44,
            "avoid_highways+avoid_tolls": 43,
        },
    }

    @staticmethod
    def get_route_type(traffic_aware: bool = False, route_modifiers: Optional[str] = None) -> Tuple[str, int]:
        """
        Args:
            traffic_aware: Whether to compute routes with live traffic conditions.
            route_modifiers: Route preference string.

        Returns:
            (error_message, route_type):
                error_message: Empty string if no error, otherwise an message
                               describing the invalid modifier and the fallback behavior.
                route_type: The mapped low-level route_type code.
        """
        traffic_aware = traffic_aware or False
        route_modifiers = route_modifiers or ""

        route_type = RouteStrategyMapper.STRATEGY_MAP[traffic_aware].get(route_modifiers)
        error_message = ""

        if route_type is None:
            route_type = RouteStrategyMapper.STRATEGY_MAP[traffic_aware][""]

            allowed_values = [
                k for k in RouteStrategyMapper.STRATEGY_MAP[traffic_aware].keys()
                if k != ""
            ]
            fallback = RouteStrategyMapper.STRATEGY_MAP[traffic_aware][""]

            error_message = (
                "Unsupported value for `route_modifiers`: "
                f'"{route_modifiers}". '
                "Supported values are: "
                + ", ".join(f'"{v}"' for v in allowed_values)
                + ' or an empty/omitted value for the default "time-priority" strategy. '
                "The invalid value has been ignored and the request has been mapped to "
                f"the default time-priority strategy."
            )

            # Fallback to time-priority strategy
            route_type = fallback

        return error_message, route_type
