"""
Route computation tools for the travel benchmark framework.
Provides route planning with sandbox caching support via MCP.
"""
import json
from typing import Optional, Tuple
from .route_type import RouteStrategyMapper
from ..core.tools import SandboxBaseTool



# Travel mode mapping: API mode -> internal mode
TRAVEL_MODE_MAP = {
    "driving": "驾车",
    "walking": "步行",
    "bicycling": "骑行",
    "transit": "公交",
    "motorcycle": "摩托车",
    "truck": "货车",
}


def _parse_coordinate(coord_str: str) -> Tuple[str, str]:
    """
    Parse coordinate string and return (lat, lon).
    
    Args:
        coord_str: Coordinate string in format "latitude,longitude".
        
    Returns:
        Tuple of (latitude, longitude) strings.
    """
    parts = coord_str.strip().split(',')
    if len(parts) != 2:
        raise ValueError(f"坐标格式错误: {coord_str}")
    latitude, longitude = parts
    return latitude, longitude


def _get_route_type(traffic_aware: bool, preference: str) -> Tuple[int, str]:
    """
    Calculate route_type based on traffic_aware and preference.
    
    Args:
        traffic_aware: Whether to consider real-time traffic.
        preference: Route preference string.

    Returns:
        Tuple of (route_type, error_message).
    """
    traffic_aware = traffic_aware or False
    preference = preference or ""

    error_message, route_type = RouteStrategyMapper.get_route_type(
        traffic_aware=traffic_aware,
        route_modifiers=preference
    )

    return route_type, error_message


class MapComputeRoutesTool(SandboxBaseTool):
    """Route computation tool with sandbox caching support via MCP."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        input_schema = {
            "type": "object",
            "required": ["origin", "destination"],
            "properties": {
                "origin": {
                    "type": "string",
                    "description": '起点坐标。格式为 "latitude,longitude"。示例："39.9087,116.3974"'
                },
                "destination": {
                    "type": "string",
                    "description": '终点坐标。格式同 origin。'
                },
                "intermediates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": '途经点坐标列表。按顺序经过。坐标格式同 origin。'
                },
                "modes": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["driving", "walking", "bicycling", "transit", "motorcycle", "truck"]
                    },
                    "description": '出行方式列表。preference 和 traffic_aware 参数仅在 driving/motorcycle/truck 模式下生效。default: ["driving"]。'
                },
                "departure_time": {
                    "type": "string",
                    "description": '出发时间。用于基于未来时刻的路况预测。与 arrival_time 不可同时使用。格式：ISO8601 (如 "2025-07-11T09:00:00")。default: 当前时间。'
                },
                "arrival_time": {
                    "type": "string",
                    "description": '期望到达时间。用于反推建议出发时间。与 departure_time 不可同时使用。格式：ISO8601 (如 "2025-07-11T18:00:00")。'
                },
                "traffic_aware": {
                    "type": "boolean",
                    "description": '是否基于实时路况规划。false: 基于历史平均数据，响应最快，不考虑突发情况和路网变化，结果可能包含临时关闭的路段。同一个请求的结果可能会随着时间而变化，因为受到路网变化、交通历史数据更新等的影响。true: 基于实时路网数据和交通情况规划路线。仅在 driving/motorcycle/truck 模式下生效。default: false'
                },
                "preference": {
                    "type": "string",
                    "enum": ["prefer_highways", "avoid_highways", "avoid_tolls", "prefer_main_roads", "avoid_highways+avoid_tolls"],
                    "description": '路线偏好。仅当 modes 中包含 driving/motorcycle/truck 时生效。prefer_highways: 优先选择高速路; avoid_highways: 尽量不走高速路; avoid_tolls: 尽量避开收费路段; prefer_main_roads: 优先走主干道/大路，减少小路和复杂路口，适合新手司机、大车或夜间行驶场景; avoid_highways+avoid_tolls: 尽量不走高速路且避开收费路段。default: 不设置时返回耗时最短的路线'
                }
            }
        }
        
        super().__init__(
            name="map_compute_routes",
            description='路线规划工具。计算从起点到终点（可含途经点）的出行路线方案。适用场景：用户询问"从A到B怎么走"、"需要多长时间"、"打车/坐地铁/步行哪个快"、"如何避开高速/收费站"等路线规划类问题时调用。支持驾车、公交、步行、骑行、摩托车、货车多种出行方式，可根据实时路况、费用偏好或道路类型定制路线。返回信息包括每种出行方式对应的路线，每条路线包括：路线类型, 总距离, 预计耗时, 预估到达时间, 红绿灯数 (驾车), 费用 (公交), 途经道路, 换乘方案 (公交), 拥堵距离/时长 (驾车)。',
            input_schema=input_schema,
            cache_dir=cache_dir
        )
        
    def _validate_params(self, args: dict) -> str:
        """
        Validate parameters. Returns error message or empty string if valid.
        """
        valid_params = {
            'origin', 'destination', 'intermediates', 'modes', 
            'departure_time', 'arrival_time', 'traffic_aware', 'preference'
        }
        required_params = {'origin', 'destination'}
        
        missing_params = required_params - set(args.keys())
        if missing_params:
            missing_list = ', '.join(f"'{p}'" for p in sorted(missing_params))
            return f"缺少必须参数: {missing_list}"
        
        unknown_params = set(args.keys()) - valid_params
        if unknown_params:
            unknown_list = ', '.join(f"'{p}'" for p in sorted(unknown_params))
            valid_list = ', '.join(f"'{p}'" for p in sorted(valid_params))
            return f"包含未知参数: {unknown_list}。有效参数包括: {valid_list}"
        
        return ""

    def _validate_parameters(self, time: str, params: str) -> str:
        args = json.loads(params)
        validation_error = self._validate_params(args)
        if validation_error:
            return json.dumps({'error': validation_error}, ensure_ascii=False)

        origin = args['origin']
        destination = args['destination']
        intermediates = args.get('intermediates', [])
        modes = args.get('modes', ['driving'])
        departure_time = args.get('departure_time', '')
        arrival_time = args.get('arrival_time', '')
        traffic_aware = args.get('traffic_aware', False)
        preference = args.get('preference', '')

        if departure_time and arrival_time:
            return json.dumps({'error': 'departure_time 和 arrival_time 不能同时使用'}, ensure_ascii=False)

        # Parse origin coordinate
        try:
            start_lat, start_lon = _parse_coordinate(origin)
        except ValueError:
            return json.dumps({'error': f'origin 坐标格式错误: {origin}，应为 "latitude,longitude"'}, ensure_ascii=False)

        # Parse destination coordinate
        try:
            end_lat, end_lon = _parse_coordinate(destination)
        except ValueError:
            return json.dumps({'error': f'destination 坐标格式错误: {destination}，应为 "latitude,longitude"'}, ensure_ascii=False)

        # Parse intermediate waypoints
        via_points_str = ""
        if intermediates:
            via_list = []
            for i, coord in enumerate(intermediates):
                try:
                    lat, lon = _parse_coordinate(coord)
                    via_list.append(f"{lon},{lat}")
                except ValueError:
                    return json.dumps({'error': f'第 {i+1} 个途经点坐标格式错误: {coord}'}, ensure_ascii=False)
            via_points_str = ";".join(via_list)

        # Convert travel modes
        mode_list = []
        for mode in modes:
            if mode not in TRAVEL_MODE_MAP:
                return json.dumps({'error': f'不支持的出行方式: {mode}'}, ensure_ascii=False)
            mode_list.append(TRAVEL_MODE_MAP[mode])

        # Calculate route_type (only valid for driving/motorcycle/truck modes)
        route_type, route_error = _get_route_type(traffic_aware, preference)
        if route_type is None:
            return json.dumps({'error': route_error}, ensure_ascii=False)
        
        return ""

    def _real_execute(self, time: str, params: str) -> str:
        raise NotImplementedError("Real API execution is not implemented")
            
# Create tool instance (registration handled by __init__.py)
map_compute_routes_tool = MapComputeRoutesTool()