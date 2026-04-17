"""
Along route search tools for the travel benchmark framework.
Provides along route search with sandbox caching support via MCP.
"""

import json
from typing import Optional
from ..core.tools import SandboxBaseTool   

class MapSearchAlongRouteTool(SandboxBaseTool):
    """Along route search tool with sandbox caching support via MCP."""
    
    def __init__(self, cache_dir: Optional[str] = None):        
        input_schema = {
            "type": "object",
            "required": ["query", "origin", "destination"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": '搜索关键词，例如 "加油站"、"麦当劳"、"充电桩"。'
                },
                "origin": {
                    "type": "string",
                    "description": '起点坐标。格式为 "latitude,longitude" (例如 "39.90,116.40")。'
                },
                "destination": {
                    "type": "string",
                    "description": '终点坐标。格式为 "latitude,longitude" (例如 "31.23,121.47")。'
                },
                "transport_mode": {
                    "type": "string",
                    "enum": ["driving", "walking"],
                    "description": '交通方式，决定基础路线的规划逻辑。driving: 驾车; walking: 步行。default:\'driving\' '
                },
                "buffer_radius": {
                    "type": "number",
                    "description": '搜索缓冲区半径，即地点允许偏离路线的最大距离（单位：米）。default: 2000 (driving), 500 (walking)'
                },
                "limit": {
                    "type": "number",
                    "description": '返回结果数量上限。default: 10'
                }
            }
        }
        super().__init__(
            name="map_search_along_route",
            description='沿途搜索工具。查找从 A 到 B 顺路的地点（例如"从公司回家顺路加个油"）。基于起终点自动规划默认路线，并在路线两侧缓冲区内搜索。返回一组地点，每个地点包含：place_id, 名称，类别，地址，经纬度，标签，营业时间，营业状态，排行榜。',
            input_schema=input_schema,
            cache_dir=cache_dir
        )

    def _validate_params(self, args: dict) -> str:
        """
        Validate parameters. Returns error message or empty string if valid.
        """
        valid_params = {
            'query', 'origin', 'destination', 'transport_mode', 
            'buffer_radius', 'limit'
        }
        required_params = {'query', 'origin', 'destination'}
        
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

        query = args['query']
        origin = args['origin']
        destination = args['destination']
        transport_mode = args.get('transport_mode', 'driving')
        buffer_radius = args.get('buffer_radius')
        limit = args.get('limit', 10)

        # Parameter boundary validation
        if limit < 1:
            return json.dumps({'error': f'limit 参数不能小于 1，实际值: {limit}'}, ensure_ascii=False)

        # Parse origin coordinate
        try:
            origin_lat, origin_lon = origin.split(',')
            start_y = origin_lat.strip()
            start_x = origin_lon.strip()
        except:
            return json.dumps({'error': f'origin 参数格式错误，应为 "latitude,longitude"，实际为 "{origin}"'}, ensure_ascii=False)

        # Parse destination coordinate
        try:
            dest_lat, dest_lon = destination.split(',')
            end_y = dest_lat.strip()
            end_x = dest_lon.strip()
        except:
            return json.dumps({'error': f'destination 参数格式错误，应为 "latitude,longitude"，实际为 "{destination}"'}, ensure_ascii=False)    
        return ""

    def _real_execute(self, time, params: str) -> str:
        raise NotImplementedError("Real API execution is not implemented")

# Create tool instance (registration handled by __init__.py)
map_search_along_route_tool = MapSearchAlongRouteTool()
