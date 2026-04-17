"""
Central places search tools for the travel benchmark framework.
Provides multi-point central search with sandbox caching support via MCP.
"""

import json
from typing import Optional,Tuple
from ..core.tools import SandboxBaseTool   

def _parse_coordinate(coord_str: str) -> Tuple[float, float]:
    """
    Parse coordinate string and return (lat, lon).
    
    Args:
        coord_str: Coordinate string in format "latitude,longitude".
        
    Returns:
        Tuple of (latitude, longitude).
    """
    parts = coord_str.strip().split(',')
    if len(parts) != 2:
        raise ValueError(f"坐标格式错误: {coord_str}")
    lat = float(parts[0].strip())
    lon = float(parts[1].strip())
    return lat, lon


class MapSearchCentralPlacesTool(SandboxBaseTool):
    """Central places search tool with sandbox caching support via MCP."""
    
    def __init__(self, cache_dir: Optional[str] = None):    
        input_schema = {
            "type": "object",
            "required": ["query", "origins"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": '搜索关键词。'
                },
                "origins": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": '出发点/参考点列表。必须包含至少两个坐标点。格式为 "latitude,longitude" 的字符串数组。例如: ["39.90,116.40", "39.95,116.35"]'
                },
                "radius": {
                    "type": "number",
                    "description": '每个出发点的搜索半径（米），应该在100-20000之间。default: 10000 '
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["balanced", "min_max_distance", "total_distance"],
                    "description": '排序策略。balanced: 综合最优，兼顾所有出发点的距离和地点质量; min_max_distance: 最小化最大距离（照顾最远的人，避免有人跑太远）; total_distance: 最小化总距离（所有人的路程之和最短）。default: balanced'
                },
                "min_rating": {
                    "type": "number",
                    "description": '最低评分。min: 0, max: 10'
                },
                "price_min": {
                    "type": "number",
                    "description": '价格/人均消费区间 - 最低值，单位：元 (CNY)。default: 0'
                },
                "price_max": {
                    "type": "number",
                    "description": '价格/人均消费区间 - 最高值，单位：元 (CNY)。No limit if omitted.'
                },
                "price_category": {
                    "type": "string",
                    "enum": ["average_cost", "price"],
                    "description": '价格类型。当使用 price_min 或 price_max 时，必须指定此参数。average_cost: 人均消费，适用于餐厅、娱乐场所等; price: 商品价格，适用于酒店、景点等。'
                },
                "limit": {
                    "type": "number",
                    "description": '返回结果的最大数量。default: 10'
                }
            }
        }
        super().__init__(
            name="map_search_central_places",
            description='聚会选址/多点中心搜索。适用于：寻找离多个出发点（如多人的位置）都比较方便的"中间地点"。典型场景："找一个离我和小红、小明都近的火锅店"、"在A地和B地中间找个加油站"。该工具会自动计算中心度分数并推荐最优位置。返回一组地点，每个地点包含：place_id, 名称，类别，地址，经纬度，评分，人均消费，营业时间，营业状态，与各出发点的平均距离、最远距离及总距离。',
            input_schema=input_schema,
            cache_dir=cache_dir
        )

    def _validate_params(self, args: dict) -> str:
        """
        Validate parameters. Returns error message or empty string if valid.
        """
        valid_params = {
            'query', 'origins', 'radius', 'sort_by', 'min_rating', 
            'price_min', 'price_max', 'price_category', 'limit'
        }
        required_params = {'query', 'origins'}
        
        missing_params = required_params - set(args.keys())
        if missing_params:
            missing_list = ', '.join(f"'{p}'" for p in sorted(missing_params))
            return f"缺少必须参数: {missing_list}"
        
        unknown_params = set(args.keys()) - valid_params
        if unknown_params:
            unknown_list = ', '.join(f"'{p}'" for p in sorted(unknown_params))
            valid_list = ', '.join(f"'{p}'" for p in sorted(valid_params))
            return f"包含未知参数: {unknown_list}。有效参数包括: {valid_list}"
        
        # Check price parameter dependencies
        price_min = args.get('price_min')
        price_max = args.get('price_max')
        price_category = args.get('price_category')
        
        if (price_min is not None or price_max is not None) and not price_category:
            return "当使用 price_min 或 price_max 参数时，必须同时提供 price_category 参数"
        
        return ""
    def _validate_parameters(self, time: str, params: str) -> str:
        args = json.loads(params)
        validation_error = self._validate_params(args)
        if validation_error:
            return json.dumps({'error': validation_error}, ensure_ascii=False)

        query = args['query']
        origins_str = args['origins']
        radius = args.get('radius', 10000)
        sort_by = args.get('sort_by', 'balanced')
        min_rating = args.get('min_rating')
        price_min = args.get('price_min')
        price_max = args.get('price_max')
        price_category = args.get('price_category')
        limit = args.get('limit', 10)

        # Parameter boundary validation
        if radius < 100 or radius > 20000:
            return json.dumps({'error': f'radius 参数超出范围，应在 100-20000 之间，实际值: {radius}'}, ensure_ascii=False)
        if min_rating is not None and (min_rating < 0 or min_rating > 10):
            return json.dumps({'error': f'min_rating 参数超出范围，应在 0-10 之间，实际值: {min_rating}'}, ensure_ascii=False)
        
        if price_min is not None and price_min < 0:
            return json.dumps({'error': f'price_min 参数不能为负数，实际值: {price_min}'}, ensure_ascii=False)
        
        if price_max is not None and price_max < 0:
            return json.dumps({'error': f'price_max 参数不能为负数，实际值: {price_max}'}, ensure_ascii=False)
        
        if price_min is not None and price_max is not None and price_min > price_max:
            return json.dumps({'error': f'price_min ({price_min}) 不能大于 price_max ({price_max})'}, ensure_ascii=False)
        
        if limit < 1:
            return json.dumps({'error': f'limit 参数不能小于 1，实际值: {limit}'}, ensure_ascii=False)

        # Validate origin points count
        if not origins_str or len(origins_str) < 2:
            return json.dumps({'error': '至少需要提供 2 个出发点坐标'}, ensure_ascii=False)

        # Parse origin coordinates
        origins = []
        pois_for_search = []
        for i, coord_str in enumerate(origins_str):
            try:
                lat, lon = _parse_coordinate(coord_str)
                origins.append((lat, lon))
                pois_for_search.append({'x': lon, 'y': lat})
            except ValueError:
                return json.dumps({'error': f'第 {i+1} 个出发点坐标格式错误: {coord_str}，应为 "latitude,longitude"'}, ensure_ascii=False)
        return ""
        
    def _real_execute(self,time, params: str) -> str:
        raise NotImplementedError("Real API execution is not implemented")


# Create tool instance (registration handled by __init__.py)
map_search_central_places_tool = MapSearchCentralPlacesTool()
