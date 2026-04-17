"""
Place search tools for the travel benchmark framework.
Provides place search with sandbox caching support via MCP.
"""

import json
from typing import Optional, Tuple

from .get_adcode import get_adcode
from ..core.tools import SandboxBaseTool

def _resolve_region_to_adcode(region: str) -> Tuple[str, str]:
    """
    Convert region name to adcode using get_adcode.

    Args:
        region: Region name string.
        
    Returns:
        Tuple of (adcode, error_message). On success: (adcode, ""). On failure: ("", error_message).
    """
    if not region:
        return "", ""

    result = get_adcode(region)
    resolved = result.get("resolved")
    matches = result.get("matches", [])
    note = result.get("note", "")

    if resolved:
        # 精确匹配或唯一后缀匹配
        return str(resolved["adcode"]), ""

    if note == "no match":
        return "", f"无法识别的地区名称: \"{region}\"，请使用标准行政区划名称（如\"北京市\"、\"北京市朝阳区\"）"

    if note == "ambiguous matches" and matches:
        # 存在多个匹配，返回歧义错误
        candidates = [f"{m['full_name']}({m['adcode']})" for m in matches[:5]]
        return "", f"地区名称 \"{region}\" 存在歧义，匹配到多个结果: {', '.join(candidates)}。请提供更精确的地区名称。"

    return "", f"地区解析失败: {note}"


class MapSearchPlacesTool(SandboxBaseTool):
    """Place search tool with sandbox caching support via MCP."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        input_schema = {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": '搜索关键词。可以是关键词（"理发"）、类别（"加油站"）或具体地址。'
                },
                "center": {
                    "type": "string",
                    "description": '搜索中心点坐标，格式为 "latitude,longitude"。如果想搜索"附近的X"或"某地附近的X"，请提供此参数。示例: "39.9087,116.3974"。'
                },
                "radius": {
                    "type": "number",
                    "description": '搜索半径，单位：米 (meters)。仅在指定 center 时有效。default: 5000, min: 100, max: 20000'
                },
                "region": {
                    "type": "string",
                    "description": '限制搜索的行政区划，如"北京市"、"北京市朝阳区"。优先使用具体的区县级范围。不传则不限制。示例："北京市"、"北京市朝阳区"。'
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["weight", "distance", "rating:d", "price:a", "price:d"],
                    "description": '排序规则。"weight" - 综合排序; "distance" - 距离优先; "rating:d" - 高评分优先; "price:a" - 价格从低到高; "price:d" - 价格从高到低。default: \'weight\''
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
                    "description": '价格类型。当使用 price_min、price_max 或 sort_by 为 "price:a"/"price:d" 时，必须指定此参数。average_cost: 人均消费，适用于餐厅、娱乐场所等; price: 商品价格，适用于酒店、景点等。'
                },
                "limit": {
                    "type": "number",
                    "description": '返回结果的最大数量。default: 10'
                }
            }
        }
        super().__init__(
            name="map_search_places",
            description='地点搜索工具。根据关键词、类别或地址搜索地点，支持周边搜索和行政区限定。适用于：查找具体地点、关键词搜索（如"咖啡馆"）、周边搜索（如"附近的加油站"）、特定行政区内搜索。返回一组地点，每个地点包含：place_id, 名称，类别，地址，经纬度，评分，人均消费，营业时间，营业状态，品牌，特色，排行榜，精选用户评价。',
            input_schema=input_schema,
            cache_dir=cache_dir
        )

    def _validate_params(self, args: dict) -> str:
        """
        Validate parameters. Returns error message or empty string if valid.
        """
        # 定义有效参数列表
        valid_params = {
            'query', 'center', 'radius', 'region', 'sort_by', 
            'min_rating', 'price_min', 'price_max', 'price_category', 'limit'
        }
        required_params = {'query'}
        
        # 检查必须参数
        missing_params = required_params - set(args.keys())
        if missing_params:
            missing_list = ', '.join(f"'{p}'" for p in sorted(missing_params))
            return f"缺少必须参数: {missing_list}"
        
        # 检查未知参数
        unknown_params = set(args.keys()) - valid_params
        if unknown_params:
            unknown_list = ', '.join(f"'{p}'" for p in sorted(unknown_params))
            valid_list = ', '.join(f"'{p}'" for p in sorted(valid_params))
            return f"包含未知参数: {unknown_list}。有效参数包括: {valid_list}"
        
        # Check price parameter dependencies
        price_min = args.get('price_min')
        price_max = args.get('price_max')
        price_category = args.get('price_category')
        sort_by = args.get('sort_by', 'weight')
        
        # Check price filter parameters
        if (price_min is not None or price_max is not None) and not price_category:
            return "当使用 price_min 或 price_max 参数时，必须同时提供 price_category 参数"
        
        # Check price sort parameters
        if sort_by in ['price:a', 'price:d'] and not price_category:
            return "当 sort_by 设置为 price:a 或 price:d 时，必须同时提供 price_category 参数"
        
        return ""
    def _validate_parameters(self, time: str, params: str) -> str:
        args = json.loads(params)
        
        # 参数验证
        validation_error = self._validate_params(args)
        if validation_error:
            return json.dumps({'error': validation_error}, ensure_ascii=False)

        query = args['query']
        center = args.get('center', '')
        radius = args.get('radius', 5000)
        region = args.get('region', '')
        sort_by = args.get('sort_by', 'weight')
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

        # Build rating filter range
        rating_range = ""
        if min_rating is not None and min_rating > 0:
            rating_range = f"[{min_rating},10]"

        # Build price filter range - only when price_category is explicitly set
        general_price_range = ""
        hotel_price_range = ""
        if (price_min is not None or price_max is not None) and price_category:
            p_min = price_min if price_min is not None else 0
            p_max = price_max if price_max is not None else 999999
            price_range = f"[{p_min},{p_max}]"

            if price_category == "price":
                hotel_price_range = price_range
            else:
                general_price_range = price_range

        # Determine if this is nearby search or city-wide search
        is_around_search = bool(center)
        if is_around_search:
            # Nearby search: parse center parameter
            try:
                lat_str, lon_str = center.split(',')
                lat = float(lat_str.strip())
                lon = float(lon_str.strip())
            except:
                return json.dumps({'error': f'center 参数格式错误，应为 "latitude,longitude"，实际为 "{center}"'}, ensure_ascii=False)
        else:
            # City-wide search: use get_adcode to resolve region
            cur_adcode = ""
            if region:
                cur_adcode, error = _resolve_region_to_adcode(region)
                if error:
                    return json.dumps({'error': error}, ensure_ascii=False)
        return ""

    def _real_execute(self, time: str, params: str) -> str:
        raise NotImplementedError("Real API execution is not implemented")


# Create tool instance (registration handled by __init__.py)
map_search_places_tool = MapSearchPlacesTool()
