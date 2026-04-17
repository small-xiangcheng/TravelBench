"""
Train search tools for the travel benchmark framework.
Provides train/high-speed rail search with sandbox caching support via MCP.
"""

import json
from typing import Optional
from datetime import datetime, timedelta
from .get_adcode import get_adcode
from ..core.tools import SandboxBaseTool   

def _resolve_region_to_adcode(region: str) -> tuple:
    """
    Convert region name to adcode using get_adcode.

    Args:
        region: Region name string.
        
    Returns:
        Tuple of (adcode, error_message). On success: (adcode, ""). On failure: ("", error_message).
    """
    if not region:
        return "", "地区名称不能为空"

    result = get_adcode(region)
    resolved = result.get("resolved")
    matches = result.get("matches", [])
    note = result.get("note", "")

    if resolved:
        # 精确匹配或唯一后缀匹配
        return str(resolved["adcode"]), ""

    if note == "no match":
        return "", f"无法识别的地区名称: \"{region}\"，请使用标准行政区划名称（如\"北京市\"、\"上海市\"）"

    if note == "ambiguous matches" and matches:
        # 存在多个匹配，返回歧义错误
        candidates = [f"{m['full_name']}({m['adcode']})" for m in matches[:5]]
        return "", f"地区名称 \"{region}\" 存在歧义，匹配到多个结果: {', '.join(candidates)}。请提供更精确的地区名称。"

    return "", f"地区解析失败: {note}"

class SearchTrainsTool(SandboxBaseTool):
    """Train search tool with sandbox caching support via MCP."""
    
    def __init__(self, cache_dir: Optional[str] = None):        
        input_schema = {
            "type": "object",
            "required": ["origin", "destination", "date"],
            "properties": {
                "origin": {
                    "type": "string",
                    "description": '出发地。必须是中国境内的省、市、区/县等行政区名称。支持的形式：- 完整路径名：如 "北京市"、"北京市朝阳区"；- 仅末级名称：如 "朝阳区"、"浦东新区"。此时会在全国范围内尝试消歧。为免歧义，请优先提供完整路径名。'
                },
                "destination": {
                    "type": "string",
                    "description": '目的地。格式同 origin。'
                },
                "date": {
                    "type": "string",
                    "description": '查询的开始日期。格式: "YYYY-MM-DD"。示例："2025-07-15"。'
                },
                "days": {
                    "type": "number",
                    "description": '额外查询的天数。从 date 开始，共返回 days+1 天的车次。例如 date="2025-12-01", days=3 返回 12-01 至 12-04 共 4 天。default: 0, min: 0, max: 15'
                }
            }
        }
        super().__init__(
            name="travel_search_trains",
            description='中国境内火车/高铁时刻表搜索工具。适用于：查询城际交通方案、比较不同日期的车票价格。支持同时查询连续多天的车次信息（例如"查询下周去上海的高铁"）。返回车次列表，每条记录包含：车次号，是否直达，票价，出发/到达时间，出发/到达站点，全程时长。',
            input_schema=input_schema,
            cache_dir=cache_dir
        )

    def _validate_params(self, args: dict) -> str:
        """
        Validate parameters. Returns error message or empty string if valid.
        """
        valid_params = {'origin', 'destination', 'date', 'days'}
        required_params = {'origin', 'destination', 'date'}
        
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
        date = args['date']
        days = args.get('days', 0)

        # Parameter boundary validation
        if days < 0 or days > 15:
            return json.dumps({'error': f'days 参数超出范围，应在 0-15 之间，实际值: {days}'}, ensure_ascii=False)

        # Parse origin adcode
        origin_adcode, origin_error = _resolve_region_to_adcode(origin)
        if origin_error:
            return json.dumps({'error': f'出发地解析失败: {origin_error}'}, ensure_ascii=False)

        # Parse destination adcode
        dest_adcode, dest_error = _resolve_region_to_adcode(destination)
        if dest_error:
            return json.dumps({'error': f'目的地解析失败: {dest_error}'}, ensure_ascii=False)

        # Validate date format
        try:
            start_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return json.dumps({'error': f'日期格式错误: "{date}"，应为 "YYYY-MM-DD"'}, ensure_ascii=False)      
    
        # Validate date range: cannot exceed time + 15 days
        try:
            current_time = datetime.fromisoformat(time.replace(' ', 'T') if ' ' in time else time)
            max_allowed_date = current_time + timedelta(days=15)
            
            if start_date.date() > max_allowed_date.date():
                return json.dumps({
                    'error': f'查询日期超出范围：当前时间为 {current_time.strftime("%Y-%m-%d")}，最多只能查询未来15天内的车票（截止 {max_allowed_date.strftime("%Y-%m-%d")}），当前查询日期：{date}'
                }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({'error': f'日期验证失败: {str(e)}'}, ensure_ascii=False)  
        return ""
    
    def _real_execute(self, time: str, params: str) -> str:
        raise NotImplementedError("Real API execution is not implemented")


# Create tool instance (registration handled by __init__.py)
search_trains_tool = SearchTrainsTool()

