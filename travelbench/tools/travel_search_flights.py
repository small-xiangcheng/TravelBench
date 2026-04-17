"""
Flight search tools for the travel benchmark framework.
Provides flight-related tools with sandbox caching support via MCP.
"""

import json
from typing import Optional
from datetime import datetime, timedelta
from ..core.tools import SandboxBaseTool   

class SearchFlightsTool(SandboxBaseTool):
    """Flight search tool with sandbox caching support via MCP."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        # Add parameters matching tools/tools/travel_search_flights.py format
        input_schema = {
            "type": "object",
            "required": ["origin", "destination", "date"],
            "properties": {
                "origin": {
                    "type": "string",
                    "description": '出发地。必须是中国境内的省、市、区/县等行政区名称。支持的形式：完整路径名：如 "北京市"、"北京市朝阳区"；尽量避免使用如 "朝阳区" 等容易产生歧义的名称'
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
                    "description": '额外查询的天数。从 date 开始，共返回 days+1 天的航班。例如 date="2025-12-01", days=3 返回 12-01 至 12-04 共 4 天。default: 0, min: 0, max: 15'
                }
            }
        }
        super().__init__(
            name="travel_search_flights",
            description='中国境内航班搜索工具。适用于：查询从出发城市到到达城市的航班信息，可能包含中转和直飞航班。支持同时查询连续多天的航班，方便用户进行比价（例如"查询下周飞上海哪天便宜"）。返回一组航班，每个航班包含：航班号，航空公司，起降机场，起降时间，飞行时长，机型，价格范围。',
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

        # Validate city names are not empty
        if not origin or not destination:
            return "参数错误：出发地和目的地不能为空"

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
                    'error': f'查询日期超出范围：当前时间为 {current_time.strftime("%Y-%m-%d")}，最多只能查询未来15天内的航班（截止 {max_allowed_date.strftime("%Y-%m-%d")}），当前查询日期：{date}'
                }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({'error': f'日期验证失败: {str(e)}'}, ensure_ascii=False)
        
        return ""
    
    def _real_execute(self, time: str, params: str) -> str:
        raise NotImplementedError("Real API execution is not implemented")

# Create tool instance (registration handled by __init__.py)
search_flights_tool = SearchFlightsTool()