"""
Weather forecast tools for the travel benchmark framework.
Provides multi-day weather forecast with sandbox caching support via MCP.
"""

import json
from datetime import datetime, timedelta
from typing import Optional
from ..core.tools import SandboxBaseTool   

class WeatherForecastTool(SandboxBaseTool):
    """Weather forecast tool with sandbox caching support via MCP."""
    
    def __init__(self, cache_dir: Optional[str] = None):        
        input_schema = {
            "type": "object",
            "required": ["location","date"],
            "properties": {
                "location": {
                    "type": "string",
                    "description": '查询地点。支持行政区划名称或地标。'
                },
                "days": {
                    "type": "number",
                    "description": '表示查询从date开始未来多少天的预报（结果自动包含date）。示例：days=1 返回date, date+1共 2 天，days=3 返回date+未来3天共 4 天。默认为 1，最小值 1，最大值 15。'
                },
                "date": {
                    "type": "string",
                    "description": '按特定日期查询。格式：YYYY-MM-DD (例如 "2025-11-05")。限制：必须在 [当前日期, 当前日期 + 15天] 范围内。'
                }
            }
        }
        super().__init__(
            name="weather_forecast_days",
            description='获取未来几天的天气预报。支持查询特定日期的天气（如"下周五"）或从当前时间开始未来一段时间的趋势（如"未来三天"）。注意：受限于数据源，仅支持查询从今天起未来 15 天内的数据，例如今天是12月1日，最多返回12月1日到12月16日内的数据。返回一个数组，每项包含：日期 (YYYY-MM-DD）、day_label（今天/明天/周几）、天气状况、最低温度、最高温度、风力风向、空气质量。',
            input_schema=input_schema,
            cache_dir=cache_dir
        )

    def _validate_params(self, args: dict) -> str:
        """
        Validate parameters. Returns error message or empty string if valid.
        """
        valid_params = {'location', 'days', 'date'}
        required_params = {'location','date'}
        
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
        location = args.get('location')
        days = args.get('days')  
        date = args.get('date')

        if days is not None:
            # Parameter boundary validation
            if days < 1 or days > 15:
                return json.dumps({'error': f'days 参数超出范围，应在 1-15 之间，实际值: {days}'}, ensure_ascii=False)

        if date is not None:
            # Validate date format
            try:
                query_date = datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                return json.dumps({'error': f'日期格式错误: "{date}"，应为 "YYYY-MM-DD"'}, ensure_ascii=False)  
            # Validate date range
            try:
                current_time = datetime.fromisoformat(time.replace(' ', 'T') if ' ' in time else time)
                max_allowed_date = current_time + timedelta(days=15)
                
                if query_date.date() > max_allowed_date.date():
                    return json.dumps({
                        'error': f'查询日期超出范围：当前时间为 {current_time.strftime("%Y-%m-%d")}，最多只能查询未来15天内的天气（截止 {max_allowed_date.strftime("%Y-%m-%d")}），当前查询日期：{date}'
                    }, ensure_ascii=False)
            except Exception as e:
                return json.dumps({'error': f'日期验证失败: {str(e)}'}, ensure_ascii=False)
        return ""
            
    def _real_execute(self, time, params: str) -> str:
        raise NotImplementedError("Real API execution is not implemented")


# Create tool instance (registration handled by __init__.py)
weather_forecast_tool = WeatherForecastTool()
