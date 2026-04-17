"""
Current weather conditions tools for the travel benchmark framework.
Provides real-time weather information with sandbox caching support via MCP.
"""

import json
from typing import Optional
from ..core.tools import SandboxBaseTool 



class WeatherCurrentTool(SandboxBaseTool):
    """Current weather conditions tool with sandbox caching support via MCP."""
    
    def __init__(self, cache_dir: Optional[str] = None):    
        input_schema = {
            "type": "object",
            "required": ["location"],
            "properties": {
                "location": {
                    "type": "string",
                    "description": '查询地点。支持行政区划名称（如"上海市"、"北京市朝阳区"）'
                }
            }
        }
        super().__init__(
            name="weather_current_conditions",
            description='获取指定城市的实时天气状况。不适用于未来天气预报查询。包含实时温度、体感温度、天气现象（如晴/雨）、风向风力及空气质量指数（AQI）。返回信息：天气状况、最低温度、最高温度、风力风向、空气质量。',
            input_schema=input_schema,
            cache_dir=cache_dir
        )

    def _validate_params(self, args: dict) -> str:
        """
        Validate parameters. Returns error message or empty string if valid.
        """
        valid_params = {'location'}
        required_params = {'location'}
        
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
        return ""
        
    def _real_execute(self, time, params: str) -> str:
        raise NotImplementedError("Real API execution is not implemented")

# Create tool instance (registration handled by __init__.py)
weather_current_tool = WeatherCurrentTool()
