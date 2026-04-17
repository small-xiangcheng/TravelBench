"""
Web search tools for the travel benchmark framework.
Provides web search with sandbox caching support via MCP.
"""

import json
from typing import Optional
from ..core.tools import SandboxBaseTool   

class WebSearchTool(SandboxBaseTool):
    """Web search tool with sandbox caching support via MCP."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        input_schema = {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": '搜索查询词。'
                }
            }
        }
        
        super().__init__(
            name="web_search",
            description="开放域网页搜索工具，用于检索互联网上的公开信息。适用于：回答通用的、开放式、时效性高的问题，如百科知识、实时新闻、历史事件、政策法规等。示例查询：北京特色美食、上海好玩的游乐园、杭州周边城市有哪些、北京跨年哪里有活动。",
            input_schema=input_schema,
            cache_dir=cache_dir
        )

    def _validate_params(self, params_dict: dict) -> str:
        """Validate parameters. Returns error message or empty string if valid."""
        required_params = {'query'}
        valid_params = {'query'}
        
        provided_params = set(params_dict.keys())
        
        missing_params = required_params - provided_params
        if missing_params:
            return json.dumps({
                'error': f'缺少必须参数: {", ".join(missing_params)}'
            }, ensure_ascii=False)
        
        unknown_params = provided_params - valid_params
        if unknown_params:
            return json.dumps({
                'error': f'未知参数: {", ".join(unknown_params)}。有效参数: {", ".join(sorted(valid_params))}'
            }, ensure_ascii=False)
        
        return ""
        
    def _validate_parameters(self, time: str, params: str) -> str:
        try:
            args = json.loads(params)
        except Exception as e:
            return json.dumps({'error': f'参数解析失败: {str(e)}'}, ensure_ascii=False)
        validation_error = self._validate_params(args)
        if validation_error:
            return validation_error
        return ""

    def _real_execute(self, time, params: str) -> str:
        raise NotImplementedError("Real API execution is not implemented")


# Create tool instance (registration handled by __init__.py)
web_search_tool = WebSearchTool()
