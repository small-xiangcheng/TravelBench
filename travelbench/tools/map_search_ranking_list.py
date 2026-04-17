"""
Ranking list search tools for the travel benchmark framework.
Provides ranking list search with sandbox caching support via MCP.
"""

import json
from typing import Optional
from ..core.tools import SandboxBaseTool   

class MapSearchRankingListTool(SandboxBaseTool):
    """Ranking list search tool with sandbox caching support via MCP."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        input_schema = {
            "type": "object",
            "required": ["query", "region"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": '榜单类别或关键词。例如："美食", "火锅", "游乐园", "酒店"。'
                },
                "region": {
                    "type": "string",
                    "description": '榜单所在的行政区域名称。中国行政区划名称，格式为 "{城市名}市" 或 "{城市名}市{区县名}区/县"。示例："北京市", "杭州市"。'
                },
                "max_lists": {
                    "type": "number",
                    "description": '返回的榜单数量限制。default: 5, min: 1, max: 10'
                },
                "max_items": {
                    "type": "number",
                    "description": '每个榜单返回的地点数量限制。default: 3, min: 1, max: 20'
                }
            }
        }
        super().__init__(
            name="map_search_ranking_list",
            description='榜单搜索工具。用于查找特定区域内的推荐榜单（如"美食榜"、"必吃榜"）。适用于：用户寻求高质量推荐、询问"哪里最火"、或者明确提到"榜单"时的场景。返回包含: 匹配的榜单列表，每个榜单包含榜单名称及上榜地点列表；每个地点包含名称、排名、place_id、地址、商圈、坐标、类别、人均消费、收藏数、营业时间、联系电话、推荐理由、特色标签。',
            input_schema=input_schema,
            cache_dir=cache_dir
        )

    def _validate_params(self, args: dict) -> str:
        """
        Validate parameters. Returns error message or empty string if valid.
        """
        valid_params = {'query', 'region', 'max_lists', 'max_items'}
        required_params = {'query', 'region'}
        
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
        region = args['region']
        max_lists = args.get('max_lists', 5)
        max_items = args.get('max_items', 3)

        # Parameter boundary validation
        if max_lists < 1 or max_lists > 10:
            return json.dumps({'error': f'max_lists 参数超出范围，应在 1-10 之间，实际值: {max_lists}'}, ensure_ascii=False)
        
        if max_items < 1 or max_items > 20:
            return json.dumps({'error': f'max_items 参数超出范围，应在 1-20 之间，实际值: {max_items}'}, ensure_ascii=False)
        return ""

    def _real_execute(self,time, params: str) -> str:
        raise NotImplementedError("Real API execution is not implemented")


# Create tool instance (registration handled by __init__.py)
map_search_ranking_list_tool = MapSearchRankingListTool()
