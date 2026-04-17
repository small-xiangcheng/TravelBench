"""
Prompts for LLM-based tool simulator.
"""

# System prompt template for tool simulation
TOOL_SIMULATION_SYSTEM_PROMPT = """你是一个工具模拟器，需要模拟 {tool_name} 工具的真实返回结果。

工具定义：
名称：{tool_name}
描述：{tool_description}
参数定义：{tool_parameters}

任务要求：
1. 根据提供的真实示例，理解工具的输出格式和内容特点
2. 基于输入参数生成合理的模拟结果
3. 确保输出格式与示例一致
4. 生成的内容要符合工具的业务逻辑和真实场景
5. 直接返回模拟结果，不要添加任何额外说明、解释或 markdown 格式
6. 不要返回 JSON 格式的包装，直接返回工具本身应该返回的内容
"""

# Template for examples section (when examples are available)
EXAMPLES_SECTION_TEMPLATE = """
以下是 {num_examples} 个真实调用示例供参考：
"""

# Template for single example
SINGLE_EXAMPLE_TEMPLATE = """
示例 {index}：
输入参数：{params}
输出结果：{result}
"""

# Template for no examples case
NO_EXAMPLES_TEMPLATE = """
注意：没有找到 {tool_name} 的历史示例，请根据工具定义和参数生成合理的模拟结果。
"""

# User prompt template for tool simulation
TOOL_SIMULATION_USER_PROMPT = """请为以下参数生成 {tool_name} 工具的模拟返回结果：

参数：{params_json}

要求：
1. 注意参考真实调用示例，部分信息可能直接来源于给你的示例，相似的调用参数要生成相似的模拟结果
2. 内容要符合工具的业务逻辑和真实场景
3. 如果是列表类型的结果，生成若干个合理的条目
4. 数值要在合理范围内
5. 时间、日期等信息要符合参数中的约束
6. 直接返回结果内容，不要添加任何解释或格式包装
"""

TOOLS_SCHEMAS = [{'type': 'function', 'function': {'name': 'travel_search_flights', 'description': '中国境内航班搜索工具。适用于：查询从出发城市到到达城市的航班信息，可能包含中转和直飞航班。支持同时查询连续多天的航班，方便用户进行比价（例如"查询下周飞上海哪天便宜"）。返回一组航班，每个航班包含：航班号，航空公司，起降机场，起降时间，飞行时长，机型，价格范围。', 'parameters': {'type': 'object', 'required': ['origin', 'destination', 'date'], 'properties': {'origin': {'type': 'string', 'description': '出发地。必须是中国境内的省、市、区/县等行政区名称。支持的形式：完整路径名：如 "北京市"、"北京市朝阳区"；尽量避免使用如 "朝阳区" 等容易产生歧义的名称'}, 'destination': {'type': 'string', 'description': '目的地。格式同 origin。'}, 'date': {'type': 'string', 'description': '查询的开始日期。格式: "YYYY-MM-DD"。示例："2025-07-15"。'}, 'days': {'type': 'number', 'description': '额外查询的天数。从 date 开始，共返回 days+1 天的航班。例如 date="2025-12-01", days=3 返回 12-01 至 12-04 共 4 天。default: 0, min: 0, max: 15'}}}}}, {'type': 'function', 'function': {'name': 'travel_search_trains', 'description': '中国境内火车/高铁时刻表搜索工具。适用于：查询城际交通方案、比较不同日期的车票余量或价格。支持同时查询连续多天的车次信息（例如"查询下周去上海的高铁"）。返回车次列表，每条记录包含：车次号，是否直达，票价，出发/到达时间，出发/到达站点，全程时长。', 'parameters': {'type': 'object', 'required': ['origin', 'destination', 'date'], 'properties': {'origin': {'type': 'string', 'description': '出发地。必须是中国境内的省、市、区/县等行政区名称。支持的形式：- 完整路径名：如 "北京市"、"北京市朝阳区"；- 仅末级名称：如 "朝阳区"、"浦东新区"。此时会在全国范围内尝试消歧。为免歧义，请优先提供完整路径名。'}, 'destination': {'type': 'string', 'description': '目的地。格式同 origin。'}, 'date': {'type': 'string', 'description': '查询的开始日期。格式: "YYYY-MM-DD"。示例："2025-07-15"。'}, 'days': {'type': 'number', 'description': '额外查询的天数。从 date 开始，共返回 days+1 天的车次。例如 date="2025-12-01", days=3 返回 12-01 至 12-04 共 4 天。default: 0, min: 0, max: 15'}}}}}, {'type': 'function', 'function': {'name': 'weather_current_conditions', 'description': '获取指定城市的实时天气状况。不适用于未来天气预报查询。包含实时温度、体感温度、天气现象（如晴/雨）、风向风力及空气质量指数（AQI）。返回信息：天气状况、最低温度、最高温度、风力风向、空气质量。', 'parameters': {'type': 'object', 'required': ['location'], 'properties': {'location': {'type': 'string', 'description': '查询地点。支持行政区划名称（如"上海市"、"北京市朝阳区"）'}}}}}, {'type': 'function', 'function': {'name': 'weather_forecast_days', 'description': '获取未来几天的天气预报。支持查询特定日期的天气（如"下周五"）或从当前时间开始未来一段时间的趋势（如"未来三天"）。注意：受限于数据源，仅支持查询从今天起未来 15 天内的数据，例如今天是12月1日，最多返回12月1日到12月16日内的数据。返回一个数组，每项包含：日期 (YYYY-MM-DD）、day_label（今天/明天/周几）、天气状况、最低温度、最高温度、风力风向、空气质量。', 'parameters': {'type': 'object', 'required': ['location', 'date'], 'properties': {'location': {'type': 'string', 'description': '查询地点。支持行政区划名称或地标。'}, 'days': {'type': 'number', 'description': '表示查询从date开始未来多少天的预报（结果自动包含date）。示例：days=1 返回date, date+1共 2 天，days=3 返回date+未来3天共 4 天。默认为 1，最小值 1，最大值 15。'}, 'date': {'type': 'string', 'description': '按特定日期查询。格式：YYYY-MM-DD (例如 "2025-11-05")。限制：必须在 [当前日期, 当前日期 + 15天] 范围内。'}}}}}, {'type': 'function', 'function': {'name': 'map_compute_routes', 'description': '路线规划工具。计算从起点到终点（可含途经点）的出行路线方案。适用场景：用户询问"从A到B怎么走"、"需要多长时间"、"打车/坐地铁/步行哪个快"、"如何避开高速/收费站"等路线规划类问题时调用。支持驾车、公交、步行、骑行、摩托车、货车多种出行方式，可根据实时路况、费用偏好或道路类型定制路线。返回信息包括每种出行方式对应的路线，每条路线包括：路线类型, 总距离, 预计耗时, 预估到达时间, 红绿灯数 (驾车), 费用 (公交), 途经道路, 换乘方案 (公交), 拥堵距离/时长 (驾车)。', 'parameters': {'type': 'object', 'required': ['origin', 'destination'], 'properties': {'origin': {'type': 'string', 'description': '起点坐标。格式为 "latitude,longitude"。示例："39.9087,116.3974"'}, 'destination': {'type': 'string', 'description': '终点坐标。格式同 origin。'}, 'intermediates': {'type': 'array', 'items': {'type': 'string'}, 'description': '途经点坐标列表。按顺序经过。坐标格式同 origin。'}, 'modes': {'type': 'array', 'items': {'type': 'string', 'enum': ['driving', 'walking', 'bicycling', 'transit', 'motorcycle', 'truck']}, 'description': '出行方式列表。preference 和 traffic_aware 参数仅在 driving/motorcycle/truck 模式下生效。default: ["driving"]。'}, 'departure_time': {'type': 'string', 'description': '出发时间。用于基于未来时刻的路况预测。与 arrival_time 不可同时使用。格式：ISO8601 (如 "2025-07-11T09:00:00")。default: 当前时间。'}, 'arrival_time': {'type': 'string', 'description': '期望到达时间。用于反推建议出发时间。与 departure_time 不可同时使用。格式：ISO8601 (如 "2025-07-11T18:00:00")。'}, 'traffic_aware': {'type': 'boolean', 'description': '是否基于实时路况规划。false: 基于历史平均数据，响应最快，不考虑突发情况和路网变化，结果可能包含临时关闭的路段。同一个请求的结果可能会随着时间而变化，因为受到路网变化、交通历史数据更新等的影响。true: 基于实时路网数据和交通情况规划路线。仅在 driving/motorcycle/truck 模式下生效。default: false'}, 'preference': {'type': 'string', 'enum': ['prefer_highways', 'avoid_highways', 'avoid_tolls', 'prefer_main_roads', 'avoid_highways+avoid_tolls'], 'description': '路线偏好。仅当 modes 中包含 driving/motorcycle/truck 时生效。prefer_highways: 优先选择高速路; avoid_highways: 尽量不走高速路; avoid_tolls: 尽量避开收费路段; prefer_main_roads: 优先走主干道/大路，减少小路和复杂路口，适合新手司机、大车或夜间行驶场景; avoid_highways+avoid_tolls: 尽量不走高速路且避开收费路段。default: 不设置时返回耗时最短的路线'}}}}}, {'type': 'function', 'function': {'name': 'map_search_places', 'description': '地点搜索工具。根据关键词、类别或地址搜索地点，支持周边搜索和行政区限定。适用于：查找具体地点、关键词搜索（如"咖啡馆"）、周边搜索（如"附近的加油站"）、特定行政区内搜索。返回一组地点，每个地点包含：place_id, 名称，类别，地址，经纬度，评分，人均消费，营业时间，营业状态，品牌，特色，排行榜，精选用户评价。', 'parameters': {'type': 'object', 'required': ['query'], 'properties': {'query': {'type': 'string', 'description': '搜索关键词。可以是关键词（"理发"）、类别（"加油站"）或具体地址。'}, 'center': {'type': 'string', 'description': '搜索中心点坐标，格式为 "latitude,longitude"。如果想搜索"附近的X"或"某地附近的X"，请提供此参数。示例: "39.9087,116.3974"。'}, 'radius': {'type': 'number', 'description': '搜索半径，单位：米 (meters)。仅在指定 center 时有效。default: 5000, min: 100, max: 20000'}, 'region': {'type': 'string', 'description': '限制搜索的行政区划，如"北京市"、"北京市朝阳区"。优先使用具体的区县级范围。不传则不限制。示例："北京市"、"北京市朝阳区"。'}, 'sort_by': {'type': 'string', 'enum': ['weight', 'distance', 'rating:d', 'price:a', 'price:d'], 'description': '排序规则。"weight" - 综合排序; "distance" - 距离优先; "rating:d" - 高评分优先; "price:a" - 价格从低到高; "price:d" - 价格从高到低。default: \'weight\''}, 'min_rating': {'type': 'number', 'description': '最低评分。min: 0, max: 10'}, 'price_min': {'type': 'number', 'description': '价格/人均消费区间 - 最低值，单位：元 (CNY)。default: 0'}, 'price_max': {'type': 'number', 'description': '价格/人均消费区间 - 最高值，单位：元 (CNY)。No limit if omitted.'}, 'price_category': {'type': 'string', 'enum': ['average_cost', 'price'], 'description': '价格类型。当使用 price_min、price_max 或 sort_by 为 "price:a"/"price:d" 时，必须指定此参数。average_cost: 人均消费，适用于餐厅、娱乐场所等; price: 商品价格，适用于酒店、景点等。'}, 'limit': {'type': 'number', 'description': '返回结果的最大数量。default: 10'}}}}}, {'type': 'function', 'function': {'name': 'map_search_along_route', 'description': '沿途搜索工具。查找从 A 到 B 顺路的地点（例如"从公司回家顺路加个油"）。基于起终点自动规划默认路线，并在路线两侧缓冲区内搜索。返回一组地点，每个地点包含：place_id, 名称，类别，地址，经纬度，标签，营业时间，营业状态，排行榜。', 'parameters': {'type': 'object', 'required': ['query', 'origin', 'destination'], 'properties': {'query': {'type': 'string', 'description': '搜索关键词，例如 "加油站"、"麦当劳"、"充电桩"。'}, 'origin': {'type': 'string', 'description': '起点坐标。格式为 "latitude,longitude" (例如 "39.90,116.40")。'}, 'destination': {'type': 'string', 'description': '终点坐标。格式为 "latitude,longitude" (例如 "31.23,121.47")。'}, 'transport_mode': {'type': 'string', 'enum': ['driving', 'walking'], 'description': "交通方式，决定基础路线的规划逻辑。driving: 驾车; walking: 步行。default:'driving' "}, 'buffer_radius': {'type': 'number', 'description': '搜索缓冲区半径，即地点允许偏离路线的最大距离（单位：米）。default: 2000 (driving), 500 (walking)'}, 'limit': {'type': 'number', 'description': '返回结果数量上限。default: 10'}}}}}, {'type': 'function', 'function': {'name': 'map_search_central_places', 'description': '聚会选址/多点中心搜索。适用于：寻找离多个出发点（如多人的位置）都比较方便的"中间地点"。典型场景："找一个离我和小红、小明都近的火锅店"、"在A地和B地中间找个加油站"。该工具会自动计算中心度分数并推荐最优位置。返回一组地点，每个地点包含：place_id, 名称，类别，地址，经纬度，评分，人均消费，营业时间，营业状态，与各出发点的平均距离、最远距离及总距离。', 'parameters': {'type': 'object', 'required': ['query', 'origins'], 'properties': {'query': {'type': 'string', 'description': '搜索关键词。'}, 'origins': {'type': 'array', 'items': {'type': 'string'}, 'description': '出发点/参考点列表。必须包含至少两个坐标点。格式为 "latitude,longitude" 的字符串数组。例如: ["39.90,116.40", "39.95,116.35"]'}, 'radius': {'type': 'number', 'description': '每个出发点的搜索半径（米），应该在100-20000之间。default: 10000 '}, 'sort_by': {'type': 'string', 'enum': ['balanced', 'min_max_distance', 'total_distance'], 'description': '排序策略。balanced: 综合最优，兼顾所有出发点的距离和地点质量; min_max_distance: 最小化最大距离（照顾最远的人，避免有人跑太远）; total_distance: 最小化总距离（所有人的路程之和最短）。default: balanced'}, 'min_rating': {'type': 'number', 'description': '最低评分。min: 0, max: 10'}, 'price_min': {'type': 'number', 'description': '价格/人均消费区间 - 最低值，单位：元 (CNY)。default: 0'}, 'price_max': {'type': 'number', 'description': '价格/人均消费区间 - 最高值，单位：元 (CNY)。No limit if omitted.'}, 'price_category': {'type': 'string', 'enum': ['average_cost', 'price'], 'description': '价格类型。当使用 price_min 或 price_max 时，必须指定此参数。average_cost: 人均消费，适用于餐厅、娱乐场所等; price: 商品价格，适用于酒店、景点等。'}, 'limit': {'type': 'number', 'description': '返回结果的最大数量。default: 10'}}}}}, {'type': 'function', 'function': {'name': 'map_search_ranking_list', 'description': '榜单搜索工具。用于查找特定区域内的推荐榜单（如"美食榜"、"必吃榜"）。适用于：用户寻求高质量推荐、询问"哪里最火"、或者明确提到"榜单"时的场景。返回包含: 匹配的榜单列表，每个榜单包含榜单名称及上榜地点列表；每个地点包含名称、排名、place_id、地址、商圈、坐标、类别、人均消费、收藏数、营业时间、联系电话、推荐理由、特色标签。', 'parameters': {'type': 'object', 'required': ['query', 'region'], 'properties': {'query': {'type': 'string', 'description': '榜单类别或关键词。例如："美食", "火锅", "游乐园", "酒店"。'}, 'region': {'type': 'string', 'description': '榜单所在的行政区域名称。中国行政区划名称，格式为 "{城市名}市" 或 "{城市名}市{区县名}区/县"。示例："北京市", "杭州市"。'}, 'max_lists': {'type': 'number', 'description': '返回的榜单数量限制。default: 5, min: 1, max: 10'}, 'max_items': {'type': 'number', 'description': '每个榜单返回的地点数量限制。default: 3, min: 1, max: 20'}}}}}, {'type': 'function', 'function': {'name': 'web_search', 'description': '开放域网页搜索工具，用于检索互联网上的公开信息。适用于：回答通用的、开放式、时效性高的问题，如百科知识、实时新闻、历史事件、政策法规等。示例查询：北京特色美食、上海好玩的游乐园、杭州周边城市有哪些、北京跨年哪里有活动。', 'parameters': {'type': 'object', 'required': ['query'], 'properties': {'query': {'type': 'string', 'description': '搜索查询词。'}}}}}]
