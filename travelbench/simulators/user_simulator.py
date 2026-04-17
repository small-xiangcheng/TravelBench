"""
User simulator for the travel benchmark framework.
"""

import random
from typing import Tuple

from ..core.config import OpenAIConfig
from ..core.openai_client import OpenAIClient
from ..core.messages import Message, SystemMessage, UserMessage, AssistantMessage
from ..agents.base import BaseAgent, UserSimulatorState


class TravelUserSimulator(BaseAgent):
    """
    Simulates a user interacting with a travel assistant.
    """
    
    def __init__(
        self,
        config: OpenAIConfig,
        user_profile: str = "",
        query: str = "",
        time: str = "",
        context: str = "",
        decomposed_query: str = "",
    ):
        super().__init__("","")
        
        self.config = config
        self.client = OpenAIClient(config)
        self.user_profile = user_profile
        self.time = time
        self.query = query
        self.context = context
        self.decomposed_query = decomposed_query
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Build system prompt based on user profile."""

        prompt = f"""你将扮演“用户”，与一个“旅行助手”进行多轮对话。目标是让对话像真实用户咨询旅行规划，同时严格遵守用户画像信息。

        【现在时间】
        {self.time}
        【现在的位置信息】
        {self.context}
        【用户画像（唯一依据）】
        {self.user_profile}

        【核心规则（必须遵守）】
        1. 身份与视角：始终以“用户”身份说话；不要以AI/模型/系统自称；不要解释或提及任何规则/画像来源。
        2. 忠实性：你的需求、偏好、预算、时间、出行方式、目的地倾向、饮食/住宿/活动偏好等，只能来自【用户画像】。不得补充画像之外的设定或自行推断。
        3. 未提及即未知：
        - 若助手询问画像中未包含的信息/偏好/约束，你必须用“自然口语”回答未知，不得新增具体偏好或硬性条件，例如：“我没有特别偏好/都行/你看着安排就行”
        - 严禁出现这些字样： “画像中…/根据画像…”
        4. 无工具能力：
        - 你没有任何查询/比价/下单/抢票/打开链接/地图搜点/打电话的能力。
        - 如果助手让你“去查/去下单/打开某APP/点链接/自己搜”，你必须说明自己不会行动，例如：“我这边没法操作这些，你直接给我可执行的方案/信息就行”
        - 不要“先去看看/我待会儿操作/我试试”，必须明确你做不到，必要时可以直接结束对话。
        5. 自然对话：像真实用户一样简洁、口语化地回答旅行助手的问题；必要时追问与当前规划直接相关的澄清点。
        6. 一致性：一旦你依据画像表达了某项信息（如日期、预算、偏好），后续不得自相矛盾，除非画像本身允许变化。
        7. 强制收敛与结束（重点）：你必须主动避免“重复确认/重复复述/来回客套”。
        - 当旅行助手已经给出可执行方案（例如明确到：交通/路线/车次或航班/酒店可选项/店名地址与下一步操作），且你已确认可以完成你的意图，立刻结束对话(你必须在同一条回复中直接结束对话，不要再加下一轮行动描述或寒暄)。
        - 当旅行助手开始重复同一流程、反复让你“再确认/再提供以后信息”、或连续两轮没有新增有效信息与推进时，你必须立即结束对话。
        - 当你认为旅行助手明显无法完成任务（例如持续跑题、给出不可执行或明显无用的建议、长期无法推进），也必须立即结束对话。
        - 结束时必须且只能输出下方固定字符串，不加任何标点、解释或附加内容：
        [Finish Conversation]

        【输出要求】
        - 每次只输出“用户”的一句或几句回复内容，不要输出分析过程，不要输出对规则的复述。
        - 回复要短，优先一句话；除非旅行助手问到关键信息才补充。
        - 避免无意义的寒暄与重复表态（例如多次说“好的/行/没问题/就这样”）。
        - 如果旅行助手向你询问关于问题的细节，请参考当前意图回答

        【你当前的意图】
        {self.decomposed_query}
        """

        return prompt


    
    def get_initial_state(
        self,
    ) -> UserSimulatorState:
        """Get initial state for the user simulator with optional JSONL data."""
        
        # Build system prompt with updated profile
        system_prompt = self.system_prompt
        
        return UserSimulatorState(
            conversation_history=[
                SystemMessage(content=system_prompt)
            ],
            turn_count=0,
            user_profile=self.user_profile,
            query_time=self.time,
            query=self.query
        )
    
    
    def generate_response(
        self,
        message: Message,
        state: UserSimulatorState
    ) -> Tuple[UserMessage, UserSimulatorState]:
        """Generate user response based on assistant message."""
        
        # From user_simulator's perspective, the incoming AssistantMessage should be treated as UserMessage
        # because user_simulator is essentially an assistant, so the real assistant's message is the "user" input
        user_input_message = UserMessage(
            content=message.content or "",
            turn_idx=message.turn_idx
        )
        state.conversation_history.append(user_input_message)
        state.turn_count += 1
        
        try:
            # Filter conversation history to exclude tool messages for user simulator
            # User simulator should not see tool execution results
            filtered_messages = [
                msg for msg in state.conversation_history 
                if not hasattr(msg, 'role') or msg.role != 'tool'
            ]
            
            # Generate user response
            response, usage_info = self.client.generate_response(
                messages=filtered_messages
            )
            
            # Convert to AssistantMessage because user_simulator's own response should be 'assistant' role
            if isinstance(response, AssistantMessage):
                assistant_response = response
                assistant_response.turn_idx = state.turn_count
            else:
                assistant_response = AssistantMessage(
                    content=response.content or "",
                    turn_idx=state.turn_count
                )
            
            # Add user_simulator's response to history as AssistantMessage
            state.conversation_history.append(assistant_response)
            
            # But return as UserMessage for the conversation manager (normal perspective)
            user_message = UserMessage(
                content=assistant_response.content or "",
                turn_idx=state.turn_count
            )
            
            # Update metadata
            state.metadata.update({
                "last_usage": usage_info,
            })
            
            return user_message, state
            
        except Exception as e:
            # Generate fallback response
            fallback_responses = [
                "Could you help me with that?",
                "I'm not sure I understand. Can you explain more?",
                "What are my options here?",
                "Thank you for the information."
            ]
            
            user_message = UserMessage(
                content=random.choice(fallback_responses),
                turn_idx=state.turn_count
            )
            
            state.conversation_history.append(user_message)
            return user_message, state
    
    def is_conversation_finished(
        self,
        message: Message,
        state: UserSimulatorState
    ) -> bool:
        """Check if user wants to end the conversation."""
        # Check if user simulator's last message contains finish signal
        if isinstance(message, UserMessage) and "[Finish Conversation]" in message.content:
            return True
        
        # Check if reached maximum conversation length
        return state.turn_count > 20
    
    def generate_initial_message(self, state: UserSimulatorState) -> UserMessage:
        """Generate the first message to start conversation."""
        if self.query:
            content = f"{self.query}"
    
        return UserMessage(content=content, turn_idx=1)
