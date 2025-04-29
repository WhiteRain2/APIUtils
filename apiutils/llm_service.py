# SPDX-License-Identifier: MIT

"""
LLMService 模块

提供对 OpenAI Async API 的封装，支持：
- 单轮或多轮对话（streaming / non-streaming）
- 批量并发查询（带限流、超时与重试）
- 会话历史管理

依赖：
  - openai.AsyncOpenAI
  - tqdm.asyncio.tqdm
"""

import time
import asyncio
from enum import Enum
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, NamedTuple


class _Roles(Enum):
    """定义对话中各参与方的角色标识。"""
    USER = 'user'
    ASSISTANT = 'assistant'
    DEVELOPER = 'system'  # developer


class QueriesResponse(NamedTuple):
    """定义查询响应的命名元组。"""
    query: str
    answer: str
    tokens: int


class LLMService:
    """
    LLMService 是对 OpenAI Async API 的异步封装。

    Attributes:
        client_config (Optional[Dict[str, Any]]): 全局客户端配置（api_key、base_url 等）。
        client (Optional[AsyncOpenAI]): 已初始化的 AsyncOpenAI 客户端实例。
    """
    client_config: Optional[Dict[str, Any]] = None
    client: Optional[AsyncOpenAI] = None

    def __init__(
        self,
        model: str,
        system_prompt: str = 'A helper',
        configs: Optional[Dict[str, Any]] = None
    ):
        """
        初始化会话。

        Args:
            model: 模型名称，例如 "gpt-4o-mini"。
            system_prompt: system 角色的初始提示，用于定制助手风格。
            configs: 其它 chat 接口参数，如 temperature、max_tokens 等。
        Raises:
            AttributeError: 如果尚未通过 `set_llm_client_config` 设置 client。
        """
        self.model: str = model
        self.system_prompt: str = system_prompt
        self.configs: Dict[str, Any] = configs or {}
        self.historical_messages: List[Dict[str, str]] = []
        self.total_tokens: int = 0
        if not LLMService.client:
            raise AttributeError('Please call LLMService.set_llm_client_config(...) to set the OpenAI client first.')
        self.client = LLMService.client

    @classmethod
    def set_llm_client_config(cls, **kwargs: Any) -> None:
        """
        类方法：初始化全局 AsyncOpenAI 客户端。

        必须传入 api_key 和 base_url。

        Args:
            kwargs: 包含 api_key、base_url 等参数。
        Raises:
            ValueError: 缺少 api_key 或 base_url。
        """
        # Error Check For api_key and base_url not in kwargs
        if 'api_key' not in kwargs or 'base_url' not in kwargs:
            raise ValueError('api_key and base_url must be provided')
        cls.client_config = kwargs
        cls.client = AsyncOpenAI(**kwargs)

    async def chat(
        self,
        user_query: str,
        stream: bool = True,
        configs: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        与 LLM 进行单轮或多轮对话。

        Args:
            user_query: 本轮用户提问内容。
            stream: 是否使用流式返回。
            configs: 本轮可覆写的接口参数。
        Yields:
            assistant 生成的文本片段（stream=True）或完整回答（stream=False）。
        """
        cfg = configs or self.configs
        await self._add_historical_message(_Roles.USER.value, user_query)
        completion = await self._create_completion(messages=self.historical_messages, stream=stream, configs=cfg)
        if not stream:
            async for chunk in self._none_stream_chat(completion):
                yield chunk
        else:
            async for chunk in self._stream_chat(completion):
                yield chunk

    async def _none_stream_chat(
        self,
        completion: Any
    ) -> AsyncGenerator[str, None]:
        """
        处理非流式返回。

        Args:
            completion: AsyncOpenAI.chat.completions.create 的返回对象。
        Yields:
            完整回答字符串（一次产生）。
        """
        # 非流式时，choices[0] 应为最终回答
        assistant_reply: str = completion.choices[0].message.content
        # 更新 token 计数
        self.total_tokens += completion.usage.total_tokens  # type: ignore[attr-defined]
        # 添加历史
        await self._add_historical_message(_Roles.ASSISTANT.value, assistant_reply)
        yield assistant_reply

    async def _stream_chat(
        self,
        completion: Any
    ) -> AsyncGenerator[str, None]:
        """
        处理流式返回。

        Args:
            completion: 异步可迭代的流数据。
        Yields:
            每次流式内容增量。
        """
        buffer: List[str] = []
        async for chunk in completion:
            # choices 可能为空，需判空后再访问
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    buffer.append(delta.content)
                    yield delta.content
            # 有 usage 时累加（一般在流尾）
            if hasattr(chunk, 'usage') and chunk.usage:
                self.total_tokens += getattr(chunk.usage, 'total_tokens', 0) or 0
        # 合并并落历史
        full_reply = ''.join(buffer)
        await self._add_historical_message(_Roles.ASSISTANT.value, full_reply)

    async def queries(
        self,
        questions: List[str],
        tqdm_title: Optional[str] = 'Processing',
        batch_size: int = 50,
        delay: float = 0.5,
        max_retries: int = 3
    ) -> List[QueriesResponse]:
        """
        并发批量提问接口，带限流、重试与超时。

        Args:
            questions: 待提问列表。
            tqdm_title: 进度条文案, 默认 'Processing'，若指定为 None 则不显示进度条。
            batch_size: 并发量上限。
            delay: 重试前等待基准时长。
            max_retries: 单条最大重试次数。
        Returns:
            List[QueriesResponse]: [(问题, 回答, 消耗 token), ...]
        """
        semaphore = asyncio.Semaphore(batch_size)

        async def fetch_with_retry(q: str) -> Tuple[Optional[str], int]:
            retries = 0
            while True:
                try:
                    async with semaphore:
                        ans, tok = await self.query(q)
                    if not ans:
                        raise ValueError("Empty response")
                    return ans, tok
                except Exception as e:
                    if retries >= max_retries:
                        # 达到重试上限，记录空结果
                        print(f"Retry limit exceeded: {q[:10]}... -> {e}")
                        return '', 0
                    retries += 1
                    await asyncio.sleep(delay * (2 ** retries))

        # 并发执行并用 tqdm 显示进度
        tasks = [fetch_with_retry(q) for q in questions]
        if tqdm_title:
            raw_results = await tqdm.gather(*tasks, desc=tqdm_title)
        else:
            raw_results = await asyncio.gather(*tasks)

        # 结构化返回
        return [
            QueriesResponse(query=q, answer=ans, tokens=tok)
            for q, (ans, tok) in zip(questions, raw_results)
        ]

    async def query(
        self,
        question: str,
        configs: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, int]:
        """
        单次流式提问，返回完整回答和消耗 token。

        Args:
            question: 用户提问。
            configs: 可选的本轮接口参数覆盖。
        Returns:
            (完整回答, 消耗 token 数)
        """
        cfg = configs or self.configs
        # 调用底层流式接口
        completion = await self._create_completion(
            messages=[{'role': _Roles.USER.value, 'content': question}],
            stream=True,
            configs=cfg
        )

        total_tokens = 0
        buffer: List[str] = []
        iterator = aiter(completion)
        start_ts = time.time()

        while True:
            # 总超时：60 秒
            if time.time() - start_ts > 60:
                print("Total timeout reached, stop reading")
                break
            try:
                # 单次等待超时：20 秒
                chunk = await asyncio.wait_for(anext(iterator), timeout=20)
            except asyncio.TimeoutError:
                print("Single timeout, stop streaming reads")
                break
            except StopAsyncIteration:
                break
            except Exception as err:
                print(f"Error processing block: {err}")
                break

            # 读取内容
            if chunk.choices and chunk.choices[0].delta.content:
                buffer.append(chunk.choices[0].delta.content)
            # 累加 token（部分流响应最后会携带 usage）
            if hasattr(chunk, 'usage') and chunk.usage:
                total_tokens += getattr(chunk.usage, 'total_tokens', 0) or 0

        return ''.join(buffer), total_tokens

    async def init_history(self, history: List[Tuple[str, str]]) -> None:
        """
        根据已有问答对初始化历史记录。

        Args:
            history: [(用户提问, 助手回答), ...]
        """
        await self._cleanup_historical_message()
        for q, a in history:
            await self._add_historical_message(_Roles.USER.value, q)
            await self._add_historical_message(_Roles.ASSISTANT.value, a)

    async def _create_completion(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        configs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        底层封装：调用 OpenAI Async 客户端的 chat 接口。

        Args:
            messages: 完整的消息列表（含 system+历史+本轮 user）。
            stream: 是否流式返回。
            configs: 覆写的接口参数。
        Returns:
            AsyncOpenAI.chat.completions.create 的返回值（可能是异步迭代器）。
        """
        cfg = configs or self.configs
        if stream:
            # 流式时开启 usage 汇报
            cfg.setdefault('stream_options', {'include_usage': True})

        # 构造 system prompt
        system_msgs: List[Dict[str, str]] = []
        if self.system_prompt:
            system_msgs.append({
                'role': _Roles.DEVELOPER.value,
                'content': self.system_prompt
            })

        try:
            # 整体超时控制
            return await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=system_msgs + messages,
                    stream=stream,
                    **cfg
                ),
                timeout=50
            )
        except asyncio.TimeoutError:
            raise RuntimeError("LLM Request timeout (>50s)")

    async def _add_historical_message(self, role: str, message: str) -> None:
        """
        向会话历史追加一条消息，并维护最大长度。

        Args:
            role: 消息角色，如 'user'、'assistant'。
            message: 消息内容。
        """
        self.historical_messages.append({'role': role, 'content': message})
        # 只保留最新 100 条记录（或按需调整）
        if len(self.historical_messages) > 100:
            # 删除最早的第二条（保留 system 在首）
            self.historical_messages.pop(1)

    async def _cleanup_historical_message(self) -> None:
        """清空所有会话历史。"""
        self.historical_messages.clear()

    def __len__(self) -> int:
        """
        支持 len(service)，返回当前会话历史消息总数。
        """
        return len(self.historical_messages)
