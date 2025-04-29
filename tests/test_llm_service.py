import pytest
import asyncio
import time
from typing import Any, List, Dict, Tuple
from apiutils.llm_service import LLMService, QueriesResponse, _Roles

# Use pytest-asyncio for async tests
env = pytest.importorskip("pytest_asyncio")


class DummyChunk:
    def __init__(self, content: str = "", usage: int = None):
        # Simulate "choices" structure
        class Delta:
            def __init__(self, content):
                self.content = content
        class Choice:
            def __init__(self, delta):
                self.delta = delta
                self.usage = None
        self.choices = [Choice(Delta(content))] if content is not None else []
        if usage is not None:
            self.usage = type("U", (), {"total_tokens": usage})()
        else:
            self.usage = None


class DummyNonStreamCompletion:
    def __init__(self, text: str, tokens: int):
        # choices[0].message.content and usage.total_tokens
        class Message:
            def __init__(self, content): self.content = content
        self.choices = [type("C", (), {"message": Message(text)})]
        self.usage = type("U", (), {"total_tokens": tokens})()


class DummyAsyncOpenAI:
    """
    Dummy AsyncOpenAI replacement with chained attributes:
    client.chat.completions.create
    """
    def __init__(self, **kwargs: Any):
        self.chat = self
        self.completions = self

    async def create(self, model: str, messages: List[Dict[str, str]], stream: bool, **cfg) -> Any:
        # Return dummy based on stream
        if stream:
            # async generator with two chunks
            async def gen():
                yield DummyChunk("hello")
                chunk_end = DummyChunk(None, usage=5)
                yield chunk_end
            return gen()
        else:
            return DummyNonStreamCompletion("final", 10)


@pytest.fixture(autouse=True)
def patch_openai(monkeypatch):
    # Patch AsyncOpenAI in module
    monkeypatch.setattr(LLMService, 'client', None)
    monkeypatch.setattr(LLMService, 'client_config', None)
    monkeypatch.setattr('apiutils.llm_service.AsyncOpenAI', DummyAsyncOpenAI)


@pytest.fixture(autouse=True)
def setup_client():
    # Initialize client config
    LLMService.set_llm_client_config(api_key='test', base_url='https://api.example.com')

@pytest.mark.asyncio
async def test_chat_streaming_and_tokens():
    service = LLMService('model-1')
    # Test streaming
    chunks = []
    async for c in service.chat('hi', stream=True):
        chunks.append(c)
    # Should collect 'hello'
    assert chunks == ['hello']
    # total_tokens updated from usage
    assert service.total_tokens == 5

@pytest.mark.asyncio
async def test_chat_non_streaming_and_history():
    service = LLMService('model-1')
    # Test non-streaming
    responses = [c async for c in service.chat('hi', stream=False)]
    assert responses == ['final']
    # tokens and history updated
    assert service.total_tokens == 10
    # history should have two entries: user + assistant
    assert len(service.historical_messages) == 2
    assert service.historical_messages[0]['role'] == _Roles.USER.value
    assert service.historical_messages[1]['role'] == _Roles.ASSISTANT.value

@pytest.mark.asyncio
async def test_queries_with_retry_and_namedtuple(monkeypatch):
    service = LLMService('model-1')
    # Patch service.query to fail first then succeed
    calls = {'count': 0}
    async def fake_query(q: str) -> Tuple[str, int]:
        calls['count'] += 1
        if calls['count'] == 1:
            return '', 0  # first empty triggers retry
        return ('ok:' + q, len(q))
    monkeypatch.setattr(service, 'query', fake_query)
    # Provide 2 questions
    qs = ['one', 'two']
    results = await service.queries(qs, batch_size=2, max_retries=2)
    # Results should be list of QueriesResponse
    assert all(isinstance(r, QueriesResponse) for r in results)
    assert results[0].query == 'one'
    assert results[0].answer == 'ok:one'
    assert results[0].tokens == 3

# @pytest.mark.asyncio
# async def test_query_timeout_and_buffers(monkeypatch):
#     # Simulate _create_completion that hangs
#     service = LLMService('model-1')
#     async def never_ending(*args, **kwargs):
#         while True:
#             await asyncio.sleep(0.01)
#     monkeypatch.setattr(service, '_create_completion', lambda *args, **kwargs: never_ending())
#     # Calling query should eventually break after timeouts, but returns empty
#     result, tokens = await service.query('q')
#     assert result in ('',)
#     assert isinstance(tokens, int)

@pytest.mark.asyncio
async def test_init_history_and_len():
    service = LLMService('model-1')
    hist = [('q1', 'a1'), ('q2', 'a2')]
    await service.init_history(hist)
    # len returns count of messages (2 pairs -> 4 messages)
    assert len(service) == 4

@pytest.mark.asyncio
async def test_set_llm_client_config_errors():
    # Missing keys
    with pytest.raises(ValueError):
        LLMService.set_llm_client_config(api_key='k')
    with pytest.raises(ValueError):
        LLMService.set_llm_client_config(base_url='url')
