# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.v1.engine.core_client import AsyncMPClient


class InputStreamerAsync:

    def __init__(self, request_id: str, engine_core: AsyncMPClient,
                 tokenizer: AnyTokenizer):
        self.request_id = request_id
        self._engine_core = engine_core
        self._tokenizer = tokenizer
        self._ended = False

    async def stream(self, text: str) -> None:
        assert not self._ended
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        await self.stream_tokens(token_ids)

    async def stream_tokens(self, token_ids: list[int]) -> None:
        assert not self._ended

        if not token_ids:
            return

        asyncio.create_task(
            self._engine_core.stream_prefill_tokens_async(
                self.request_id, token_ids))
        await asyncio.sleep(0)

    async def end(self) -> None:
        assert not self._ended

        self._ended = True

        asyncio.create_task(
            self._engine_core.stop_prefill_stream_async(self.request_id))
        await asyncio.sleep(0)
