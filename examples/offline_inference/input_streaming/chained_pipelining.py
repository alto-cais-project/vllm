# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from collections.abc import AsyncGenerator

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM


class StoryGenerator:

    def __init__(self, prompt: str):
        engine_args = AsyncEngineArgs(model="facebook/opt-6.7B",
                                      enforce_eager=True,
                                      gpu_memory_utilization=0.4)

        self.prompt = prompt
        self.engine = AsyncLLM.from_engine_args(engine_args)
        self.sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=300,
            output_kind=RequestOutputKind.DELTA)
        self.num_requests = 0

    def call(self) -> AsyncGenerator[RequestOutput, None]:
        self.num_requests += 1
        request_id = f"{self.__class__.__name__}-{self.num_requests}"
        return self.engine.generate(prompt=self.prompt,
                                    sampling_params=self.sampling_params,
                                    request_id=request_id)

    def __del__(self):
        self.engine.shutdown()


class StorySummarizer:

    def __init__(self):
        engine_args = AsyncEngineArgs(model="facebook/opt-6.7B",
                                      enforce_eager=True,
                                      gpu_memory_utilization=0.4)

        self.engine = AsyncLLM.from_engine_args(engine_args)
        self.sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=300,
            output_kind=RequestOutputKind.DELTA)
        self.num_requests = 0

    async def call(self, story_generator: StoryGenerator):

        async def process_outputs(generator: AsyncGenerator[RequestOutput,
                                                            None]):
            print("=" * 10, "Summary:")
            async for request_output in generator:
                for output in request_output.outputs:
                    if output.text:
                        print(output.text, end="", flush=True)
                if request_output.finished:
                    print("\n✅ Generation complete!")
                    break

        self.num_requests += 1
        request_id = f"{self.__class__.__name__}-{self.num_requests}"
        generator, input_streamer = await self.engine.create_input_streamer(
            prompt="Summarize the following story: ",
            sampling_params=self.sampling_params,
            request_id=request_id)

        output_task = asyncio.create_task(process_outputs(generator))

        story_generator_output = ""
        async for request_output in story_generator.call():
            for output in request_output.outputs:
                await input_streamer.stream(output.text)
                story_generator_output += output.text
            if request_output.finished:
                break

        await input_streamer.end()

        await output_task

        print("=" * 10, "Original story:")
        print(story_generator_output)

    def __del__(self):
        self.engine.shutdown()


async def main():
    story_generator = StoryGenerator(
        "Describe, in rich and continuous prose, the complete narrative of a vast interstellar epic that begins with political unrest and a hidden rebellion against a tyrannical empire. The story should follow a young protagonist from a remote desert world who becomes entangled in a galactic struggle between freedom and domination. Include themes of destiny, mentorship, redemption, the rise and fall of empires, and the conflict between an ancient spiritual order and a corrupted shadow of itself. The tone should feel mythic yet human—mixing the sweep of legend with the intimacy of personal transformation. Summarize the full arc of this saga as if recounting an old myth passed down through generations, complete with key events, turning points, and the ultimate reconciliation of light and darkness."
    )

    story_summarizer = StorySummarizer()

    await story_summarizer.call(story_generator)


if __name__ == "__main__":
    asyncio.run(main())
