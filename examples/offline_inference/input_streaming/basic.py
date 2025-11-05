# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM


async def main():
    engine_args = AsyncEngineArgs(
        model="facebook/opt-125m",
        enforce_eager=True,  # Faster startup for examples
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=100,
        output_kind=RequestOutputKind.DELTA,
    )

    # Create a request with input streaming enabled
    output_generator, input_streamer = await engine.create_input_streamer(
        prompt="The future of",
        sampling_params=sampling_params,
        request_id="example-request-1",
    )

    # Process outputs in a background task
    async def process_outputs():
        async for output in output_generator:
            for completion in output.outputs:
                new_text = completion.text
                if new_text:
                    print(new_text, end="", flush=True)

            if output.finished:
                print("\n✅ Generation complete!")
                break

    output_task = asyncio.create_task(process_outputs())

    # Stream additional prompt text dynamically
    # These can be called at any time, even while the model is processing
    await asyncio.sleep(0.1)  # Simulate some delay
    input_streamer.stream(" artificial")

    await asyncio.sleep(0.1)
    input_streamer.stream(" intelligence")

    await asyncio.sleep(0.1)
    input_streamer.stream(" is")

    # Signal that prompt streaming is complete
    input_streamer.end()

    # Wait for output processing to complete
    await output_task

    engine.shutdown()

    print("Streaming complete!")


if __name__ == "__main__":
    asyncio.run(main())
