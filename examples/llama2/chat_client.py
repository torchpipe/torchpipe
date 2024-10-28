import time
import asyncio
import aiohttp

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=60)

async def fetch_completion(session, url, headers, data):
    async with session.post(url, headers=headers, json=data) as response:
        async for line in response.content:
            line = line.strip()
            if not line:
                continue
            # print(line)
            yield line.decode('utf-8')

async def main():
    url = "http://0.0.0.0:8080/v1/completions"
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {
        "model": "string",
        "prompt": "San Francisco is a",
        "priority": "default",
        "n": 1,
        "best_of": 0,
        "max_tokens": 7,
        "stream": True,
        "stream_options": {
            "include_usage": True
        },
        "logprobs": 0,
        "echo": False,
        "temperature": 0.0,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "repetition_penalty": 1,
        "top_p": 1,
        "top_k": -1,
        "skip_special_tokens": True,
        "ignore_eos": False,
        "stop": "string",
        "stop_token_ids": [0]
    }
    times = []
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        # start_time = time.time()
        
        times = [time.perf_counter()]
        async for line in fetch_completion(session, url, headers, data):
            print(line, end='\n')
            # print(len(times))
        # end_time = time.time()
            times.append(time.perf_counter())

        # elapsed_time = end_time - start_time
        # print(f"\nElapsed time: {elapsed_time:.2f} seconds")
        print("Time taken for each request: ", [int(1000*(times[i] - times[i-1])) for i in range(1, len(times))])
        print(f"total: {times[-1] - times[0]}")
if __name__ == "__main__":
    asyncio.run(main())