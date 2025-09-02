import time
import asyncio
import aiohttp

import json

def extract_text_from_data_str(data_str: str) -> str:
    # 去除前后空白字符
    data_str = data_str.strip()
    
    # 检查是否是结束标记
    if data_str == "data: [DONE]":
        return "\n<DONE>"
    
    # 去除"data: "前缀（如果存在）
    json_str = data_str[6:] if data_str.startswith("data: ") else data_str
    
    # 再次去除空白字符
    json_str = json_str.strip()
    
    # 如果是空字符串，返回空
    if not json_str:
        return ""
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        # 尝试修复常见的JSON格式问题
        try:
            # 尝试匹配最内层的JSON对象
            match = re.search(r'\{.*\}', json_str, re.DOTALL)
            if match:
                fixed_json = match.group()
                data = json.loads(fixed_json)
            else:
                raise ValueError(f"Invalid JSON string: {json_str}") from e
        except Exception:
            raise ValueError(f"Failed to parse JSON from string: {json_str}") from e
    
    # 安全地提取text字段
    if isinstance(data, dict):
        if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if isinstance(choice, dict) and "text" in choice:
                return choice["text"]
    
    return ""

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=1000)

async def fetch_completion(session, url, headers, data):
    async with session.post(url, headers=headers, json=data) as response:
        async for line in response.content:
            line = line.strip()
            if not line:
               continue
            # print(line)
            yield line.decode('utf-8')


async def main(prompt="San Francisco is a", port = 8000, max_tokens=7):
    url = f"http://0.0.0.0:{port}/v1/completions"
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {
        "model": "./Llama-2-7b-chat-hf/",
        "prompt": prompt,
        # "priority": "default",
        "n": 1,
        "best_of": 1,
        "max_tokens": max_tokens,
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
        "stop_token_ids": []
    }
    # times = []
    all_result = []
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        # start_time = time.time()
        
        times = [time.perf_counter()]
        async for line in fetch_completion(session, url, headers, data):
            # print(line, end='\n')
            all_result.append(line)
            # print(len(times))
        # end_time = time.time()
            times.append(time.perf_counter())

        # elapsed_time = end_time - start_time
        # print(f"\nElapsed time: {elapsed_time:.2f} seconds")
        print("Time taken for each request: ", [int(1000*(times[i] - times[i-1])) for i in range(1, len(times) - 1)])
        print(f"total: {times[-1] - times[0]}")
        
    text = ""
    for line in all_result:
        print(line)
        if not line:
            continue
        # print(type(line))
        text += extract_text_from_data_str(line)
    print(f"Final text: {text}")
if __name__ == "__main__":
    # asyncio.run(main())
    import fire
    fire.Fire(main)
