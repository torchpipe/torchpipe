from openai import OpenAI
import time

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://0.0.0.0:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

times = [time.perf_counter()]
# start_time = time.time()

completion = client.completions.create(model="",
                                      prompt="San Francisco is a",
                                      stream=True,
                                      max_tokens=7)
# times.append(time.time())
# print("Completion result:")
msg = []
for message in completion:
    times.append(time.perf_counter())
    pass
    msg.append(message)
    # print(message, '\n')
    # times.append(time.time())

for m in msg:
    print(m)
    
print("Time taken for each request: ", [int(1000*(times[i] - times[i-1])) for i in range(1, len(times))])

print(f"total: {times[-1] - times[0]}")