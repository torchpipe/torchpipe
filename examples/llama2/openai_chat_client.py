from openai import OpenAI
import time

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://0.0.0.0:8080/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

times = [time.perf_counter()]
# start_time = time.time()

completion = client.completions.create(model="",
                                      prompt="San Francisco is a",
                                      stream=True)
# times.append(time.time())
# print("Completion result:")
for message in completion:
    times.append(time.perf_counter())
    pass
    # print(message, '\n')
    # times.append(time.time())

print("Time taken for each request: ", [int(1000*(times[i] - times[i-1])) for i in range(1, len(times))])

print(f"total: {times[-1] - times[0]}")