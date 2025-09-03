import os
import dashscope

messages = [
    {
	"role": "system",
	"content": [{"text": "You are a helpful assistant."}]},
    {
        "role": "user",
        "content": [
            # 第一张图像url
            # 如果传入本地文件，请将url替换为：file://ABSOLUTE_PATH/test.jpg，ABSOLUTE_PATH为本地文件的绝对路径
            {"image": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"},
            # 第二张图像url
            {"image": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/tiger.png"},
            # 第三张图像url
            {"image": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241108/hbygyo/rabbit.jpg"},
            {"text": "这些图描绘了什么内容?"}
        ]
    }
]

response = dashscope.MultiModalConversation.call(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    # 此处以qwen-vl-max-latest为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/models
    model='qwen-vl-max-latest',
    messages=messages
)

print(response.output.choices[0].message.content[0]["text"])
