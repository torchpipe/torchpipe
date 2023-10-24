// 推断请求参数
struct InferenceParams{
    // 唯一标识
    1:required string uuid;
    // 请求数据
    2:required binary data;
}

// 推断结果
struct InferenceResult{
    1: required string label;
    2: required double score;
}

// 推断接口
service InferenceService {
    void ping(),
    InferenceResult infer_batch(1:InferenceParams params),
}