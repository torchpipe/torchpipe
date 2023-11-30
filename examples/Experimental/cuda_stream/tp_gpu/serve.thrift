// 推断请求参数
struct InferenceParams{
    // 唯一标识(追踪单次请求，日志打印)
    1:required string uuid;
    // 请求数据
    2:required binary data;
}


// 推断状态枚举
enum InferenceStatusEnum {
    // 成功
    OK = 200,
    // 异常，配合异常信息使用
    EXCEPTION = 400
}

// 推断结果
struct InferenceResult{
    // 推断状态 (必填)
    1: required InferenceStatusEnum status;
    // 推断耗时（必填）
    2: required i32 index;
    // 推算综合得分（必填）
    3: required double score;
    // 补充信息说明 (可选,处理异常时可填写错误信息)
    4: optional string message;
}

// 推断接口
service InferenceService {
    void ping(),
    // 批量推断 (map中的key为“标签，如porno，cnmap”，value为推断结果)
    InferenceResult forward(1:InferenceParams params),
}