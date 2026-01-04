namespace py image_processing

struct ProcessRequest {
    1: binary image_data,
}

struct ProcessResponse {
    1: bool success,
    2: string result_data,
    3: string error_message
}

service ImageProcessingService {
    ProcessResponse process_image(1: ProcessRequest request)
}