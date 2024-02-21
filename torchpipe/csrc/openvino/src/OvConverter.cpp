#include "OvConverter.hpp"

#include "openvino/openvino.hpp"
#include "opencv2/core.hpp"
#include "base_logging.hpp"
namespace ipipe {
namespace model {

template <typename T>
T* SingleInstance() {
  static T _instance{};
  return &_instance;
};

ov::Layout get_layout(const ov::Output<ov::Node>& node) {
  auto layout = ov::layout::get_layout(node);
  if (layout.empty()) {
    auto shp = node.get_shape();
    if (shp.size() == 4) layout = {"NCHW"};
  }
  return layout;
}

struct ModelObject {
  ModelObject(std::string p, int instance_num = -1) {
    model = SingleInstance<ov::Core>()->read_model(p);

    // info from model
    ov::OutputVector outputs = model->outputs();
    for (auto& item : outputs) {
      out_names.push_back(item.get_any_name());
      out_layouts.push_back(get_layout(item));
    }

    ov::OutputVector inputs = model->inputs();
    for (auto& item : inputs) {
      in_names.push_back(item.get_any_name());
      in_layouts.push_back(get_layout(item));
    }

    // model shape
    // https://docs.openvino.ai/2023.3/openvino_docs_OV_UG_DynamicShapes.html
    std::map<ov::Output<ov::Node>, ov::PartialShape> port_to_shape;
    for (const ov::Output<ov::Node>& input : model->inputs()) {
      ov::PartialShape shape = input.get_partial_shape();
      shape[0] = 1;
      port_to_shape[input] = shape;
      in_shapes.push_back(shape.get_min_shape());
    }
    model->reshape(port_to_shape);

    // preprocess

    ov::preprocess::PrePostProcessor ppp(model);

    const ov::Layout img_layout = {"HWC"};
    for (std::size_t i = 0; i < in_layouts.size(); i++) {
      ppp.input(in_names[i]).tensor().set_layout(img_layout);
      ppp.input(in_names[i]).preprocess().convert_element_type(ov::element::f32);
      // .mean({103.94f, 116.78f, 123.68f})
      // .scale({57.21f, 57.45f, 57.73f});
      ppp.input(in_names[0]).model().set_layout(in_layouts[i]);
    }
    for (std::size_t i = 0; i < out_layouts.size(); i++) {
      ppp.output(out_names[i]).model().set_layout(out_layouts[i]);
      if (out_layouts[i] == "NCHW") {
        ppp.output(out_names[i]).tensor().set_layout({"NHWC"}).set_element_type(ov::element::f32);
      } else {
        ppp.output(out_names[i]).tensor().set_element_type(ov::element::f32);
      }
    }

    model = ppp.build();

    compiled_model = SingleInstance<ov::Core>()->compile_model(
        model, "CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
    optimal_number_requests = compiled_model.get_property(ov::optimal_number_of_infer_requests);
  }

  int optimal_number_requests;
  std::vector<std::string> out_names;
  std::vector<std::string> in_names;
  std::vector<ov::Layout> in_layouts;
  std::vector<ov::Layout> out_layouts;
  std::vector<ov::Shape> in_shapes;
  // std::vector<ov::Shape> out_shapes;

  ov::CompiledModel compiled_model;
  std::shared_ptr<ov::Model> model;  // = m_config.m_core.read_model(m_config.m_path_to_model);
};

bool OvConverter::init(const std::unordered_map<std::string, std::string>& config_param) {
  params_ = std::unique_ptr<Params>(
      new Params({{"_independent_thread_index", "0"}, {"instance_num", "1"}, {"precision", "fp32"}},
                 {"model"}, {}, {}));
  if (!params_->init(config_param)) return false;

  int instance_num = std::stoi(params_->at("instance_num"));
  // instance_num = (instance_num <= 0) ? 1 : instance_num;
  // if (instance_num <= _independent_thread_index) {
  //   SPDLOG_ERROR("instance_num <= _independent_thread_index: " + std::to_string(instance_num) +
  //                " <= " + std::to_string(_independent_thread_index));
  //   return false;
  // }

  model_ = std::make_shared<ModelObject>((params_->at("model")), instance_num);

  int optimal_number_requests = model_->optimal_number_requests;
  if (optimal_number_requests < instance_num) {
    SPDLOG_ERROR(
        "The number of instances({}) is greater than the optinal number of instances({}). This "
        "generally "
        "means that there are not enough CPU resources. Reset instance_num please.",
        instance_num, optimal_number_requests);
    return false;
  } else if (optimal_number_requests > instance_num) {
    SPDLOG_WARN(
        "The number of instances({}) is less than the optinal number of instances({}). {}",
        instance_num, optimal_number_requests,
        colored(" This generally means that there are excess CPU resources available for achieving "
                "better throughput. "
                "Increase instance_num to improve throughput if you can use more CPU resources."));
  }

  // std::string precision = params_->at("precision");
  // //
  // https://docs.openvino.ai/2023.0/groupov_dev_api_system_conf.html#doxid-group-ov-dev-api-system-conf-1gad1a071adcef91309ca90878afd83f4fe
  // // if (precision == "bf16") {
  // //   IPIPE_ASSERT(ov::with_cpu_x86_bfloat16() || ov::with_cpu_x86_avx512_core_amx_bf16());
  // // } else if (precision == "int8") {
  // //   IPIPE_ASSERT(ov::with_cpu_x86_avx512_core_vnni() ||
  // ov::with_cpu_x86_avx512_core_amx_int8());
  // // }
  // // else if (precision == "fp32"){
  // //   IPIPE_ASSERT(ov::with_cpu_x86_avx512f()||ov::with_cpu_x86_avx512_core_amx());

  // IPIPE_ASSERT(precision == "fp32");  // todo: support bf16, int8

  return true;
}

template <typename T = std::size_t>
static inline std::vector<T> getShape(const cv::Mat& mat) {
  std::vector<T> result(mat.dims + 1);  // Add one for the channels
  for (int i = 0; i < mat.dims; i++) result[i] = (T)mat.size[i];
  result[mat.dims] = mat.channels();  // Add the channels
  return result;
}

cv::Mat OvTensor2Mat(const ov::Tensor& tensor) {
  ov::Shape shape = tensor.get_shape();
  if (shape.size() == 4) {
    if (shape[0] != 1) {
      throw std::runtime_error("Invalid batch size(!=1) in output");
    }
    int h = shape[1];
    int w = shape[2];
    int c = shape[3];
    assert(c == 3 || c == 1);

    auto cc = c == 3 ? CV_32FC3 : CV_32FC1;
    // SPDLOG_DEBUG("tensor.data<float>()  {} {} {} ", tensor.data<float>()[0],
    //              tensor.data<float>()[1], tensor.data<float>()[2]);
    return cv::Mat(std::vector<int>{h, w}, cc, tensor.data<float>()).clone();
  } else if (shape[2] == 3 || shape[2] == 1) {
    int h = shape[0];
    int w = shape[1];
    int c = shape[2];
    auto cc = c == 3 ? CV_32FC3 : CV_32FC1;

    return cv::Mat(std::vector<int>{h, w}, cc, tensor.data<float>()).clone();
  } else if (shape.size() == 3) {
    int n = shape[0];
    int t = shape[1];
    int c = shape[2];
    if (n != 1) {
      throw std::runtime_error("Invalid batch size(!=1) in output");
    }
    return cv::Mat(std::vector<int>{t, c}, CV_32F, tensor.data<float>()).clone();

  } else if (shape.size() == 2) {
    // if (shape[0] != 1) {
    //   throw std::runtime_error("Invalid batch size(!=1) in output");
    // }
    return cv::Mat(std::vector<int>{shape[0], shape[1]}, CV_32F, tensor.data<float>()).clone();
  }
  throw std::runtime_error("Invalid output shape");
  return cv::Mat();  // make gcc happy
}

ov::Tensor Mat2OvTensor(const cv::Mat& m) {
  std::vector<std::size_t> shape = getShape<std::size_t>(m);
  if (m.type() == CV_32F || m.type() == CV_32FC3)
    return ov::Tensor(ov::element::f32, shape, m.data);
  else if (m.type() == CV_8U || m.type() == CV_8UC3)
    return ov::Tensor(ov::element::u8, shape, m.data);
  else if (m.type() == CV_8SC1)
    return ov::Tensor(ov::element::i8, shape, m.data);
  else if (m.type() == CV_32SC1)
    return ov::Tensor(ov::element::i32, shape, m.data);
  else
    throw std::runtime_error(std::string("Mat2OvTensor: unsupported type"));
}

class OvInstance : public ModelInstance {
 public:
  OvInstance(ov::CompiledModel* cm) { infer_request_ = cm->create_infer_request(); };
  virtual ~OvInstance() = default;
  virtual void forward() override {
    infer_request_.start_async();
    infer_request_.wait();
  }
  // virtual bool init(const std::unordered_map<std::string, std::string>& config_param) override {
  //   return true;
  // }
  virtual void set_input(const std::string& name, const any& data) override {
    cv::Mat input = any_cast<cv::Mat>(data);
    infer_request_.set_tensor(name, Mat2OvTensor(input));
  }
  virtual any get_output(const std::string& name) override {
    auto data = infer_request_.get_tensor(name);
    return OvTensor2Mat(data);
  }

 private:
  ov::InferRequest infer_request_;
};

std::unique_ptr<ModelInstance> OvConverter::createInstance() {
  model_->compiled_model.create_infer_request();
  return std::make_unique<model::OvInstance>(&(model_->compiled_model));
}
std::vector<std::string> OvConverter::get_input_names() { return model_->in_names; }
std::vector<std::string> OvConverter::get_output_names() { return model_->out_names; }

}  // namespace model
}  // namespace ipipe