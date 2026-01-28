#include "omniback/builtin/partial_backend.hpp"


namespace om {
using ffi::DictRef;
using om::dict;

PartialBackend::InitCallback PartialBackend::partial_ffi2init(FfiPartialInitFunc init_func){
    return 
        [init_func](const om::dict& self,
            const std::unordered_map<std::string, std::string>& params,
            const om::dict& options) {

          tvm::ffi::Map<tvm::ffi::String, tvm::ffi::String> params_map{params.begin(), params.end()};

          tvm::ffi::Optional<DictRef> options_dict;
          if (options) {
            auto options_dict =
                DictRef(tvm::ffi::make_object<om::ffi::DictObj>(options));
          }
          auto self_dict = DictRef(tvm::ffi::make_object<om::ffi::DictObj>(self));

          (init_func)(self_dict, params_map, options_dict);
        };
}

PartialBackend::ForwardCallback PartialBackend::partial_ffi2forward(FfiPartialForwardFunc forward_func){
  return
      [forward_func](const om::dict& self,
        const std::vector<om::dict>& ios) {

        tvm::ffi::Array<DictRef> arr;
        for (const auto& io_dict : ios) {
          auto dict_obj =
              tvm::ffi::make_object<om::ffi::DictObj>(io_dict);
          arr.push_back(DictRef(dict_obj));
        }
        auto self_dict = tvm::ffi::make_object<om::ffi::DictObj>(self);

        (forward_func)(DictRef(self_dict), std::move(arr));
      };
}

// PartialBackend::MaxMinCallback PartialBackend::partial_ffi2maxmin(FfiPartialMaxMinFunc max_min_func){
//   if (auto value = max_min_func.as<uint32_t>()) {
//         const uint32_t re = value.value()
//        return [re](const om::dict& self) { return re;};
//     }

//     auto func =
//               max_min_func.get<tvm::ffi::TypedFunction<uint32_t(om::ffi::DictRef)>>(); 
    
//     return [func](const om::dict& self) {
//         auto self_dict = DictRef(tvm::ffi::make_object<om::ffi::DictObj>(self));
//          return (func)(self_dict);
//       };
// }


PartialBackend::MaxMinCallback PartialBackend::partial_ffi2maxmin(FfiPartialMaxMinFunc max_min_func){
    return [max_min_func](const om::dict& self) {
        auto self_dict = DictRef(tvm::ffi::make_object<om::ffi::DictObj>(self));
         return (max_min_func)(self_dict);
      };
}

}  // namespace om

// OMNI_PARTIAL_REGISTER(Backend, PartialBackend, "your_partial_backend_name");

