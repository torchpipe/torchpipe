

#include "omniback/ffi/dict.h"
#include "tvm/ffi/container/map.h"
#include "tvm/ffi/container/variant.h"
#include <tvm/ffi/reflection/registry.h>
#include "tvm/ffi/container/array.h"
#include "tvm/ffi/extra/stl.h"

#include <tvm/ffi/function.h>
#include <tvm/ffi/error.h>

namespace omniback::ffi {

// tvm::ffi::Any dict2obj(const dict value){
//    return  omniback::ffi::DictRef(tvm::ffi::make_object<omniback::ffi::DictObj>(value));
// }



namespace ffi = tvm::ffi;
namespace refl = tvm::ffi::reflection;


TVM_FFI_STATIC_INIT_BLOCK() {
  refl::ObjectDef<DictObj>()
      .def(refl::init<tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>())
      .def(
          "__len__",
          [](const DictObj* self) -> size_t { return self->GetMap().size(); })

      .def_rw("callback", &DictObj::py_callback)
      .def(
          "__getitem__",
          [](const DictObj* self,
             const tvm::ffi::String& key) -> omniback::any {
            const auto& map = self->GetMap();
            auto it = map.find(key);
            if (it == map.end()) {
              TVM_FFI_THROW(KeyError)
                  << "Key '" << key << "' not found in dict";
            }
            return it->second;
          })
      .def(
          "__setitem__",
          [](DictObj* self,
             const tvm::ffi::String& key,
             const tvm::ffi::Any& value) {
            self->GetMutableMap()[key] = value;
          })

      // // __delitem__
      .def(
          "__delitem__",
          [](DictObj* self, const tvm::ffi::String& key) {
            auto& map = self->GetMutableMap();
            if (map.erase(key) == 0) {
              TVM_FFI_THROW(KeyError)
                  << "Key '" << key << "' not found for deletion";
            }
          })
      .def(
          "__contains__",
          [](const DictObj* self, const tvm::ffi::String& key) -> bool {
            return self->GetMap().find(key) != self->GetMap().end();
          })
      .def(
          "get",
          [](const DictObj* self, const tvm::ffi::String& key
             //  ,const tvm::ffi::Optional<omniback::any>& default_val
             ) -> std::optional<omniback::any> {
            const auto& map = self->GetMap();
            auto it = map.find(key);
            return it != map.end() ? it->second : std::nullopt;
          })
      .def(
          "keys",
          [](const DictObj* self) -> tvm::ffi::Array<tvm::ffi::String> {
            const auto& map = self->GetMap();
            std::vector<tvm::ffi::String> keys;
            keys.reserve(map.size());
            for (const auto& pair : map) {
              keys.push_back(pair.first);
            }
            return tvm::ffi::Array<tvm::ffi::String>{keys.begin(), keys.end()};
          })

      // values()
      .def(
          "values",
          [](const DictObj* self) -> tvm::ffi::Array<omniback::any> {
            const auto& map = self->GetMap();
            std::vector<omniback::any> values;
            values.reserve(map.size());
            for (const auto& pair : map) {
              values.push_back(pair.second);
            }
            return {values.begin(), values.end()};
          })

      // items()
      .def(
          "items_old",
          [](const DictObj* self)
              -> std::vector<std::tuple<tvm::ffi::String, omniback::any>> {
            const auto& map = self->GetMap();
            std::vector<std::tuple<tvm::ffi::String, omniback::any>> items;
            items.reserve(map.size());
            for (const auto& pair : map) {
              items.emplace_back(pair.first, pair.second);
            }
            return items;
          })
      .def(
          "items",
          [](const DictObj* self)
              -> std::vector<std::tuple<tvm::ffi::String, omniback::any>> {
            const auto& map = self->GetMap();
            std::vector<std::tuple<tvm::ffi::String, omniback::any>> items;
            items.reserve(map.size());
            for (const auto& pair : map) {
              items.emplace_back(pair.first, pair.second);
            }
            return items;
          })

      // clear()
      .def("clear", [](DictObj* self) { self->GetMutableMap().clear(); })
      .def(
          "pop",
          [](DictObj* self, const std::string& key) -> omniback::any {
            auto& map = self->GetMutableMap();
            auto it = map.find(key);
            if (it != map.end()) {
              omniback::any val = std::move(it->second);
              map.erase(it);
              return val;
            }
            TVM_FFI_THROW(KeyError) << "Key '" << key << "' not found in dict";
          })
      .def(
          "update",
          [](DictObj* self,
             const tvm::ffi::Variant<
                 DictObj*,
                 tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>& other) {
            // Dict
            if (auto data = other.as<DictObj*>()) {
              auto& map = self->GetMutableMap();
              const auto& other_map = data.value()->GetMap();
              for (const auto& pair : other_map) {
                map.insert_or_assign(pair.first, pair.second);
              }
            } else {
              auto map_data =
                  other.as<tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>();
              auto& map = self->GetMutableMap();
              for (const auto& pair : map_data.value()) {
                map.insert_or_assign(pair.first, pair.second);
              }
            }
          })
      .def("copy", [](const DictObj* self) {
        return std::make_shared<std::unordered_map<std::string, omniback::any>>(
            self->GetMap());
      });
}
}
