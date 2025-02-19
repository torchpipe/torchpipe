// // Copyright 2021-2024 NetEase.
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //  http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.

// #include <string>

// #include <stdexcept>
// #include "hami/helper/symbol.hpp"
// #include "hami/core/any.hpp"

// namespace nonstd {
// namespace any_lite {
// namespace {
// void preplace(std::string& input, const std::string& a, const std::string& b)
// {
//   size_t index;

//   while ((index = input.find(a, 0)) != std::string::npos) {
//     input.replace(index, a.size(), b);
//   }
// }
// }  // namespace
// std::string get_type_name(const std::type_info& info) {
//   auto src = hami::local_demangle(info.name());
//   preplace(src, "std::", "");

//   // auto iter = src.find(',');
//   // if (iter != std::string::npos) {
//   //   src = src.substr(0, iter) + '>';
//   // }
//   return src;
// }

// const char* bad_any_cast::what() const noexcept {
//   // if (msg_.empty())
//   {
//     auto src = hami::local_demangle(src_.name());
//     auto dst = hami::local_demangle(dst_.name());
//     preplace(src, "std::", "");
//     preplace(dst, "std::", "");

//     msg_ = "any_cast [dst = " + dst + "] [src = " + src + "] UNMATCH";
//   }

//   return msg_.c_str();
// }
// }  // namespace any_lite
// }  // namespace nonstd