//
// Copyright (c) 2016-2018 Martin Moene
//
// https://github.com/martinmoene/any-lite
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// modified from https://github.com/martinmoene/any-lite/blob/master/include/nonstd/any.hpp by
// torchpipe team

#pragma once

#ifndef NONSTD_ANY_LITE_HPP
#define NONSTD_ANY_LITE_HPP

#include <string>
#include <type_traits>

#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <stdexcept>
#include "hami/core/string.hpp"
#include "hami/core/reflect.h"
// from
// https://stackoverflow.com/questions/9407367/determine-if-a-type-is-an-stl-container-at-compile-time/31105859#31105859
template <typename T>
struct is_vector_impl : std::false_type {};

template <typename... Args>
struct is_vector_impl<std::vector<Args...>> : std::true_type {};

template <typename T>
struct is_vector {
  static constexpr bool const value = is_vector_impl<std::decay_t<T>>::value;
};

template <typename T>
struct is_unordered_set_impl : std::false_type {};

template <typename... Args>
struct is_unordered_set_impl<std::unordered_set<Args...>> : std::true_type {};
template <typename T>
struct is_unordered_set {
  static constexpr bool const value = is_unordered_set_impl<std::decay_t<T>>::value;
};

template <class T>
using str_dict = std::unordered_map<hami::string, T>;

template <typename T>
struct is_str_dict_impl : std::false_type {};

template <typename T>
struct is_str_dict_impl<str_dict<T>> : std::true_type {};
template <typename T>
struct is_str_dict {
  static constexpr bool const value = is_str_dict_impl<std::decay_t<T>>::value;
};

// is_arithmetic

#define any_lite_MAJOR 0
#define any_lite_MINOR 4
#define any_lite_PATCH 0

#define any_lite_VERSION \
  any_STRINGIFY(any_lite_MAJOR) "." any_STRINGIFY(any_lite_MINOR) "." any_STRINGIFY(any_lite_PATCH)

#define any_STRINGIFY(x) any_STRINGIFY_(x)
#define any_STRINGIFY_(x) #x

// any-lite configuration:

#define any_ANY_DEFAULT 0
#define any_ANY_NONSTD 1
#define any_ANY_STD 2

// tweak header support:

#ifdef __has_include
#if __has_include(<nonstd/any.tweak.hpp>)
#include <nonstd/any.tweak.hpp>
#endif
#define any_HAVE_TWEAK_HEADER 1
#else
#define any_HAVE_TWEAK_HEADER 0
// # pragma message("any.hpp: Note: Tweak header not supported.")
#endif

// Control presence of exception handling (try and auto discover):

#ifndef any_CONFIG_NO_EXCEPTIONS
#if defined(_MSC_VER)
#include <cstddef>  // for _HAS_EXCEPTIONS
#endif
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || (_HAS_EXCEPTIONS)
#define any_CONFIG_NO_EXCEPTIONS 0
#else
#define any_CONFIG_NO_EXCEPTIONS 1
#endif
#endif

// C++ language version detection (C++23 is speculative):
// Note: VC14.0/1900 (VS2015) lacks too much from C++14.

#ifndef any_CPLUSPLUS
#if defined(_MSVC_LANG) && !defined(__clang__)
#define any_CPLUSPLUS (_MSC_VER == 1900 ? 201103L : _MSVC_LANG)
#else
#define any_CPLUSPLUS __cplusplus
#endif
#endif

#define any_CPP14_OR_GREATER (any_CPLUSPLUS >= 201402L)
#define any_CPP17_OR_GREATER (any_CPLUSPLUS >= 201703L)
#define any_CPP20_OR_GREATER (any_CPLUSPLUS >= 202002L)
#define any_CPP23_OR_GREATER (any_CPLUSPLUS >= 202300L)

// Use C++17 std::any if available and requested:

#if any_CPP17_OR_GREATER && defined(__has_include)
#if __has_include(<any> )
#define any_HAVE_STD_ANY 1
#else
#define any_HAVE_STD_ANY 0
#endif
#else
#define any_HAVE_STD_ANY 0
#endif

#define any_USES_STD_ANY 0

//
// in_place: code duplicated in any-lite, expected-lite, optional-lite, value-ptr-lite,
// variant-lite:
//

#ifndef nonstd_lite_HAVE_IN_PLACE_TYPES
#define nonstd_lite_HAVE_IN_PLACE_TYPES 1

// C++17 std::in_place in <utility>:

#if any_CPP17_OR_GREATER

#include <utility>

namespace nonstd {

using std::in_place;
using std::in_place_index;
using std::in_place_index_t;
using std::in_place_t;
using std::in_place_type;
using std::in_place_type_t;

#define nonstd_lite_in_place_t(T) std::in_place_t
#define nonstd_lite_in_place_type_t(T) std::in_place_type_t<T>
#define nonstd_lite_in_place_index_t(K) std::in_place_index_t<K>

#define nonstd_lite_in_place(T) \
  std::in_place_t {}
#define nonstd_lite_in_place_type(T) \
  std::in_place_type_t<T> {}
#define nonstd_lite_in_place_index(K) \
  std::in_place_index_t<K> {}

}  // namespace nonstd

#else  // any_CPP17_OR_GREATER

#include <cstddef>

namespace nonstd {

namespace detail {
template <class T>
struct in_place_type_tag {};

template <std::size_t K>
struct in_place_index_tag {};

}  // namespace detail

struct in_place_t {};

template <class T>
inline in_place_t in_place(detail::in_place_type_tag<T> = detail::in_place_type_tag<T>()) {
  return in_place_t();
}

template <std::size_t K>
inline in_place_t in_place(detail::in_place_index_tag<K> = detail::in_place_index_tag<K>()) {
  return in_place_t();
}

template <class T>
inline in_place_t in_place_type(detail::in_place_type_tag<T> = detail::in_place_type_tag<T>()) {
  return in_place_t();
}

template <std::size_t K>
inline in_place_t in_place_index(detail::in_place_index_tag<K> = detail::in_place_index_tag<K>()) {
  return in_place_t();
}

// mimic templated typedef:

#define nonstd_lite_in_place_t(T) nonstd::in_place_t (&)(nonstd::detail::in_place_type_tag<T>)
#define nonstd_lite_in_place_type_t(T) nonstd::in_place_t (&)(nonstd::detail::in_place_type_tag<T>)
#define nonstd_lite_in_place_index_t(K) \
  nonstd::in_place_t (&)(nonstd::detail::in_place_index_tag<K>)

#define nonstd_lite_in_place(T) nonstd::in_place_type<T>
#define nonstd_lite_in_place_type(T) nonstd::in_place_type<T>
#define nonstd_lite_in_place_index(K) nonstd::in_place_index<K>

}  // namespace nonstd

#endif  // any_CPP17_OR_GREATER
#endif  // nonstd_lite_HAVE_IN_PLACE_TYPES

#include <utility>

// Compiler versions:
//
// MSVC++  6.0  _MSC_VER == 1200  any_COMPILER_MSVC_VERSION ==  60  (Visual Studio 6.0)
// MSVC++  7.0  _MSC_VER == 1300  any_COMPILER_MSVC_VERSION ==  70  (Visual Studio .NET 2002)
// MSVC++  7.1  _MSC_VER == 1310  any_COMPILER_MSVC_VERSION ==  71  (Visual Studio .NET 2003)
// MSVC++  8.0  _MSC_VER == 1400  any_COMPILER_MSVC_VERSION ==  80  (Visual Studio 2005)
// MSVC++  9.0  _MSC_VER == 1500  any_COMPILER_MSVC_VERSION ==  90  (Visual Studio 2008)
// MSVC++ 10.0  _MSC_VER == 1600  any_COMPILER_MSVC_VERSION == 100  (Visual Studio 2010)
// MSVC++ 11.0  _MSC_VER == 1700  any_COMPILER_MSVC_VERSION == 110  (Visual Studio 2012)
// MSVC++ 12.0  _MSC_VER == 1800  any_COMPILER_MSVC_VERSION == 120  (Visual Studio 2013)
// MSVC++ 14.0  _MSC_VER == 1900  any_COMPILER_MSVC_VERSION == 140  (Visual Studio 2015)
// MSVC++ 14.1  _MSC_VER >= 1910  any_COMPILER_MSVC_VERSION == 141  (Visual Studio 2017)
// MSVC++ 14.2  _MSC_VER >= 1920  any_COMPILER_MSVC_VERSION == 142  (Visual Studio 2019)

#if defined(_MSC_VER) && !defined(__clang__)
#define any_COMPILER_MSVC_VER (_MSC_VER)
#define any_COMPILER_MSVC_VERSION (_MSC_VER / 10 - 10 * (5 + (_MSC_VER < 1900)))
#else
#define any_COMPILER_MSVC_VER 0
#define any_COMPILER_MSVC_VERSION 0
#endif

#define any_COMPILER_VERSION(major, minor, patch) (10 * (10 * (major) + (minor)) + (patch))

#if defined(__clang__)
#define any_COMPILER_CLANG_VERSION \
  any_COMPILER_VERSION(__clang_major__, __clang_minor__, __clang_patchlevel__)
#else
#define any_COMPILER_CLANG_VERSION 0
#endif

#if defined(__GNUC__) && !defined(__clang__)
#define any_COMPILER_GNUC_VERSION \
  any_COMPILER_VERSION(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)
#else
#define any_COMPILER_GNUC_VERSION 0
#endif

// half-open range [lo..hi):
// #define any_BETWEEN( v, lo, hi ) ( (lo) <= (v) && (v) < (hi) )

// Presence of language and library features:

#define any_HAVE(feature) (any_HAVE_##feature)

#define any_CPP14_000 (any_CPP14_OR_GREATER)
#define any_CPP17_000 (any_CPP17_OR_GREATER)

// Presence of C++11 language features:

#define any_HAVE_TR1_REMOVE_REFERENCE (!!any_COMPILER_GNUC_VERSION)

// Presence of C++14 language features:

#define any_HAVE_CONSTEXPR_14 any_CPP14_000

// Presence of C++17 language features:

#define any_HAVE_NODISCARD any_CPP17_000

// Presence of C++ language features:

#if any_HAVE_NODISCARD
#define any_nodiscard [[nodiscard]]
#else
#define any_nodiscard /*[[nodiscard]]*/
#endif

// additional includes:

#if any_CONFIG_NO_EXCEPTIONS
#include <cassert>
#else
#include <typeinfo>
#endif

#include <initializer_list>

#include <type_traits>

// Method enabling

#define any_REQUIRES_0(...) \
  template <bool B = (__VA_ARGS__), typename std::enable_if<B, int>::type = 0>

#define any_REQUIRES_T(...) , typename std::enable_if<(__VA_ARGS__), int>::type = 0

#define any_REQUIRES_R(R, ...) typename std::enable_if<__VA_ARGS__, R>::type

#define any_REQUIRES_A(...) , typename std::enable_if<__VA_ARGS__, void*>::type = nullptr

// template <typename T>
// struct inside_type<std::vector<T>> {
//   using type = T;
// };

template <typename T>
struct inner_type {
  using type = T;
};

template <typename T>
struct inner_type<std::vector<T>> {
  // Had to change this line
  using type = typename inner_type<T>::type;
};

//
// any:
//

namespace nonstd {
namespace any_lite {
std::string get_type_name(const std::type_info& info);

namespace detail {

// for any_REQUIRES_T

/*enum*/ class enabler {};

}  // namespace detail

#if !any_CONFIG_NO_EXCEPTIONS

class bad_any_cast : public std::bad_cast {
 public:
  // bad_any_cast() = default;
  bad_any_cast(const std::type_info& src, const std::type_info& dst) : src_(src), dst_(dst) {}
  virtual const char* what() const noexcept override;

 private:
  const std::type_info& src_;
  const std::type_info& dst_;
  mutable std::string msg_;
};

#endif  // any_CONFIG_NO_EXCEPTIONS

struct UnknownContainerTag {};
// #define TYPEID(x)
class TypeInfo {
 public:
  TypeInfo() = delete;
  TypeInfo(const std::type_info& info) : type_(&info) {}

  bool operator==(const std::type_info& info) const noexcept {
    if (*type_ == info) return true;
    if (info == typeid(void) || *type_ == typeid(void)) return false;
    return *type_ == typeid(UnknownContainerTag);
  }
  bool operator!=(const std::type_info& info) const noexcept { return !operator==(info); }
  bool operator==(const TypeInfo& info) const noexcept {
    if (info.type() == *type_) return true;
    if (info.type() == typeid(void) || *type_ == typeid(void)) return false;
    return (typeid(UnknownContainerTag) == *type_) || (typeid(UnknownContainerTag) == info.type());
  };
  bool operator!=(const TypeInfo& info) const noexcept { return !operator==(info); };

  const char* name() const noexcept { return type_->name(); }
  const std::type_info& type() const noexcept { return *type_; }
  // operator const std::type_info&() const noexcept { return *type_; }
  std::size_t hash_code() const noexcept { return type_->hash_code(); }

 private:
  const std::type_info* type_;
};

static inline bool operator==(const std::type_info& info, const TypeInfo& infoT) {
  return infoT == info;
}
static inline bool operator!=(const std::type_info& info, const TypeInfo& infoT) {
  return infoT != info;
}

enum class PyClassType {
  list,
  set,
  str_dict,
  set_of_arithmetic,
  str_dict_of_arithmetic,
  arithmetic,
  list_of_arithmetic,
  list2_of_arithmetic,
  other,
  unknown_container
};
// template <class T>
// PyClassType get_class_type() {
//   return PyClassType::other;
// }

class any {
 public:
  constexpr any() noexcept : content(nullptr) {}
  any(const char*) = delete;
  any(const unsigned char*) = delete;
  any(UnknownContainerTag&&) noexcept { content = new UnknownEmptyHolder(); };

  any(any const& other) : content(other.content ? other.content->clone() : nullptr) {}

  any(any&& other) noexcept : content(std::move(other.content)) { other.content = nullptr; }

  template <class ValueType,
            class T = typename std::decay<ValueType>::type any_REQUIRES_T(
                !std::is_same<T, any>::value && !std::is_same<T, UnknownContainerTag>::value)>
  any(ValueType&& value) noexcept : content(new holder<T>(std::forward<ValueType>(value))) {}

  template <class ValueType,
            class T = typename std::decay<ValueType>::type any_REQUIRES_T(
                !std::is_same<T, any>::value && !std::is_same<T, UnknownContainerTag>::value)>
  void reset_holder(ValueType&& value) noexcept {
    if (content) delete content;
    content = (new holder<T>(std::forward<ValueType>(value)));
  }

  template <class T, class... Args any_REQUIRES_T(std::is_constructible<T, Args&&...>::value)>
  explicit any(nonstd_lite_in_place_type_t(T), Args&&... args)
      : content(new holder<T>(T(std::forward<Args>(args)...))) {}

  template <class T, class U,
            class... Args any_REQUIRES_T(
                std::is_constructible<T, std::initializer_list<U>&, Args&&...>::value)>
  explicit any(nonstd_lite_in_place_type_t(T), std::initializer_list<U> il, Args&&... args)
      : content(new holder<T>(T(il, std::forward<Args>(args)...))) {}

  ~any() { reset(); }

  any& operator=(any const& other) {
    any(other).swap(*this);
    return *this;
  }

  any& operator=(any&& other) noexcept {
    any(std::move(other)).swap(*this);
    return *this;
  }

  template <class ValueType, class T = typename std::decay<ValueType>::type any_REQUIRES_T(
                                 !std::is_same<T, any>::value)>
  any& operator=(T&& value) {
    any(std::move(value)).swap(*this);
    return *this;
  }

  template <class T, class... Args>
  void emplace(Args&&... args) {
    any(T(std::forward<Args>(args)...)).swap(*this);
  }

  template <class T, class U,
            class... Args any_REQUIRES_T(
                std::is_constructible<T, std::initializer_list<U>&, Args&&...>::value)>
  void emplace(std::initializer_list<U> il, Args&&... args) {
    any(T(il, std::forward<Args>(args)...)).swap(*this);
  }

  void reset() noexcept {
    delete content;
    content = nullptr;
  }

  void swap(any& other) noexcept { std::swap(content, other.content); }

  bool has_value() const noexcept { return content != nullptr; }

  const TypeInfo type() const noexcept {
    return has_value() ? content->type() : TypeInfo(typeid(void));
  }

  const std::type_info& inner_type() const {
    return has_value() ? content->inner_type() : (typeid(void));
  }

  const std::size_t size() const noexcept { return has_value() ? content->size() : 0; }

  any at(std::size_t i) {
    if (has_value())
      return content->at(i);
    else {
      throw std::runtime_error("any at(std::size_t i) failed");
    };
  }
  std::vector<PyClassType> get_class_type() const noexcept {
    return has_value() ? content->get_class_type() : std::vector<PyClassType>();
  }

  void* get_ptr() { return content ? content->get_ptr() : nullptr; }
  const void* get_ptr() const { return content ? content->get_ptr() : nullptr; }

 private:
  class placeholder {
   public:
    virtual ~placeholder() {}

    virtual TypeInfo const type() const = 0;

    virtual placeholder* clone() const = 0;

    virtual std::vector<PyClassType> get_class_type() const = 0;

    virtual std::size_t size() const { return 0; }
    virtual any at(std::size_t i) = 0;
    virtual const void* get_ptr() const = 0;
    virtual void* get_ptr() = 0;
    virtual const std::type_info& inner_type() const = 0;

   protected:
  };

  template <typename ValueType>
  class holder : public placeholder {
   public:
    holder(ValueType const& value) : held(value) {}

    holder(ValueType&& value) : held(std::move(value)) {}

    virtual TypeInfo const type() const override { return typeid(ValueType); }

    virtual placeholder* clone() const override { return new holder(held); }

    virtual any at(std::size_t i) { return at_impl<ValueType>(i); };
    virtual std::size_t size() const { return size_impl<ValueType>(); };

    template <class T, typename std::enable_if<((is_vector<T>::value)), int>::type = 0>
    any at_impl(std::size_t i) {
      return held.at(i);
    }

    std::vector<PyClassType> get_class_type() const override {
      std::vector<PyClassType> final_type;
      get_class_type<ValueType>(final_type);
      return final_type;
    }

    template <class T, typename std::enable_if<((is_vector<T>::value)), int>::type = 0>
    void get_class_type(std::vector<PyClassType>& final_type) const {
      final_type.push_back(PyClassType::list);
      get_class_type<typename T::value_type>(final_type);
    }

    template <class T, typename std::enable_if<((is_vector<T>::value)), int>::type = 0>
    const std::type_info& inner_type() const {
      return typeid(typename T::value_type);
    }

    template <class T, typename std::enable_if<((is_str_dict<T>::value)), int>::type = 0>
    void get_class_type(std::vector<PyClassType>& final_type) const {
      final_type.push_back(PyClassType::str_dict);
      get_class_type<typename T::mapped_type>(final_type);
    }

    template <class T, typename std::enable_if<((is_str_dict<T>::value)), int>::type = 0>
    const std::type_info& inner_type() const {
      return typeid(typename T::mapped_type);
    }

    template <class T, typename std::enable_if<((is_unordered_set<T>::value)), int>::type = 0>
    void get_class_type(std::vector<PyClassType>& final_type) const {
      final_type.push_back(PyClassType::set);
      get_class_type<typename T::value_type>(final_type);
    }

    template <class T, typename std::enable_if<((is_unordered_set<T>::value)), int>::type = 0>
    const std::type_info& inner_type() const {
      return typeid(typename T::value_type);
    }

    template <class T, typename std::enable_if<((std::is_arithmetic<T>::value)), int>::type = 0>
    void get_class_type(std::vector<PyClassType>& final_type) const {
      final_type.push_back(PyClassType::arithmetic);
    }

    template <class T,
              typename std::enable_if<(!(std::is_arithmetic<T>::value || is_vector<T>::value ||
                                         is_str_dict<T>::value || is_unordered_set<T>::value)),
                                      int>::type = 0>
    void get_class_type(std::vector<PyClassType>& final_type) const {
      final_type.push_back(PyClassType::other);
    }

    template <class T, typename std::enable_if<(!(is_vector<T>::value || is_str_dict<T>::value ||
                                                  is_unordered_set<T>::value)),
                                               int>::type = 0>
    const std::type_info& inner_type() const {
      return typeid(void);
    }

    const std::type_info& inner_type() const override { return inner_type<ValueType>(); }

    template <class T, typename std::enable_if<(!(is_vector<T>::value)), int>::type = 0>
    any at_impl(std::size_t i) {
      throw std::runtime_error("any: no `at` method. The type is " + get_type_name(typeid(T)));
      return any();
    }

    template <class T, typename std::enable_if<(is_vector<T>::value || is_unordered_set<T>::value ||
                                                is_str_dict<T>::value),
                                               int>::type = 0>
    std::size_t size_impl() const {
      return held.size();
    }

    template <class T,
              typename std::enable_if<(!(is_vector<T>::value || is_unordered_set<T>::value ||
                                         is_str_dict<T>::value)),
                                      int>::type = 0>
    std::size_t size_impl() const {
      throw std::runtime_error("any: no `size` method. The type is " + get_type_name(typeid(T)));
      // return -1;
    }

    void* get_ptr() { return &held; }
    const void* get_ptr() const { return static_cast<const void*>(&held); }

    ValueType held;
    // PyClassType type_class;
  };

  class UnknownEmptyHolder : public placeholder {
   private:
   public:
    UnknownEmptyHolder() : type_info(TypeInfo(typeid(UnknownContainerTag))) {}

    virtual TypeInfo const type() const override { return type_info; }

    virtual placeholder* clone() const override { return new UnknownEmptyHolder(); }

    virtual any at(std::size_t i) { throw std::runtime_error("empty container"); };
    virtual std::size_t size() const { return 0; };
    virtual void* get_ptr() { return nullptr; }
    virtual const void* get_ptr() const { return nullptr; }

    const std::type_info& inner_type() const { return typeid(void); };

    std::vector<PyClassType> get_class_type() const override {
      return {PyClassType::unknown_container};
    }

    TypeInfo type_info;
  };

  placeholder* content;
};

inline void swap(any& x, any& y) noexcept { x.swap(y); }

template <class T, class... Args>
inline any make_any(Args&&... args) {
  return any(nonstd_lite_in_place_type(T), std::forward<Args>(args)...);
}

template <class T, class U, class... Args>
inline any make_any(std::initializer_list<U> il, Args&&... args) {
  return any(nonstd_lite_in_place_type(T), il, std::forward<Args>(args)...);
}

template <class ValueType,
          typename = typename std::enable_if<(std::is_reference<ValueType>::value ||
                                              std::is_copy_constructible<ValueType>::value),
                                             nonstd::any_lite::detail::enabler>::type>
any_nodiscard inline ValueType any_cast(any const& operand) {
  const ValueType* result =
      any_cast<typename std::add_const<typename std::remove_reference<ValueType>::type>::type>(
          &operand);

#if any_CONFIG_NO_EXCEPTIONS
  assert(result);
#else
  if (!result) {
    throw bad_any_cast(operand.type().type(), typeid(ValueType));
  }
#endif

  return *result;
}

template <class ValueType,
          typename = typename std::enable_if<(std::is_reference<ValueType>::value ||
                                              std::is_copy_constructible<ValueType>::value),
                                             nonstd::any_lite::detail::enabler>::type>
any_nodiscard inline ValueType any_cast(any& operand) {
  const ValueType* result = any_cast<typename std::remove_reference<ValueType>::type>(&operand);

#if any_CONFIG_NO_EXCEPTIONS
  assert(result);
#else
  if (!result) {
    throw bad_any_cast(operand.type().type(), typeid(ValueType));
  }
#endif

  return *result;
}

template <class ValueType any_REQUIRES_T(std::is_reference<ValueType>::value ||
                                         std::is_copy_constructible<ValueType>::value)>
any_nodiscard inline ValueType any_cast(any&& operand) {
  const ValueType* result = any_cast<typename std::remove_reference<ValueType>::type>(&operand);

#if any_CONFIG_NO_EXCEPTIONS
  assert(result);
#else
  if (!result) {
    throw bad_any_cast(operand.type().type(), typeid(ValueType));
  }
#endif

  return *result;
}

template <class ValueType>
any_nodiscard inline ValueType const* any_cast(any const* operand) noexcept {
  return operand && operand->get_ptr() && operand->type() == typeid(ValueType)
             ? static_cast<const ValueType*>(operand->get_ptr())
             : nullptr;
}

template <class ValueType>
any_nodiscard inline ValueType* any_cast(any* operand) noexcept {
  if (!operand) return nullptr;
  if (!operand->get_ptr()) {
    operand->reset_holder<ValueType>(ValueType());
  }
  return operand->type() == typeid(ValueType) ? static_cast<ValueType*>(operand->get_ptr())
                                              : nullptr;
}

}  // namespace any_lite

using namespace any_lite;

}  // namespace nonstd

namespace hami {
using namespace nonstd;
namespace core {
using namespace nonstd;
}
}  // namespace hami
#endif  // NONSTD_ANY_LITE_HPP