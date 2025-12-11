/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/ffi/container/queue.h
 * \brief FIFO queue backed by ArrayObj container.
 *
 * Queue implements copy-on-write semantics and stores elements as Any,
 * similar to Array<T>. The API is intentionally similar to STL containers
 * but tailored to the tvm-ffi ObjectRef/Any model.
 */

#ifndef TVM_FFI_CONTAINER_QUEUE_H_
#define TVM_FFI_CONTAINER_QUEUE_H_

#include <tvm/ffi/container/array.h>

#include <cstddef>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

namespace tvm {
namespace ffi {

/*!
 * \brief FIFO queue backed by ArrayObj.
 *
 * \tparam T The element type of the queue.
 */
template <typename T>
class Queue : public ObjectRef {
 public:
  static_assert(
      details::all_storage_enabled_v<T>,
      "Type used in Queue<T> must be compatible with Any");
  /*! \brief Default constructor (empty queue) */
  Queue() : ObjectRef(MakeEmptyNode()) {}
  /*! \brief Unsafe init constructor */
  explicit Queue(UnsafeInit tag) : ObjectRef(tag) {}
  /*! \brief Copy constructor */
  Queue(const Queue<T>& other) : ObjectRef(other) {}
  /*! \brief Move constructor */
  Queue(Queue<T>&& other) noexcept : ObjectRef(std::move(other)) {}

  /*!
   * \brief Construct from initializer list
   */
  Queue(std::initializer_list<T> init) : ObjectRef(MakeFromInit(init)) {}

  /*!
   * \brief Assignment from another queue
   */
  TVM_FFI_INLINE Queue& operator=(const Queue<T>& other) {
    data_ = other.data_;
    return *this;
  }

  /*!
   * \brief Assignment from another queue (move)
   */
  TVM_FFI_INLINE Queue& operator=(Queue<T>&& other) noexcept {
    data_ = std::move(other.data_);
    return *this;
  }

  /*! \return number of elements */
  TVM_FFI_INLINE size_t size() const {
    return GetArrayObj()->size();
  }

  /*! \return whether the queue is empty */
  TVM_FFI_INLINE bool empty() const {
    return size() == 0;
  }

  /*!
   * \brief Push an element to the back of the queue.
   */
  template <
      typename U,
      typename = std::enable_if_t<std::is_constructible_v<T, U&&>, int>>
  void push(U&& item) {
    // create a new array with size + 1 and copy existing elements, then append
    // new
    const ArrayObj* old = GetArrayObj();
    size_t old_n = old->size();
    ObjectPtr<ArrayObj> p = ArrayObj::Empty(old_n + 1);
    Any* dst = p->MutableBegin();
    const Any* src = old->begin();
    for (size_t i = 0; i < old_n; ++i) {
      new (dst++) Any(*src++);
      p->size_++;
    }
    new (dst) Any(T(std::forward<U>(item)));
    p->size_++;
    data_ = std::move(p);
  }

  /*!
   * \brief Emplace an element to the back of the queue.
   */
  template <typename... Args>
  void emplace(Args&&... args) {
    const ArrayObj* old = GetArrayObj();
    size_t old_n = old->size();
    ObjectPtr<ArrayObj> p = ArrayObj::Empty(old_n + 1);
    Any* dst = p->MutableBegin();
    const Any* src = old->begin();
    for (size_t i = 0; i < old_n; ++i) {
      new (dst++) Any(*src++);
      p->size_++;
    }
    new (dst) Any(T(std::forward<Args>(args)...));
    p->size_++;
    data_ = std::move(p);
  }

  /*!
   * \brief Access the front element (by value).
   *
   * \note const& version returns a copy from Any.
   */
  TVM_FFI_INLINE T front() const& {
    if (empty()) {
      TVM_FFI_THROW(RuntimeError) << "Queue::front() called on empty queue";
    }
    const Any* ptr = GetArrayObj()->begin();
    return details::AnyUnsafe::CopyFromAnyViewAfterCheck<T>(*ptr);
  }

  /*!
   * \brief Access the front element by move if the queue is unique (rvalue).
   *
   * If not unique, fallback to copy via const& version.
   */
  TVM_FFI_INLINE T front() && {
    if (!this->unique()) {
      return std::as_const(*this).front();
    }
    if (empty()) {
      TVM_FFI_THROW(RuntimeError) << "Queue::front() called on empty queue";
    }
    Any* ptr = GetArrayObj()->MutableBegin();
    return details::AnyUnsafe::MoveFromAnyAfterCheck<T>(std::move(*ptr));
  }

  /*!
   * \brief Remove the front element.
   *
   * This operation performs copy-on-write by creating a new ArrayObj
   * with remaining elements.
   */
  void pop() {
    if (empty()) {
      TVM_FFI_THROW(RuntimeError) << "Queue::pop() called on empty queue";
    }
    const ArrayObj* old = GetArrayObj();
    size_t old_n = old->size();
    if (old_n == 1) {
      data_ = ArrayObj::Empty(0);
      return;
    }
    ObjectPtr<ArrayObj> p = ArrayObj::Empty(old_n - 1);
    Any* dst = p->MutableBegin();
    const Any* src = old->begin() + 1; // copy elements 1..end
    for (size_t i = 0; i < old_n - 1; ++i) {
      new (dst++) Any(*src++);
      p->size_++;
    }
    data_ = std::move(p);
  }

  /*!
   * \brief Try to pop the front element and return it.
   * \return optional containing the front element if non-empty.
   *
   * This uses copy semantics for extraction.
   */
  std::optional<T> try_pop() {
    if (empty())
      return std::nullopt;
    const Any* ptr = GetArrayObj()->begin();
    T v = details::AnyUnsafe::CopyFromAnyViewAfterCheck<T>(*ptr);
    pop();
    return std::optional<T>(std::move(v));
  }

  /*! \brief specify container node */
  using ContainerType = ArrayObj;

 private:
  static ObjectPtr<ArrayObj> MakeEmptyNode() {
    return ArrayObj::Empty(0);
  }

  static ObjectPtr<ArrayObj> MakeFromInit(std::initializer_list<T> init) {
    ObjectPtr<ArrayObj> p = ArrayObj::Empty(init.size());
    Any* itr = p->MutableBegin();
    for (const T& v : init) {
      new (itr++) Any(T(v));
      p->size_++;
    }
    return p;
  }

  /*! \return The underlying ArrayObj */
  ArrayObj* GetArrayObj() const {
    return static_cast<ArrayObj*>(data_.get());
  }

  template <typename U>
  friend class Queue;
};

template <typename T>
inline constexpr bool use_default_type_traits_v<Queue<T>> = false;

template <typename T>
struct TypeTraits<Queue<T>> : public ObjectRefTypeTraitsBase<Queue<T>> {
  using ObjectRefTypeTraitsBase<Queue<T>>::CopyFromAnyViewAfterCheck;

  TVM_FFI_INLINE static std::string GetMismatchTypeInfo(const TVMFFIAny* src) {
    if (src->type_index != TypeIndex::kTVMFFIArray) {
      return TypeTraitsBase::GetMismatchTypeInfo(src);
    }
    const ArrayObj* n = reinterpret_cast<const ArrayObj*>(src->v_obj);
    // For queue we accept any length, report array length only if needed for
    // mismatch messaging.
    return GetMismatchTypeInfoHelper<0>(n->begin(), n->size());
  }

  template <size_t I = 0>
  TVM_FFI_INLINE static std::string GetMismatchTypeInfoHelper(
      const Any* arr,
      size_t len) {
    for (size_t i = 0; i < len; ++i) {
      if constexpr (!std::is_same_v<T, Any>) {
        const Any& any_v = arr[i];
        if (!details::AnyUnsafe::CheckAnyStrict<T>(any_v) &&
            !(any_v.try_cast<T>().has_value())) {
          return "Array[index " + std::to_string(i) + ": " +
              details::AnyUnsafe::GetMismatchTypeInfo<T>(any_v) + "]";
        }
      }
    }
    // no mismatch
    TVM_FFI_THROW(InternalError) << "Cannot reach here";
    TVM_FFI_UNREACHABLE();
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    if (src->type_index != TypeIndex::kTVMFFIArray)
      return false;
    const ArrayObj* n = reinterpret_cast<const ArrayObj*>(src->v_obj);
    const TVMFFIAny* ffi_any_arr =
        reinterpret_cast<const TVMFFIAny*>(n->begin());
    for (size_t i = 0; i < n->size(); ++i) {
      if constexpr (!std::is_same_v<T, Any>) {
        if (!TypeTraits<T>::CheckAnyStrict(ffi_any_arr + i))
          return false;
      }
    }
    return true;
  }

  TVM_FFI_INLINE static std::optional<Queue<T>> TryCastFromAnyView(
      const TVMFFIAny* src) {
    if (src->type_index != TypeIndex::kTVMFFIArray)
      return std::nullopt;
    const ArrayObj* n = reinterpret_cast<const ArrayObj*>(src->v_obj);
    // fast path: elements already strict
    if (CheckAnyStrict(src)) {
      return CopyFromAnyViewAfterCheck(src);
    }
    // slow path: try to convert elements to T
    Array<Any> arr = TypeTraits<Array<Any>>::CopyFromAnyViewAfterCheck(src);
    Any* ptr = arr.CopyOnWrite()->MutableBegin();
    for (size_t i = 0; i < n->size(); ++i) {
      if constexpr (!std::is_same_v<T, Any>) {
        if (auto opt_convert = ptr[i].try_cast<T>()) {
          ptr[i] = *std::move(opt_convert);
        } else {
          return std::nullopt;
        }
      }
    }
    return details::ObjectUnsafe::ObjectRefFromObjectPtr<Queue<T>>(
        details::ObjectUnsafe::ObjectPtrFromObjectRef<Object>(arr));
  }

  TVM_FFI_INLINE static std::string TypeStr() {
    return details::ContainerTypeStr<T>("Queue");
  }

  TVM_FFI_INLINE static std::string TypeSchema() {
    std::ostringstream oss;
    oss << R"({"type":"Queue","args":[)" << details::TypeSchema<T>::v() << "]}";
    return oss.str();
  }
};

namespace details {
template <typename T, typename U>
inline constexpr bool type_contains_v<Queue<T>, Queue<U>> =
    type_contains_v<T, U>;
} // namespace details

} // namespace ffi
} // namespace tvm

namespace std {

template <typename T>
struct tuple_size<::tvm::ffi::Queue<T>>
    : public std::integral_constant<size_t, 0> {};

} // namespace std

#endif // TVM_FFI_CONTAINER_QUEUE_H_