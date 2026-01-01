#pragma once

#include <type_traits> 
#include <cstddef> 
#include <utility>  
#include <cstdint> 

namespace omniback::ffi {

template <typename, typename = void>
struct OmTypeTraits {
  /*! \brief Whether the type is enabled in FFI. */
  static constexpr bool convert_enabled = false;
  // /*! \brief Whether the type can appear as a storage type in Container */
  // static constexpr bool storage_enabled = false;
};

// Helper alias to strip const & reference
template <typename T>
using OmTypeTraitsNoCR =
    OmTypeTraits<std::remove_const_t<std::remove_reference_t<T>>>;

// Control whether to fall back to default/generalized traits
template <typename T>
inline constexpr bool om_use_default_type_traits_v = true;

// Base class for user-defined type traits
struct OmTypeTraitsBase {
  static constexpr bool convert_enabled = true;
  // static constexpr bool storage_enabled = true;
};

} // namespace omniback