

#include <filesystem>
namespace fs = std::filesystem;

// #include <experimental/filesystem>
// namespace fs = std::experimental::filesystem;

#include <string>

namespace omniback::filesystem {

bool exists(const std::string& path) {
  return fs::exists(path);
}

} // namespace omniback::filesystem
