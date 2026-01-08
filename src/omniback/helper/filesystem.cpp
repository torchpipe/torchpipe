

#include <filesystem>
namespace fs = std::filesystem;

// #include <experimental/filesystem>
// namespace fs = std::experimental::filesystem;

#include <string>

namespace om::filesystem {

bool exists(const std::string& path) {
  return fs::exists(path);
}

} // namespace om::filesystem
