

// Feature detection for std::filesystem
#if __has_include(<filesystem>) && __cpp_lib_filesystem >= 201703
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#error "No filesystem library available!"
#endif

namespace hami::filesystem {

bool exists(const std::string& path) { return fs::exists(path); }

}  // namespace hami::filesystem
