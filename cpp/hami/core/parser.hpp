#ifndef __PARSER_HPP__
#define __PARSER_HPP__
#include <unordered_set>

#include <vector>
#include <unordered_map>

#include "hami/core/string.hpp"
#include "hami/core/dict.hpp"
namespace hami::parser {
void broadcast_global(str::mapmap& config);
std::unordered_set<std::string> set_node_name(str::mapmap& config);

str::str_map get_global_config(const str::mapmap& config);
size_t count(const str::mapmap& config);
// str::mapmap parse_toml(const std::string& toml_path);

class DagParser {
   public:
    DagParser(const str::mapmap& config);

    /**
     * @brief get the independent subgraph of a node
     *
     * @throw there is no subgraph for the node
     */
    std::unordered_set<std::string> get_subgraph(const std::string& root);
    // const std::unordered_set<std::string>& get_previous(const std::string&
    // node) {
    //   return dag_config_.at(node).previous;
    // }

    bool skip_if_no_result(const std::string& node) {
        return dag_config_.at(node).or_filter;
    }

    dict prepare_data_from_previous(
        const std::string& node,
        std::unordered_map<std::string, dict>& processed);

    template <typename T>
    bool is_ready(const std::string& curr_node, const T& processed) {
        for (const auto& node : dag_config_.at(curr_node).previous) {
            if (processed.find(node) == processed.end()) {
                return false;
            }
        }
        return true;
    }

    const std::unordered_set<std::string>& get_roots() { return roots_; }

   private:
    struct Node {
        std::unordered_set<std::string> next;
        std::unordered_set<std::string> previous;
        bool or_filter = false;
        str::mapmap map_config;
        std::unordered_set<std::string> cited;
    };

   private:
    std::vector<std::string> try_topological_sort();
    void update_previous();
    void update_roots();
    void update_cited_from_map();
    std::unordered_map<std::string, Node> dag_config_;
    str::mapmap parse_map_config(const std::string& config);
    std::unordered_set<std::string> roots_;
};

void update(const std::unordered_map<std::string, std::string>& config,
            std::unordered_map<std::string, std::string>& str_kwargs);

std::pair<std::vector<std::string>,
          std::unordered_map<std::string, std::string>>
parse_args_kwargs(std::string config);
}  // namespace hami::parser

#endif