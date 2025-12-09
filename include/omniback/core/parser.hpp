#ifndef __PARSER_HPP__
#define __PARSER_HPP__
#include <unordered_set>

#include <unordered_map>
#include <vector>

#include <stack>
#include <stdexcept>
#include <vector>
#include "omniback/core/dict.hpp"
#include "omniback/core/string.hpp"
namespace omniback::parser {
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

  const std::unordered_set<std::string>& get_roots() {
    return roots_;
  }

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

void update(
    const std::unordered_map<std::string, std::string>& config,
    std::unordered_map<std::string, std::string>& str_kwargs);

std::
    pair<std::vector<std::string>, std::unordered_map<std::string, std::string>>
    parse_args_kwargs(std::string config);
} // namespace omniback::parser

namespace omniback::parser_v2 {

struct BracketPair {
  char left;
  char right;

  BracketPair(char l, char r) : left(l), right(r) {
    if (l == r) {
      throw std::invalid_argument(
          "Bracket pairs cannot have identical left/right characters");
    }
  }
};

/**
 * Determines if delimiters exist outside all bracket nesting levels.
 *
 * @param input The string to analyze
 * @param bracket_pairs Supported bracket pairs (default: (){}[]<>)
 * @param delimiters Valid delimiter characters (default: {',', ';'})
 * @return true if valid delimiters exist outside brackets, false otherwise
 * @throws std::invalid_argument for mismatched brackets or invalid delimiter
 * positions
 */
bool has_valid_unnested_delimiters(
    const std::string& input,
    const std::vector<BracketPair>& bracket_pairs =
        {{'(', ')'}, {'{', '}'}, {'[', ']'}},
    const std::unordered_set<char>& delimiters = {',', ';'});

bool is_delimiter_separable(
    const std::string& input,
    const std::unordered_map<char, char>& left_to_right =
        {{'(', ')'}, {'{', '}'}, {'[', ']'}},
    const std::unordered_set<char>& delimiters = {',', ';'});

/**
 * @brief Parses the input string and extracts specific patterns based on
 * bracket pairs.
 *
 * @param input The input string to be parsed.
 * @param open_to_close_bracket A map of opening brackets to their corresponding
 * closing brackets.
 * @return std::vector<std::pair<std::string, char>> A vector of pairs
 * containing the extracted strings and corresponding characters.
 *
 * Example usage:
 * - Input: "a[b(d)]"
 *   Output: [ {"a", ''}, {"b(d)", '['} ]
 * - Input: "a(x)[() b[]]"
 *   Output: [ {"a", ''}, {"x", '('}, {"()b[]", '['} ]
 * - Input: "a[b[x()]]"
 *   Output: [ {"a", ''}, {"b[x()]", '['} ]
 * - Input: "a"
 *   Output: [ {"a", ''} ]
 * - Input: "A[(xx,de3)B]"
 *   Output: [ {"A", '(xx,de3)B'} ]
 */
std::vector<std::pair<std::string, char>> expend_outmost_brackets(
    const std::string& input,
    std::unordered_map<char, char> open_to_close_bracket = {
        {'(', ')'},
        {'{', '}'},
        {'[', ']'}});

/**
 * @brief Recursively extracts nested bracket contents while preserving
 * delimiter-separable blocks
 *
 * Processes a bracket-enclosed string by:
 * 1. Removing outer brackets layer by layer
 * 2. Preserving inner blocks that contain delimiter-separable content
 * 3. Maintaining proper nesting order through recursive decomposition
 *
 * @param strtem_in Input string to process (assumed bracket-balanced and
 * space-cleaned)
 * @return std::vector<std::string> Ordered list of decomposed content layers
 *
 * @throws std::invalid_argument For:
 * - Exceeding maximum decomposition depth (10000 layers)
 * - Structural inconsistencies in bracket pairs
 *
 * @par Operational Semantics:
 * - Given "a[b[c[d]]]", returns ["a", "b", "c", "d"]
 * - Given "[[[]]]", returns ["", "", ""]
 * - Given "a(e,r,([zz]))[b,c{d,e}]", returns ["a(e,r,([zz]))", "b,c{d,e}]"]
 * - Given "[[[]]]x", throw invalid_argument as there is an extra 'x'
 *
 * @note Preconditions:
 * - Input must be pre-validated with are_brackets_balanced() and should not be
 * delimiter separable.
 * - remove_space_and_ctrl() must be applied to input
 * - Bracket characters must be non-identical
 */
std::vector<std::pair<std::string, char>> flatten_brackets(
    const std::string& strtem_in);

class Parser {
 public:
  explicit Parser(
      std::vector<BracketPair> bracket_pairs =
          {{'(', ')'}, {'[', ']'}, {'{', '}'}, {'<', '>'}},
      std::unordered_set<char> delimiters = {',', ';'})
      : bracket_pairs_(std::move(bracket_pairs)),
        delimiters_(std::move(delimiters)) {
    build_bracket_maps();
  }

  std::string parse(
      const std::string& input,
      std::unordered_map<std::string, std::string>& config_output);
  static std::vector<std::string> split_by_delimiter(
      const std::string& input,
      char delimiter = ',');
  std::pair<std::vector<char>, std::vector<std::string>> split_by_delimiters(
      const std::string& input,
      char delimiter,
      char delimiter_outter);
  static std::pair<std::string, std::string> prifix_split(
      const std::string& input,
      char left_bracket = '(',
      char right_bracket = ')');

  static std::pair<
      std::vector<std::string>,
      std::unordered_map<std::string, std::string>>
  parse_args_kwargs(std::string config) {
    return omniback::parser::parse_args_kwargs(config);
  }

 private:
  std::vector<BracketPair> bracket_pairs_;
  std::unordered_set<char> delimiters_;
  std::unordered_map<char, char> left_to_right_;
  std::unordered_map<char, char> right_to_left_;
  std::unordered_set<char> left_brackets_;
  std::unordered_set<char> right_brackets_;
  std::unordered_set<std::string> seen_keys_;

  void build_bracket_maps() {
    left_to_right_.clear();
    right_to_left_.clear();
    left_brackets_.clear();
    right_brackets_.clear();

    for (const auto& pair : bracket_pairs_) {
      if (left_to_right_.count(pair.left) || right_to_left_.count(pair.right)) {
        throw std::invalid_argument("Duplicate bracket pair definition");
      }
      left_to_right_[pair.left] = pair.right;
      right_to_left_[pair.right] = pair.left;
      left_brackets_.insert(pair.left);
      right_brackets_.insert(pair.right);
    }
  }
};

} // namespace omniback::parser_v2
#endif