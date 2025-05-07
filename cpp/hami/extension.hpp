#ifndef __HAMI_EXTENSION_HPP__
#define __HAMI_EXTENSION_HPP__

#include "hami/helper/macro.h"
#include "hami/helper/base_logging.hpp"
#include "hami/helper/filesystem.hpp"
#include "hami/helper/range.hpp"
#include "hami/helper/timer.hpp"

#include "hami/core/backend.hpp"
#include "hami/core/helper.hpp"
#include "hami/core/task_keys.hpp"
#include "hami/core/string.hpp"
#include "hami/core/event.hpp"

#include "hami/core/parser.hpp"
#include "hami/builtin/basic_backends.hpp"
#include "hami/builtin/proxy.hpp"
#include "hami/builtin/generate_backend.hpp"

using hami::TASK_DATA_KEY;
using hami::TASK_DEFAULT_NODE_NAME_KEY;
using hami::TASK_ENTRY_KEY;
using hami::TASK_GLOBAL_KEY;
using hami::TASK_INDEX_KEY;
using hami::TASK_RESTART_KEY;
using hami::TASK_RESULT_KEY;
using hami::TASK_STREAM_KEY;
using hami::TASK_TMP_KEY;

// using dict = hami::dict;
// // using Backend = hami::Backend;

#endif