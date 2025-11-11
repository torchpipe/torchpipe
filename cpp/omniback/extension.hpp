#ifndef __OMNI_EXTENSION_HPP__
#define __OMNI_EXTENSION_HPP__

#include "omniback/helper/base_logging.hpp"
#include "omniback/helper/filesystem.hpp"
#include "omniback/helper/macro.h"
#include "omniback/helper/range.hpp"
#include "omniback/helper/timer.hpp"

#include "omniback/core/backend.hpp"
#include "omniback/core/event.hpp"
#include "omniback/core/helper.hpp"
#include "omniback/core/string.hpp"
#include "omniback/core/task_keys.hpp"

#include "omniback/builtin/basic_backends.hpp"
#include "omniback/builtin/generate_backend.hpp"
#include "omniback/builtin/proxy.hpp"
#include "omniback/core/parser.hpp"

using omniback::TASK_DATA_KEY;
using omniback::TASK_DEFAULT_NODE_NAME_KEY;
using omniback::TASK_ENTRY_KEY;
using omniback::TASK_GLOBAL_KEY;
using omniback::TASK_INDEX_KEY;
using omniback::TASK_RESTART_KEY;
using omniback::TASK_RESULT_KEY;
using omniback::TASK_STREAM_KEY;
using omniback::TASK_TMP_KEY;

// using dict = omniback::dict;
// // using Backend = omniback::Backend;

#endif