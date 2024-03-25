// copy from https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/ort/core/ort_defs.h

#ifndef LITE_AI_ORT_CORE_ORT_DEFS_H
#define LITE_AI_ORT_CORE_ORT_DEFS_H

#include "config.h"
#include "lite.ai.defs.h"

#ifdef ENABLE_DEBUG_STRING
# define LITEORT_DEBUG 1
#else
# define LITEORT_DEBUG 0
#endif

#ifdef LITE_WIN32
# define LITEORT_CHAR wchar_t
#else
# define LITEORT_CHAR char
#endif

#ifdef LITE_WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#endif

#ifdef ENABLE_ONNXRUNTIME_CUDA
# define USE_CUDA
#endif

#endif //LITE_AI_ORT_CORE_ORT_DEFS_H