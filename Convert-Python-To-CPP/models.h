#ifndef LITE_AI_MODELS_H
#define LITE_AI_MODELS_H

#include "config.h"


#ifdef ENABLE_ONNXRUNTIME
#include "scrfd.h"
#endif

#ifdef ENABLE_ONNXRUNTIME
typedef ortcv::SCRFD _ONNXSCRFD;
namespace face {
	namespace detect
	{
		typedef _ONNXSCRFD SCRFD;
	}
}
#endif

#endif //LITE_AI_MODELS_H