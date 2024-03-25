
#ifndef LITE_AI_ORT_ORT_CORE_H
#define LITE_AI_ORT_ORT_CORE_H

#include "ort_config.h"
#include "ort_handler.h"
#include "ort_types.h"

namespace ortcv {

	class LITE_EXPORTS SCRFD;        

}

namespace ortnlp
{
	class LITE_EXPORTS TextCNN; 
}

namespace ortcv
{
	using core::BasicOrtHandler;
	using core::BasicMultiOrtHandler;
}
namespace ortnlp
{
	using core::BasicOrtHandler;
	using core::BasicMultiOrtHandler;
}
namespace ortasr
{
	using core::BasicOrtHandler;
	using core::BasicMultiOrtHandler;
}
#endif //LITE_AI_ORT_ORT_CORE_H