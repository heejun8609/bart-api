import bentoml
from bentoml.io import JSON, Text
from utils import get_logger
from runner import twc_runner, base_runner
from model import Model
from typing import Dict, Any
from inference_metrics import summary_counter

svc = bentoml.Service("bart-r3f", runners=[twc_runner, base_runner])


@svc.api(input=JSON(pydantic_model=Model), output=JSON())
async def predict(input_data: Model) -> Dict[str, Any]:
    bentoml_logger = get_logger("Summarization")
    data = input_data.dict()
    model_name = data['model_name']
    dialogue = data['dialogue']
    try:
        if model_name == 'base':
            bentoml_logger.info("* Base Summarization *")
            summary = await base_runner.async_run(dialogue[:230])
        elif model_name == 'twc':
            bentoml_logger.info("* TWC Summarization *")
            summary = await twc_runner.async_run(dialogue[:500])
        else:
            summary = "'model_name'을 'base' 또는 'twc'로 넣어주세요"
            bentoml_logger.error(summary)
        summary_counter.labels(summary=summary).inc()
        return {'summary': summary}
    except BaseException as e:
        bentoml_logger.error(e)
        return e


