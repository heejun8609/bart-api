import bentoml
from transformers import AutoTokenizer, BartForConditionalGeneration
from typing import TYPE_CHECKING
import time
from inference_metrics import inference_duration

if TYPE_CHECKING:
    from bentoml._internal.runner.runner import RunnerMethod

    class RunnerImpl(bentoml.Runner):
        predict: RunnerMethod


class SummarizerRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu","cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self, model_path, max_length, num_beams, length_penalty):
        self.model = BartForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_length = max_length
        self.num_beams = num_beams
        self.length_penalty = length_penalty

    @bentoml.Runnable.method(batchable=False)
    def predict(self, input_data):
        start = time.perf_counter()
        self.model.eval()
        inputs = self.tokenizer(input_data, return_tensors="pt")
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            num_beams=self.num_beams,
            length_penalty=self.length_penalty,
            max_length=self.max_length,
            use_cache=True,
        )
        inference_duration.labels(
            summary_model=self.model.name_or_path.split('/')[-1], summary_cls=self.model.__class__.__name__
        ).observe(time.perf_counter() - start)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)