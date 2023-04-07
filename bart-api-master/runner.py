import bentoml
import typing as t
from model_config import twc_config, base_config
from runnable import SummarizerRunnable

twc_model_path = twc_config['model_path']
base_model_path = base_config['model_path']
max_length = base_config['max_length']
num_beams = base_config['num_beams']
base_length_penalty = base_config['length_penalty']
twc_length_penalty = twc_config['length_penalty']

twc_runner = t.cast(
    "RunnerImpl",
    bentoml.Runner(SummarizerRunnable, name="twc_summarization",
                   runnable_init_params={
                       "model_path": twc_model_path,
                       "max_length": max_length,
                       "num_beams": num_beams,
                       "length_penalty": twc_length_penalty
                   })
)

base_runner = t.cast(
    "RunnerImpl",
    bentoml.Runner(SummarizerRunnable, name="base_summarization",
                   runnable_init_params={
                       "model_path": base_model_path,
                       "max_length": max_length,
                       "num_beams": num_beams,
                       "length_penalty": base_length_penalty
                   })
)