import bentoml


inference_duration = bentoml.metrics.Histogram(
    name="inference_duration",
    documentation="Duration of inference",
    labelnames=["summary_model", "summary_cls"],
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        7.5,
        10.0,
        float("inf"),
    ),
)

summary_counter = bentoml.metrics.Counter(
    name="summary_total",
    documentation="Count total number of analysis by summary scores",
    labelnames=["summary"],
)