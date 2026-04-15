from rest_calculator import WorkerRawInput
from rest_model_engine import RestModelEngine

engine = RestModelEngine(
    model_path="rest_recommendation_model.pkl",
    forced_rest_work_min=120,
)

data = WorkerRawInput(
    worker_id="worker_01",
    hr=128.0,
    temp_c=33.5,
    humid=68.0,
    age=61,
    gender=1,
    height_cm=170.0,
    weight_kg=68.0,
    work_duration_min=95,
    elderly_flag=1,
    heart_disease=0,
    hypertension=1,
    other_disease=0,
    baseline_hr=88.0,
)

result = engine.predict(data)
print(result)