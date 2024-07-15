import json
import numpy as np

def serialize_complex_types(obj):
    if isinstance(obj, (list, dict, set)):
        return json.dumps(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def deserialize_complex_types(obj):
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            return obj
    return obj