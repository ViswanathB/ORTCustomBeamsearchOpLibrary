import onnxruntime as ort
import numpy as np

def test_custom_op():
    #TODO find a way to add relative paths
    sessionOptions = ort.SessionOptions()
    sessionOptions.register_custom_ops_library("D:\\ai\\other_repos\\customops\\ORTCustomBeamsearchOpLibrary\\build\\Debug\\custom_beamsearch_op_library_cpu.dll")
    ort_session = ort.InferenceSession("D:\\ai\\other_repos\\customops\\ORTCustomBeamsearchOpLibrary\\test\\beamsearch_op.onnx", sess_options=sessionOptions)

    model_inputs = {}
    model_inputs['input_ids'] = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    model_inputs['num_beams'] = np.array([10], dtype = np.float32)

    print(ort_session.run(None, model_inputs))

test_custom_op()