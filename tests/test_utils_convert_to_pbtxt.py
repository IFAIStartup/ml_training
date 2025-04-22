import sys
import pytest
from ml_training.utils import dict_to_pbtxt
from ml_training.utils import DataType

def test_empty():
    res_dict = dict_to_pbtxt({})
    res_list = dict_to_pbtxt([])
    res_str = dict_to_pbtxt("")
    
    assert res_dict == ''
    assert res_list == '[  ]'
    assert res_str == '""'


def test_all_types():
    res = dict_to_pbtxt(
        {
            "name": "run_name", 
            "platform": "onnx", 
            "input": {
                "name": "images", 
                "data_type": DataType.TYPE_FP32, 
                "dims": [1, 2, 3], 
                "reshape": {"shape":[-1]}
            }
        }
    )

    assert res.split('\n') == [
        "name: \"run_name\"",
        "platform: \"onnx\"",
        "input: {",
        "    name: \"images\"",
        "    data_type: TYPE_FP32",
        "    dims: [ 1, 2, 3 ]",
        "    reshape: {",
        "        shape: [ -1 ]",
        "    }",
        "}"
    ]


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))


