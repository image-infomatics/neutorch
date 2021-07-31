# prints currently alive Tensors and Variables
import torch
import gc


def print_tensors():
    print('===========================TENSORS===========================')
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(obj.dtype, obj.device, type(obj), obj.size())
        except:
            pass
