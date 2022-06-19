from demo import load_network, Train_or_Eval
import torch
from torch.autograd import Variable


def convert():
    model = load_network()
    dummy_input = Variable(torch.randn(1, 3, 256, 64))
    dummy_input.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    torch.onnx.export(model.module, 
                        dummy_input, 
                        'VisionLAN.onnx', 
                        export_params=True, 
                        opset_version=10,
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['output', 'out_length'],
                        verbose = True)


if __name__ == '__main__':
    convert()