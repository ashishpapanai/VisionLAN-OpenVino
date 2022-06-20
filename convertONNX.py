from demo import load_network, Train_or_Eval
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms

def convert():
    transf = transforms.ToTensor()
    model = load_network()
    img_width = 256
    img_height = 64
    img = Image.open('./demo/1.png').convert('RGB')
    img = img.resize((img_width, img_height))
    img = transf(img)
    img = torch.unsqueeze(img,dim = 0)
    img.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    torch.onnx.export(model.module, 
                        img, 
                        'VisionLAN.onnx', 
                        export_params=True, 
                        opset_version=10,
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['output', 'out_length'],
                        verbose = True)


if __name__ == '__main__':
    convert()