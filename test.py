import torch
from torchsummary import summary
from nets.CSPdarknet import darknet53
from nets.yolo4 import YoloBody


if __name__ == "__main__":
    # 需要使用device来指定网络在GPU还是CPU运行
    if True: 
        model_path = './wheat_yolo.pth'
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        model = YoloBody(3,1).to(device)
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    summary(model, input_size=(3, 416, 416))
