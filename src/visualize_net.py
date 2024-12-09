import utility
from model import *
from build_dataset import *
from preprocess_data import *
from torchsummary import summary
from torchviz import make_dot

if __name__ == "__main__":
    # setting device
    if torch.cuda.is_available():
        my_device = torch.device('cuda')
    else:
        my_device = torch.device('cpu')
    print('> Device : {}'.format(my_device))

    soul_net = PointWiseModel(device=my_device)
    summary(soul_net, input_size=(15, 3000))

    x = torch.randn(1, 15,3000)
    y = soul_net(x)
    make_dot(y.mean(), params=dict(soul_net.named_parameters())).render("SOUL_net_architecture", format="png")
    print("end")