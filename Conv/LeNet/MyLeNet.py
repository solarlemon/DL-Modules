import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
net = nn.Sequential(
    # 输入1*1*28*28输出1*6*14*14
    # in_channels=1,out_channels=6,kernel_size,padding
    nn.Conv2d(1, 6, kernel_size=5, padding=2),  # 得到6@14*14
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),  # 大小不会改变
    # 输入1*6*14*14输出1*16*5*5
    nn.Conv2d(6, 16, kernel_size=5),  # 得到16@10*10
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),  # 大小不会改变
    # 输入1*16*5*5输出1*10
    nn.Flatten(),  # 第一维的批量维度保持，后面平铺成一个向量
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))


def evaluate_accuracy_gpu(net, data_iter, device=None):  # @save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device  # 网络层的device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    torch.save(net.state_dict(), 'E:/learn_DeepLearning/modules/Conv/save_model/best_model.pth')
    print("模型保存至：Conv/save_model/best_model.pth")
    
def predict():
    # Compose()：将多个transforms的操作整合在一起
    data_transform = transforms.Compose([
        # ToTensor()：数据转化为Tensor格式
        transforms.ToTensor()
    ])
    
    # 加载训练数据集
    train_dataset = datasets.FashionMNIST(root='../data', train=True, transform=data_transform, download=True)
    # 给训练集创建一个数据加载器, shuffle=True用于打乱数据集，每次都会以不同的顺序返回
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    
    # 加载测试数据集
    test_dataset = datasets.FashionMNIST(root='../data', train=False, transform=data_transform, download=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)
    
    # 如果有NVIDA显卡，转到GPU训练，否则用CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 模型实例化，将模型转到device
    model = net.to(device)
    
    # 加载train.py里训练好的模型
    model.load_state_dict(torch.load("E:/learn_DeepLearning/modules/Conv/save_model/best_model.pth"))
    
    # 结果类型
    classes = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]
    # 把Tensor转化为图片，方便可视化
    show = ToPILImage()
    
    # 进入验证阶段
    for i in range(10):
        x, y = test_iter[i][0], test_iter[i][1]
        # show()：显示图片
        show(x).show()
        # unsqueeze(input, dim)，input(Tensor)：输入张量，dim (int)：插入维度的索引，最终将张量维度扩展为4维
        x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False).to(device)
        with torch.no_grad():
            pred = model(x)
            print(f"预测概率：{pred}")
            # argmax(input)：返回指定维度最大值的序号
            # 得到验证类别中数值最高的那一类，再对应classes中的那一类
            predicted, actual = classes[torch.argmax(pred[0])], classes[y]
            # 输出预测值与真实值
            print(f'predicted: "{predicted}", actual:"{actual}"')

if __name__ == '__main__':
    # 训练和评估LeNet-5模型
    lr, num_epochs = 0.9, 50
    train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())  # loss 0.471, train acc 0.822, test acc 0.8003  8639.3 examples/sec on cuda:0
    # plt.show()
    # predict()