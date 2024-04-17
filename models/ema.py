from copy import deepcopy

import torch


class ModelEMA(object):
    def __init__(self, args, model, decay):
        self.ema = deepcopy(model)  # deepcopy用于创建一个新的对象，同时将原始对象的所有内容复制到新对象中。
        self.ema.cuda()
        self.ema.eval()  # 预测阶段，model.eval()不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化，
        # pytorch框架会自动把BN和Dropout固定住，不会取平均，而是用训练好的值，不然的话，一旦test的batch_size过小，很容易就会被BN层影响结果。
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')  # hasattr(object,name)函数用于判断是否包含对应的属性
        # Fix EMA. https://github.com/valencebond/FixMatch_pytorch thank you!
        self.param_keys = [k for k, _ in self.ema.named_parameters()]  # 迭代打印model.named_parameters()将会打印每一次迭代元素的名字和param
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]  # 保存模型参数中的不可学习参数及其名字，如BN中的均值和方差(running_mean 和 running_var)，返回类型为生成器
        for p in self.ema.parameters():  # 迭代打印model.parameters()将会打印每一次迭代元素的param而不会打印名字，这是它和named_parameters的区别，两者都可以用来改变requires_grad的属性
            p.requires_grad_(False)  # 停止梯度计算，原先的梯度信息将被清除

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        # hasattr() 函数用来判断某个类实例对象是否包含指定名称的属性或方法
        # model中包含'module'，但ema中不包含，但ema不是model deepcopy得到的吗？(更新过程中)修改了？毕竟deepcopy修改一个不改变另一个
        with torch.no_grad():
            msd = model.state_dict()  # 查看网络参数，字典格式
            esd = self.ema.state_dict()
            for k in self.param_keys:  # 模型中所有迭代元素
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)
                # 指数移动平均

            for k in self.buffer_keys:  # 不可学习参数
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])
