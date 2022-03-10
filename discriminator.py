import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx, input, coeff=1.):
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.coeff, None


class GRL(nn.Module):
    def __init__(self):
        super(GRL, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

class Discriminator(nn.Module):
  def __init__(self, input_dim=768, out_dim=2, drop_rate=0.1):
    super(Discriminator, self).__init__()
    self.grl = GRL()
    self.linear1 = nn.Linear(input_dim, 256)
    self.linear2 = nn.Linear(256, 128)
    self.clf = nn.Linear(128, out_dim)
    self.drop1 = nn.Dropout(drop_rate)
    self.drop2 = nn.Dropout(drop_rate)
    self.drop3 = nn.Dropout(drop_rate)

  def forward(self, x):
    x = self.grl(x)
    x = F.relu(self.linear1(x))
    x = self.drop1(x)
    x = F.relu(self.linear2(x))
    x = self.drop2(x)
    logits = F.relu(self.clf(x))
    logits = self.drop3(x)
    return logits