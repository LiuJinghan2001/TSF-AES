import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            #nn.BatchNorm1d(bottleneck),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck, self).__init__()

        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.se(out)
        out += residual

        return out 

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

class ECAPA_TDNN(nn.Module): # Here we use a small ECAPA-TDNN, C=512. In my experiences, C=1024 slightly improves the performance but need more training time.
    def __init__(self, C = 1024, num_class=5944,dropout=0.1):
        super(ECAPA_TDNN, self).__init__()
        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            )
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # the final layer
        self.weight = nn.Parameter(torch.FloatTensor(num_class, 192), requires_grad=True).cuda()
        nn.init.xavier_normal_(self.weight, gain=1)

    def forward(self, x):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)
        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)
        t = x.size()[-1]
        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )
        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        embedding = self.bn6(x)

        if self.dropout is not None:
            embedding = self.dropout(embedding)

        logits = F.linear(F.normalize(embedding), F.normalize(self.weight, dim=-1))
        return embedding, logits
        
class SpeakerIdetification(nn.Module):
    def __init__(self,backbone, num_class=5944,dropout=0.1):
        """The speaker identification model, which includes the speaker backbone network
           and the a linear transform to speaker class num in training
        Args:
            backbone (Paddle.nn.Layer class): the speaker identification backbone network model
            num_class (_type_): the speaker class num in the training dataset
            lin_blocks (int, optional): the linear layer transform between the embedding and the final linear layer. Defaults to 0.
            lin_neurons (int, optional): the output dimension of final linear layer. Defaults to 192.
            dropout (float, optional): the dropout factor on the embedding. Defaults to 0.1.
        """
        super(SpeakerIdetification, self).__init__()
        # speaker idenfication backbone network model
        # the output of the backbond network is the target embedding
        self.backbone = backbone
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # the final layer
        self.weight = nn.Parameter(torch.FloatTensor(num_class, 192), requires_grad=True).cuda()
        nn.init.xavier_normal_(self.weight, gain=1)

    def forward(self, x, aug=True):
        """Do the speaker identification model forwrd,
           including the speaker embedding model and the classifier model network
        Args:
            x (paddle.Tensor): input audio feats,
                               shape=[batch, dimension, times]
            lengths (paddle.Tensor, optional): input audio length.
                                        shape=[batch, times]
                                        Defaults to None.
        Returns:
            paddle.Tensor: return the logits of the feats
        """
        # x.shape: (N, C, L)
        embedding = self.backbone(x,aug)  # (N, emb_size)
        if self.dropout is not None:
            embedding = self.dropout(embedding)

        logits = F.linear(F.normalize(embedding), F.normalize(self.weight, dim=-1))

        return embedding, logits

from pytorch_revgrad import RevGrad

class AATNet(nn.Module): # AAT system
    def __init__(self, **kwargs):
        super(AATNet, self).__init__()
        layers = []
        layers.append(torch.nn.Sequential(
            nn.BatchNorm1d(384),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(384,512),
        ))
        layers.append(torch.nn.Sequential(
            nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512,2),
        ))
        self.matcher = torch.nn.Sequential(*layers)

    def reset_parameters(self):
        self.matcher.reset_parameters()

    def forward(self, x):
        return self.matcher(x)

class Reverse(nn.Module):
    def __init__(self, **kwargs):
        super(Reverse, self).__init__()   
        layers = [RevGrad()]
        self.matcher = torch.nn.Sequential(*layers)

    def reset_parameters(self):
        self.matcher.reset_parameters()

    def forward(self, x):
        return self.matcher(x)