
import torch 
import torch.nn as nn 
import smoothing_block as smoothing 
import vggnet_mcdo_block
import vggnet_dnn_block

class CSRMSA(nn.Module):
    def __init__(self, 
        block=vggnet_dnn_block.BasicBlock, 
        num_blocks=[2, 2, 3, 3, 3, 3, 1, 1 ], 
        sblock=smoothing.TanhBlurBlock, 
        num_sblocks=(1,1,1,1,1,1, 1, 1), 
        name="CSRMSA"):
        super(CSRMSA, self).__init__()
        self.seen = 0 
        self.name = name 
        self.layer0 = self._make_layer(block, 3, 64, num_blocks[0], pool=False)
        self.layer1 = self._make_layer(block, 64, 128, num_blocks[1], pool=True)
        self.layer2 = self._make_layer(block, 128, 256, num_blocks[2], pool=True)
        self.layer3 = self._make_layer(block, 256, 512, num_blocks[3], pool=True)
        self.layer4 = self._make_layer(block, 512, 512, num_blocks[4], pool=True)
        ## CSR backend 
        #self.layer5 = self._make_layer(block, 512, 256, num_blocks[5], pool=False)
        #self.layer6 = self._make_layer(block, 256, 128, num_blocks[6], pool=False)
        #self.layer7 = self._make_layer(block, 128,  64, num_blocks[7], pool=False)

        self.smooth0 = self._make_smooth_layer(sblock, 64, num_sblocks[0])
        self.smooth1 = self._make_smooth_layer(sblock, 128, num_sblocks[1])
        self.smooth2 = self._make_smooth_layer(sblock, 256, num_sblocks[2])
        self.smooth3 = self._make_smooth_layer(sblock, 512, num_sblocks[3])
        self.smooth4 = self._make_smooth_layer(sblock, 512, num_sblocks[4])
        self.smooth5 = self._make_smooth_layer(sblock, 256, num_sblocks[5])
        self.smooth6 = self._make_smooth_layer(sblock, 256, num_sblocks[6])
        self.smooth7 = self._make_smooth_layer(sblock, 256, num_sblocks[7])

        self.output_layer = self._make_layer(block, 64,  1, 1, pool=False)


        self.backend_feat  = [512, 512, 512, 256,128,64]
        self.backend = self._make_backfeat(self.backend_feat, in_channels = 512, dilation = True)

    def forward(self, x):
        x = self.layer0(x) #64
        x = self.smooth0(x)

        x = self.layer1(x) #128
        x = self.smooth1(x)

        x = self.layer2(x) #256
        x = self.smooth2(x)

        x = self.layer3(x) #512
        x = self.smooth3(x)

        #x = self.layer4(x) #512
        #x = self.smooth4(x)

        #x = self.layer5(x)
        #x = self.smooth5(x)

        #x = self.layer6(x)
        #x = self.smooth7(x)

        #x = self.layer7(x)
        #x = self.smooth7(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x 
    
    # below static methods retrieved from 
    # https://github.com/xxxnell/how-do-vits-work
    @staticmethod
    def _make_layer(block, in_channels, out_channels, num_blocks, pool):
        layers, channels = [], in_channels
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        for _ in range(num_blocks):
            layers.append(block(channels, out_channels))
            channels = out_channels
        return nn.Sequential(*layers)

    @staticmethod
    def _make_smooth_layer(sblock, in_filters, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(sblock(in_filters=in_filters))
        return nn.Sequential(*layers)

    @staticmethod
    def _make_backfeat(cfg, in_channels = 3,batch_norm=False,dilation = False):
        if dilation:
            d_rate = 2
        else:
            d_rate = 1
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)   


# model obtained from CSRNet paper. 
class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        """
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            for i in xrange(len(self.frontend.state_dict().items())):
                self.frontend.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]
                """
    def forward(self,x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
                
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)