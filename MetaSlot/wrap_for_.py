import torch
import torch.nn as nn
import torch.nn.functional as F
# from object_centric_bench.utils import Config

class MetaSlot_Wrapper(nn.Module):
    def __init__(self, metaslot):
        super().__init__()
        self.metaslot = metaslot
        # self.wrapped = wrapped # ModelWrap
        self.slot_fc = nn.Identity()  # B,K,512

    def forward(self, image):
        # batch: {'image': tensor, ...}
        # fn(**batch): fn(image=x,...)
        # ModelWrap.forward expect dict/Config as the only positional arg
        feature, slotz, attent, attent2, recon = self.metaslot(image)
        #1,768,14,14 | 1,7,256 | 1,7,14,14 | 1,7,14,14 | 1,768,14,14
        # return feature, slotz, attent, attent2, recon
        return self.slot_fc(slotz), None
    
class MetaSlot_Wrapper_CVCL(nn.Module):
    def __init__(self, metaslot):
        super().__init__()
        self.metaslot = metaslot
        # self.wrapped = wrapped # ModelWrap
        self.slot_fc = nn.Identity()  # B,K,512

    def forward(self, image):
        # batch: {'image': tensor, ...}
        # fn(**batch): fn(image=x,...)
        # ModelWrap.forward expect dict/Config as the only positional arg
        # feature, slotz, attent, attent2, recon = self.metaslot(image)
        _, slotz, _,_,_ = self.metaslot(image)
        #1,768,14,14 | 1,7,256 | 1,7,14,14 | 1,7,14,14 | 1,768,14,14
        # return feature, slotz, attent, attent2, recon
        slotz = slotz.mean(dim=1)
        return self.slot_fc(slotz), None
    
    def get_loss(self, image):#B,C,H,W
        #1,768,14,14 | 1,7,256 | 1,7,14,14 | 1,7,14,14 | 1,768,14,14
        feature, slotz, attent, attent2, recon = self.metaslot(image)
        loss = F.mse_loss(feature, recon)
        return loss