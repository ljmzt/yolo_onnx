from torchvision.transforms import functional as TF
import torch

class PasteImage():
    '''
      put a size H * W image inside an H * H grey image
      H has to be the longer one of H or W
    '''
    def __init__(self, H):
        self.H = H
    def __call__(self, img):
        _, H, W = img.shape
        img_out = torch.ones(3, self.H, self.H)*0.5
        if H == self.H:
            img = TF.resize(img, [H*self.H//W, self.H])
            _, H, W = img.shape
            y0 = (self.H - H) // 2
            img_out[:,y0:y0+H,:] = img[:,:,:]
        else:
            img = TF.resize(img, [self.H, W*self.H//H])
            _, H, W = img.shape
            x0 = (self.H - W) // 2
            img_out[:,:,x0:x0+W] = img[:,:,:]
        return img_out