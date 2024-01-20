import torch
import numpy as np

class Converter():
    '''
      convert grid output to bbox
      if n_class = 1, output C for each anchor box is 5
      otherwise, it is n_anchor_box * (n_class + 5)
    '''
    def __init__(self, H_grid, W_grid, n_class, n_anchor_box, model):
        self.H_grid = H_grid
        self.W_grid = W_grid
        if n_class == 1:
            self.C_out = 5
        else:
            self.C_out = 5 + n_class
        self.n_anchor_box = n_anchor_box
        self.n_class = n_class
        self.idx = np.arange(self.H_grid * self.W_grid * self.n_anchor_box)
        self.labels = model.labels
        self.anchors = model.anchors

    def grid_to_bbox(self, grid, cmin):
        grid = grid.permute(1,2,0).reshape(-1, self.C_out)
        
        # select out according to cmin
        cmin = torch.logit(torch.tensor(cmin))
        ijk = self.idx[grid[:,4] > cmin]
        grid = grid[ijk,:]  # take out only these for further analysis
        kk = ijk % self.n_anchor_box  # the kth anchor box
        jj = (ijk // self.n_anchor_box) % self.W_grid  # the jth block in the W dimension, to be added to hx
        ii = (ijk // self.n_anchor_box) // self.W_grid % self.H_grid  # the ith block in the H dimension, to be added to hy
        # print(ii, jj, kk)
        # convert to hx, hy, hw, hh
        grid[:, :2] = torch.sigmoid(grid[:, :2])
        grid[:, 2:4] = torch.exp(grid[:, 2:4])
        grid[:, 4:] = torch.sigmoid(grid[:, 4:])
        bbox_output, class_output, p_output = [], [], []
        for i, j, k, tmp in zip(ii, jj, kk, grid):
            hx, hy, hw, hh = tmp[:4].tolist()
            # print(hx, hy, hw, hh)
            if self.n_class > 1:
                i_class = np.argmax(tmp[5:])
                name_class = self.labels[i_class]
                p = tmp[i_class+5] * tmp[4]
            else:
                name_class = self.labels[0]
                p = tmp[4]
            hw = hw * self.anchors[k][0] / self.W_grid
            hh = hh * self.anchors[k][1] / self.H_grid
            hx = float(hx + j) / self.W_grid
            hy = float(hy + i) / self.H_grid
            bbox_output.append((hx, hy, hw, hh))
            class_output.append(name_class)
            p_output.append(p.item())
        return bbox_output, class_output, p_output

    def _bbox_to_xy(self, bbox):
        hx, hy, hw, hh = bbox
        xmin = hx - 0.5 * hw 
        xmax = hx + 0.5 * hw 
        ymin = hy - 0.5 * hh 
        ymax = hy + 0.5 * hh
        return xmin, xmax, ymin, ymax

    def _get_i_u(self, x1min, x1max, x2min, x2max):
        # get the intersection and union for two segments
        if x1max < x2min or x2max < x1min:
            return 0,1
        return min(x1max,x2max) - max(x1min, x2min),\
               max(x1max,x2max) - min(x1min, x2min)

    def iou(self, bbox1, bbox2):
        x1min, x1max, y1min, y1max = self._bbox_to_xy(bbox1)
        x2min, x2max, y2min, y2max = self._bbox_to_xy(bbox2)
        x_i, x_u = self._get_i_u(x1min, x1max, x2min, x2max)
        y_i, y_u = self._get_i_u(y1min, y1max, y2min, y2max)
        return float(x_i)*float(y_i)/(x_u*y_u)

    def nonmax_compressed(self, bbox_output, class_output, p_output):
        bbox_sv, class_sv, p_sv = [], [], []
        while len(p_output) > 0:
            idx = np.argmax(p_output)
            # print('selected:', idx)
            bbox_sv.append(bbox_output[idx])
            class_sv.append(class_output[idx])
            p_sv.append(p_output[idx])
            to_pop = []
            for i in range(len(bbox_output)):
                if i == idx or class_output[i] == class_output[idx] and self.iou(bbox_output[i], bbox_output[idx]) > 0.5:
                    to_pop.append(i)
            # print(to_pop)
            for i in sorted(to_pop, reverse=True):
                bbox_output.pop(i)
                class_output.pop(i)
                p_output.pop(i)
        return bbox_sv, class_sv, p_sv