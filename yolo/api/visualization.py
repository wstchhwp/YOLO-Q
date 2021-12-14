import random
import numpy as np
from itertools import repeat
from ..utils.plots import plot_one_box


class Visualizer(object):
    """Visualization of one model."""
    def __init__(self, names, colors=None) -> None:
        super().__init__()
        num_classes = len(names)

        self.names = names
        self.colors = colors or [
            [random.randint(0, 255) for _ in range(3)] for _ in range(num_classes)
        ]

    def draw_one_img(self, img, output, vis_conf=0.4, offset=0):
        """Visualize one images.
        
        Args:
            imgs (numpy.ndarray): one image.
            outputs (torch.Tensor): one output, (num_boxes, classes+5)
            vis_confs (float, optional): Visualize threshold.
        Return:
            img (numpy.ndarray): Image after visualization.           
        """
        if output is None or len(output) == 0:
            return img
        if isinstance(output, list):
            output = output[0]
        for (*xyxy, conf, cls) in reversed(output[:, :6]):
            if conf < vis_conf:
                continue
            label = '%s %.2f' % (self.names[int(cls)], conf)
            color = self.colors[int(cls)]
            # xyxy[0] += offset
            plot_one_box(xyxy, img, label=label,
                         color=color, 
                         line_thickness=2)
        return img

    def draw_multi_img(self, imgs, outputs, vis_confs=0.4):
        """Visualize multi images.
        
        Args:
            imgs (List[numpy.array]): multi images.
            outputs (List[torch.Tensor]): multi outputs, List[num_boxes, classes+5].
            vis_confs (float | tuple[float], optional): Visualize threshold.
        Return:
            imgs (List[numpy.ndarray]): Images after visualization.           
        """
        if isinstance(vis_confs, float):
            vis_confs = list(repeat(vis_confs, len(imgs)))
        assert len(imgs) == len(outputs) == len(vis_confs)
        for i, output in enumerate(outputs):  # detections per image
            self.draw_one_img(imgs[i], output, vis_confs[i])
        return imgs

    def draw_imgs(self, imgs, outputs, vis_confs=0.4, offset=0):
        if isinstance(imgs, np.ndarray):
            return self.draw_one_img(imgs, outputs, vis_confs, offset)
        else:
            return self.draw_multi_img(imgs, outputs, vis_confs)
