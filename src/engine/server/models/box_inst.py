import numpy as np
import torch
import os
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from server.services.errors import Errors, PortalError
from server.services.hashing import get_hash

from server.models.abstract.BaseModel import BaseModel
from adet.config import get_cfg


class NuviBoxInst(BaseModel):
    def register(self):
        self.config_path = os.path.join(self._directory_, "config.yaml")
        self.weights_path = os.path.join(self._directory_, "model.pth")
        if not os.path.isfile(self.config_path):
            raise PortalError(
                Errors.INVALIDFILEPATH,
                f"{self.config_path} is not found in given directory.",
            )

        if not os.path.isfile(self.weights_path):
            raise PortalError(
                Errors.INVALIDFILEPATH,
                f"{self.weights_path} is not found in given directory",
            )
        self.threshold = 0.33
        self.cfg = self.setup_cfg()
        self._load_label_map_()
        self.cpu_device = torch.device("cpu")
        self.predictor = DefaultPredictor(self.cfg)
        self.instance_mode = ColorMode.IMAGE

        self._height_ = 1024 if self._height_ is None else self._height_
        self._width_ = 1024 if self._width_ is None else self._width_

        self._key_ = get_hash(self._directory_)
        print(self._key_)
        return self._key_, self

    def _load_label_map_(self):
        self._label_map_ = {}

        for label_index, label_name in enumerate(self.cfg.CLASS_NAMES):
            print(label_index, label_name)
            self._label_map_[str(label_index)] = {"id": label_index,
                                                  "name": label_name,
                                                  }

    def setup_cfg(self):
        # load config from file and command-line arguments
        cfg = get_cfg()
        cfg.__setattr__('CLASS_NAMES', [])
        cfg.merge_from_file(self.config_path)
        cfg.merge_from_list(['MODEL.WEIGHTS', self.weights_path])
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = self.threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
        cfg.MODEL.FCOS.INFERENCE_TH_TEST = self.threshold
        cfg.MODEL.MEInst.INFERENCE_TH_TEST = self.threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = self.threshold
        cfg.freeze()
        return cfg

    def load(self):
        loaded_model = DefaultPredictor(self.cfg)
        return loaded_model

    def predict(self, model, image_array):
        try:
            predictions = model(image_array)
            if not predictions["instances"].has('pred_masks'):
                masks = np.empty(0)
            else:
                masks = predictions["instances"].pred_masks.cpu().detach().numpy()
            classes = predictions["instances"].pred_classes.cpu().detach().numpy()
            scores = predictions["instances"].scores.cpu().detach().numpy()
            boxes = predictions["instances"].pred_boxes.tensor.cpu().detach().numpy().astype(np.int64)

            # Hacky way to make classes start from 0 (because originally background is 0)
            # classes -= 1
            results = {'detection_masks': masks, 'detection_classes': classes,
                       'detection_scores': scores, 'detection_boxes': boxes}

            return results

        except Exception as e:  # pylint: disable=broad-except
            raise PortalError(Errors.FAILEDPREDICTION, str(e)) from e