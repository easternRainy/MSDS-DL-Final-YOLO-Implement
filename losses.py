import torch
import torch.nn as nn

# Referenced from: 
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/utils.py
def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        
    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C+self.B*5)
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, best_box = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3)
        
        box_loss = self._box_coord_loss(exists_box, best_box, predictions, target)
        object_loss = self._object_loss(exists_box, best_box, predictions, target)
        no_object_loss = self._no_object_loss(exists_box, best_box, predictions, target)
        class_loss = self._class_loss(exists_box, best_box, predictions, target)
        
        loss = self.lambda_coord * box_loss + \
               object_loss + \
               self.lambda_noobj * no_object_loss + \
               class_loss
        
#         print(f"box_loss: {box_loss}, object_loss: {object_loss}, no_object_loss: {no_object_loss}, class_loss: {class_loss}")
        return loss
        
    def _box_coord_loss(self, exists_box, best_box, predictions, target):
        box_predictions = exists_box * (
            best_box * predictions[..., 26:30] + 
            (1-best_box) * predictions[..., 21:25]
        )
        
        box_targets = exists_box * target[..., 21:25]
        
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(
                box_predictions[..., 2:4] + 1e-6
            )
        )
        
        
        box_targets[..., 2:4] = torch.sqrt(
            box_targets[..., 2:4]
        )
        
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )
        
        return box_loss
        
          
    def _object_loss(self, exists_box, best_box, predictions, target):
        pred_box = (
            best_box * predictions[..., 25:26] + 
            (1-best_box) * predictions[..., 20:21]
        )
        
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )
        
        return object_loss
    
    def _no_object_loss(self, exists_box, best_box, predictions, target):
        no_object_loss = self.mse(
            torch.flatten(
                (1 - exists_box) * predictions[..., 20:21],
                start_dim = 1
            ),
            
            torch.flatten(
                (1 - exists_box) * target[..., 20:21],
                start_dim = 1
            )
        )
        
        no_object_loss += self.mse(
            torch.flatten(
                (1 - exists_box) * predictions[..., 25:26],
                start_dim = 1
            ),
            
            torch.flatten(
                (1 - exists_box) * target[..., 20:21],
                start_dim = 1
            )
        )
        
        return no_object_loss
    
    def _class_loss(self, exists_box, best_box, predictions, target):
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2)
        )
        
        return class_loss
    
