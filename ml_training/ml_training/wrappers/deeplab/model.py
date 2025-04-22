import torch
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassJaccardIndex, BinaryJaccardIndex
from torch.optim.lr_scheduler import ReduceLROnPlateau


class DeepLabV3PlusModule(pl.LightningModule):
    def __init__(self, classes, lr=1e-3, encoder='resnet50'):
        super(DeepLabV3PlusModule, self).__init__()
        num_classes = len(classes) + 1
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes
        )
        self.lr = lr
        self.loss_fn = smp.losses.SoftBCEWithLogitsLoss() # TODO: check this
        self.metrics = MulticlassJaccardIndex(num_classes=num_classes)
        self.binary_metrics = {cls: BinaryJaccardIndex() for cls in classes}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch

        if len(images) == 1:
            # Batch Noramlization cant handle batch with size 1
            # That's why we need turn this layers to eval mode
            self.turn_off_batchnorm()
            outputs = self.model(images)
            self.turn_on_batchnorm()
        else:
            outputs = self.model(images)

        loss = self.loss_fn(outputs, masks)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        
        pred_mask = torch.argmax(outputs, dim=1)
        target_mask = torch.argmax(masks, dim=1)
        iou = self.metrics(pred_mask, target_mask)
        self.log('train_iou', iou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, masks)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        
        pred_mask = torch.argmax(outputs, dim=1)
        target_mask = torch.argmax(masks, dim=1)
        iou = self.metrics(pred_mask, target_mask)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        
        for i, cls_name in enumerate(self.binary_metrics):
            self.binary_metrics[cls_name].to(pred_mask.device)
            iou = self.binary_metrics[cls_name]((pred_mask == (i + 1)), (target_mask == (i + 1)))
            if torch.isnan(iou).item():
                continue
            self.log(f'{cls_name}_val_iou', iou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
            
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min')
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'train_loss'}

    def turn_off_batchnorm(self):
        for m in self.model.modules():
            if not isinstance(m, torch.nn.BatchNorm2d):
                continue
            
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False
    
    def turn_on_batchnorm(self):
        for m in self.model.modules():
            if not isinstance(m, torch.nn.BatchNorm2d):
                continue
            
            m.train()
            m.weight.requires_grad = True
            m.bias.requires_grad = True