# Final-Year-Project---Fetal-Ventriculomegaly

## For the UI: 

![Screenshot_20240515_120638](https://github.com/Takashi069/Final-Year-Project---Fetal-Ventriculomegaly/assets/73834506/ff36160d-1ede-46fb-847a-4b3d89d93318)

* Make sure the file structure is as follows:
* * temp_cor_images (folder)
  * temp_cor_masks (folder)
  * temp_sag_images (folder)
  * temp_sag_masks (folder)
  * temp_trs_images (folder)
  * temp_trs_images (folder)
  * YOLO_models (folder) --> contains the models for YOLOv8
  * * cor (folder) 
    * sag (folder)
    * trs (folder)

> [!NOTE] 
> Ensure that there is a checkpoint for SAM_model within the UI folder, the file after doing the command
> `!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth` is stored in the UI folder.
