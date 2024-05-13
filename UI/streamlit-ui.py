import streamlit as st
import vtk
from skimage import measure
from scipy.ndimage import gaussian_filter1d


#----------------------------------------------------------------------------------------------------------------------------------------------------------

#The below libraries are for segmentation

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def segment_image(plane, upload):
    #Images are already resized during temp-file upload
    #delete contents in mask folder before populating it 
    path="temp_"+plane+"_masks"
    temp_dir = Path(path)
    temp_dir.mkdir(exist_ok=True)
    # remove contents in the folder before populating 
    for filename in os.listdir(temp_dir):
        file_path = temp_dir / filename
        if os.path.isfile(file_path):  # Check if it's a file (not a directory)
            os.remove(file_path)
    # temp_folder = Path()
    # if plane == "trs":
    #     temp_folder = temp_folder_trs
    # elif plane == "sag":
    #     temp_folder = temp_folder_sag
    # elif plane == "cor":
    #     temp_folder = temp_folder_cor
    # image_paths_to_segment = [str(p) for p in temp_folder.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    # for indiv_image_path in image_paths_to_segment:
    #     filename = os.path.splitext(os.path.basename(indiv_image_path))[0]
    #     model_path = os.curdir+"/YOLO_models/"+plane+"/best.pt"
    #     yolo_model = YOLO(model_path)
    #     results = yolo_model.predict(source=indiv_image_path, conf=0.20)

    #     sam_checkpoint = os.curdir+"./sam_vit_h_4b8939.pth"
    #     model_type = "vit_h"
    #     sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    #     predictor = SamPredictor(sam_model)

    #     segmented_images = []

    #     for result in results:
    #         boxes = result.boxes
    #         bbox = boxes.xyxy.tolist()

    #     image = cv2.cvtColor(cv2.imread(indiv_image_path), cv2.COLOR_BGR2RGB)
    #     predictor.set_image(image)
        
    #     for i in range(len(bbox)):
    #         bbox = boxes.xyxy.tolist()[i]
            
    #         input_box = np.array(bbox)

    #         masks, _, _ = predictor.predict(
    #             point_coords=None,
    #             point_labels=None,
    #             box=input_box[None, :],
    #             multimask_output=False,
    #         )

    #         segmentation_mask = masks[0]
    #         binary_mask = np.where(segmentation_mask > 0.5, 1, 0)

    #         black_background = np.ones_like(image) * 0
    #         mask_color = np.array([255, 255, 255])  

            
    #         new_image = black_background * (1 - binary_mask[..., np.newaxis]) + mask_color * binary_mask[..., np.newaxis]
    #         segmented_images.append(new_image)

    #     overlayed_image = np.zeros_like(image)

    #     for new_image in segmented_images:
    #         overlayed_image = overlayed_image * (1 - new_image) + new_image
    #     # overlayed_image = Image.open(indiv_image_path)
    #     mask_folder = "/temp_"+plane+"_masks"
    #     output_folder = os.curdir+mask_folder
    #     output_path = os.path.join(output_folder, filename + "_mask.png")
    #     overlayed_image.save(output_path)

#----------------------------------------------------------------------------------------------------------------------------------------------------------
# Below is are the functions for visualizing a single plane

def plot_3d_surface_vtk(verts, faces, threshold):
    # Create a VTK PolyData object
    poly_data = vtk.vtkPolyData()
    
    # Create VTK points
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(len(verts))
    for i, vert in enumerate(verts):
        points.SetPoint(i, vert[0], vert[1], vert[2])
    poly_data.SetPoints(points)
    
    # Create VTK cells (triangles)
    polys = vtk.vtkCellArray()
    for face in faces:
        polys.InsertNextCell(3, face)  # Assuming faces are triangles
    poly_data.SetPolys(polys)

    # Create VTK actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create VTK renderer and render window
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    
    # Create VTK render window interactor
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Add actor to the renderer
    renderer.AddActor(actor)

    # Set up the camera
    renderer.GetActiveCamera().Azimuth(30)
    renderer.GetActiveCamera().Elevation(30)

    # Reset the camera to show the entire scene
    renderer.ResetCamera()

    # Set background color
    renderer.SetBackground(1.0, 1.0, 1.0) # (1,1,1) denotes maximum intensity for all RGB colors

    # Render the scene
    render_window.Render()

    # Start the render window interactor
    render_window_interactor.Start()

def load_3d_volume(file_path):
    return np.load(file_path)

# Marching Cubes function
def marching_cubes_3d(image_stack, threshold=0.25, voxel_spacing=(0.5, 0.5, 0.5)):
    binary_images = image_stack > threshold
    try:
        verts, faces, _, _ = measure.marching_cubes(binary_images, level=0, spacing=voxel_spacing)
        return verts, faces
    except RuntimeError as e:
        print(f"Error: {e}")
        return None, None

# Smoothing function
def smooth_surface(verts, sigma=1, axis=0):
    if verts is not None:
        smoothed_verts = np.copy(verts)
        smoothed_verts[:, axis] = gaussian_filter1d(verts[:, axis], sigma=sigma)
        return smoothed_verts
    return None

def create_volumetric_npy(output_folder,plane):
    # Create a 3D volume from the segmented images
    image_dir = Path(output_folder)
    image_files = [str(p) for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]

    if not image_files:
        st.warning("No segmented images found in 'image-output' folder.")
        st.stop()

    first_image = Image.open(image_files[0])
    width, height = first_image.size
    num_slices = len(image_files)

    volume_data = np.zeros((num_slices, height, width), dtype=np.uint8)

    for idx, image_file in enumerate(image_files):
        img = Image.open(image_file)
        img_gray = img.convert('L')  # Convert to grayscale
        img_data = np.array(img_gray)
        volume_data[idx, :, :] = img_data

    if plane == "transverse":
        np.save("volumetric_data_transverse.npy", volume_data)
    elif plane == "saggittal":
        np.save("volumetric_data_saggittal.npy", volume_data)
    elif plane == "coronal":
        np.save("volumetric_data_coronal.npy", volume_data)

#Below is the library for visualisation:
from vedo import load, show, Volume

#UI Code
from pathlib import Path

def save_uploaded_files(plane,uploaded_files):
    path="temp_"+plane+"_images"
    temp_dir = Path(path)
    temp_dir.mkdir(exist_ok=True)
    # remove contents in the folder before populating 
    for filename in os.listdir(temp_dir):
        file_path = temp_dir / filename
        if os.path.isfile(file_path):  # Check if it's a file (not a directory)
            os.remove(file_path)

    for uploaded_file in uploaded_files:
        # Resize image to desired dimensions
        resized_img = Image.open(uploaded_file).resize((800, 800))

        # Save resized image
        resized_file_path = temp_dir / uploaded_file.name
        resized_img.save(resized_file_path)

    return temp_dir




home,trs_tab, sag_tab, cor_tab = st.tabs(["Home","Transverse Plane Images", "Saggittal Plane Images", "Coronal Plane Images"]) 

with home:
    st.title("Ventricle Segmentation and Reconstruction")


    # Initialize an empty list to store log entries
    log = []

    # Function to add log entries
    def add_to_log(entry):
        log.append(entry)
        st.sidebar.write(entry)

    def display_upload_success():
        st.success("Files uploaded successfully.")

    def display_upload_error(message):
        st.error(message)

    def get_file_count(folder_path):
        # Get list of entries in the directory
        entries = os.listdir(folder_path)
        # Count files (excluding directories)
        file_count = sum(1 for entry in entries if os.path.isfile(os.path.join(folder_path, entry)))
        return file_count


    with st.sidebar:
        st.title("Program Log")

    add_to_log("➡️ App initialized.")

    #For transverse plane
    col1,col2 = st.columns([2,1])
    with col1: 
        
        transverse_upload = st.file_uploader('Upload Transverse Plane MRI',accept_multiple_files=True,type=['jpg','png','jpeg'],key=1)
    with col2: 
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        if transverse_upload is not None:
            if len(transverse_upload)>0:
                temp_folder_trs = save_uploaded_files("trs",transverse_upload)
                st.button("Segment Ventricles",use_container_width=-1,key=1.1,on_click=segment_image, args=("trs",transverse_upload))
            else:
                st.button("Segment Ventricles",use_container_width=-1,disabled=True,key="1.1disabled")
    # Check if files have been uploaded
    if transverse_upload is not None:
        if len(transverse_upload)>0:
            display_upload_success()
            log_msg = "➡️ Files uploaded: {count_transverse} Transverse Plane MRI".format(count_transverse=len(transverse_upload))
            add_to_log(log_msg)
    #The above process is repeated for saggittal and coronal planes as well with different column names
    else:
            display_upload_error("Error in uploading image file")
            log_msg = "❌ Error in file upload of transverse plane"
            add_to_log(log_msg)


    col3,col4 = st.columns([2,1])

    with col3: 
        saggittal_upload = st.file_uploader('Upload Saggittal Plane MRI',accept_multiple_files=True, type=['jpg','png','jpeg'], key=2)

    with col4: 
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        if saggittal_upload is not None:
            if len(saggittal_upload)>0:
                temp_folder_sag = save_uploaded_files("sag",saggittal_upload)
                st.button("Segment Ventricles",use_container_width=-1,key=2.1,on_click=segment_image, args=("sag",transverse_upload))
            else:
                st.button("Segment Ventricles",use_container_width=-1,disabled=True,key="2.1disabled")
    if saggittal_upload is not None:
        if len(saggittal_upload)>0:
            log_msg = "➡️ Files uploaded: {count_saggittal} Saggittal Plane MRI".format(count_saggittal=len(saggittal_upload))
            add_to_log(log_msg)
    else:
            display_upload_error("Error in uploading image file")
            log_msg = "❌ Error in file upload of saggittal plane"
            add_to_log(log_msg)

    col5,col6 = st.columns([2,1])

    with col5: 
        coronal_upload = st.file_uploader('Upload Coronal Plane MRI',accept_multiple_files=True, type=['jpg','png','jpeg'],key=3)

    with col6: 
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        if coronal_upload is not None:
            if len(coronal_upload)>0:
                temp_folder_cor = save_uploaded_files("cor",coronal_upload)
                st.button("Segment Ventricles",use_container_width=-1,key=3.1,on_click=segment_image, args=("cor",coronal_upload))
            else:
                st.button("Segment Ventricles",use_container_width=-1,disabled=True,key="3.1disabled")

    if coronal_upload is not None:
        if len(coronal_upload)>0:
            display_upload_success()
            log_msg = "➡️ Files uploaded: {count_coronal} Coronal Plane MRI".format(count_coronal=len(coronal_upload))
            add_to_log(log_msg)
    else:
            display_upload_error("Error in uploading image file")
            log_msg = "❌ Error in file upload of coronal plane"
            add_to_log(log_msg)

    mask_path_trs = os.curdir+"/temp_trs_masks/"
    mask_path_sag = os.curdir+"/temp_sag_masks/"
    mask_path_cor = os.curdir+"/temp_cor_masks/"

    st.write("")
    st.write("")

    def reconstruct(path_to_file,bg=(1,1,1), mesh_color=(1,0,0)):
    # Load a NIfTI file
        vol = Volume(path_to_file)

        # Show the volume
        show(vol, bg=bg)

    if get_file_count(mask_path_trs)>0 or get_file_count(mask_path_sag)>0 or get_file_count(mask_path_cor)>0:
            st.warning("Previous Data Alert: If the mask folder is populated with pre-exisiting masks, it will reconstruct those masks. Do ensure segmentation before reconstruction")

    if get_file_count(mask_path_trs)>0 and get_file_count(mask_path_sag)>0 and get_file_count(mask_path_cor)>0:
        path_to_file = "./good_glm.nii"
        print(path_to_file)
        st.button("Construct 3D Data with NIFti file",on_click=reconstruct,args=(path_to_file,(1,1,1),(1,0,0)))

    def visualise_singlular_plane(plane):
        if plane == "transverse":
            mask_path = mask_path_trs
            volumetric_path = "./volumetric_data_transverse.npy"

        elif plane == "saggittal":
            mask_path = mask_path_sag
            volumetric_path = "./volumetric_data_saggittal.npy"
        elif plane == "coronal":
            mask_path = mask_path_cor
            volumetric_path = "./volumetric_data_coronal.npy"
        create_volumetric_npy(mask_path,plane)

        image_stack = load_3d_volume(Path(volumetric_path))

        # Plot 3D surface
        # threshold = st.slider("Select Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        verts, faces = marching_cubes_3d(image_stack, threshold=0.30, voxel_spacing=(1, 1, 1))
        smoothed_verts = smooth_surface(verts, sigma=1, axis=0)
    
        plot_3d_surface_vtk(smoothed_verts, faces, threshold=0.30)

        # st.success("3D surface rendered with VTK.")




    # Now for individual plane segmentation: 
    if get_file_count(mask_path_trs)>0:
        if st.button("Reconstruct Ventricles for Transverse plane"):
            visualise_singlular_plane("transverse")
    if get_file_count(mask_path_sag)>0:
        if st.button("Reconstruct Ventricles for Saggittal plane"):
            visualise_singlular_plane("saggittal")
    if get_file_count(mask_path_cor)>0:
        if st.button("Reconstruct Ventricles for Coronal plane"):
            visualise_singlular_plane("coronal")
with trs_tab:
    if transverse_upload is not None:
        if len(transverse_upload) == 0:
            st.error("No Images have been uploaded yet")
        else:
            st.header("Raw Images")
            col_counter = 0  # Counter to track columns
            num_cols = 5  # Number of columns for images (5 columns with image width of 125px seems to work well)
            cols = st.columns(num_cols)

            if temp_folder_trs.exists():
                # Get list of image files
                image_paths = [str(p) for p in temp_folder_trs.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]
                image_paths.sort()
            # Display each image
            for image_path in image_paths:
                image_data = Image.open(image_path)
                filename = os.path.splitext(os.path.basename(image_path))[0]
                # Display image in a column based on counter
                with cols[col_counter]:
                    st.image(image_data, caption=filename,width=125)

                # Increment counter and reset if needed
                col_counter = (col_counter + 1) % num_cols
            
        

            st.header("Masked Images")
            
            col_counter = 0  # Counter to track columns
            cols = st.columns(num_cols)

            mask_folder = os.curdir+"/temp_trs_masks/"
            temp_mask_trs = Path(mask_folder)
            mask_paths = [str(p) for p in temp_mask_trs.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]

            for masks in mask_paths:
                mask_data = Image.open(masks)
                filename = os.path.splitext(os.path.basename(masks))[0]
                # Display image in a column based on counter
                with cols[col_counter]:
                    st.image(mask_data, caption=filename,width=125)

                # Increment counter and reset if needed
                col_counter = (col_counter + 1) % num_cols

                




with sag_tab:
    if saggittal_upload is not None:
        if len(saggittal_upload) == 0:
            st.error("No Images have been uploaded yet")
        else:
            st.header("Raw Images")

            col_counter = 0  # Counter to track columns
            num_cols = 5  # Number of columns for images (5 columns with image width of 125px seems to work well)
            cols = st.columns(num_cols)

            if temp_folder_sag.exists():
                # Get list of image files
                image_paths = [str(p) for p in temp_folder_sag.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]
                image_paths.sort()
            # Display each image
            for image_path in image_paths:
                image_data = Image.open(image_path)
                filename = os.path.splitext(os.path.basename(image_path))[0]
                # Display image in a column based on counter
                with cols[col_counter]:
                    st.image(image_data, caption=filename,width=125)

                # Increment counter and reset if needed
                col_counter = (col_counter + 1) % num_cols

            st.header("Masked Images")
            
            col_counter = 0  # Counter to track columns
            cols = st.columns(num_cols)

            mask_folder = os.curdir+"/temp_sag_masks/"
            temp_mask_sag = Path(mask_folder)
            mask_paths = [str(p) for p in temp_mask_sag.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]

            for masks in mask_paths:
                mask_data = Image.open(masks)
                filename = os.path.splitext(os.path.basename(masks))[0]
                # Display image in a column based on counter
                with cols[col_counter]:
                    st.image(mask_data, caption=filename,width=125)

                # Increment counter and reset if needed
                col_counter = (col_counter + 1) % num_cols

with cor_tab:
    if coronal_upload is not None:
        if len(coronal_upload) == 0:
            st.error("No Images have been uploaded yet")
        else:
            st.header("Raw Images")

            col_counter = 0  # Counter to track columns
            num_cols = 5  # Number of columns for images (5 columns with image width of 125px seems to work well)
            cols = st.columns(num_cols)

            if temp_folder_cor.exists():
                # Get list of image files
                image_paths = [str(p) for p in temp_folder_cor.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]
                image_paths.sort()
            # Display each image
            for image_path in image_paths:
                image_data = Image.open(image_path)
                filename = os.path.splitext(os.path.basename(image_path))[0]
                # Display image in a column based on counter
                with cols[col_counter]:
                    st.image(image_data, caption=filename,width=125)

                # Increment counter and reset if needed
                col_counter = (col_counter + 1) % num_cols

            st.header("Masked Images")
            
            col_counter = 0  # Counter to track columns
            cols = st.columns(num_cols)

            mask_folder = os.curdir+"/temp_cor_masks/"
            temp_mask_cor = Path(mask_folder)
            mask_paths = [str(p) for p in temp_mask_cor.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]

            for masks in mask_paths:
                mask_data = Image.open(masks)
                filename = os.path.splitext(os.path.basename(masks))[0]
                # Display image in a column based on counter
                with cols[col_counter]:
                    st.image(mask_data, caption=filename,width=125)

                # Increment counter and reset if needed
                col_counter = (col_counter + 1) % num_cols

