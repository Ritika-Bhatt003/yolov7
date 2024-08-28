import argparse # Parser for command-line options, arguments and sub-commands
import time # For timing the inference to print the time taken by the model
from pathlib import Path # For handling file paths
import cv2 # For image processing and displaying
import torch # PyTorch is an open source machine learning library based on the Torch library, # For deep learning model and tensor operations
#import torch.backends.cudnn as cudnn # CUDA specific functions in PyTorch 
from numpy import random # For random number generation for colors
import numpy as np # NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays

from models.experimental import attempt_load # For loading the model
from utils.datasets import LoadStreams, LoadImages # For loading the images and videos
from utils.general import check_img_size, \
    check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, strip_optimizer, set_logging, \
    increment_path # For general utility functions like logging, image size checking, non-maximum suppression, classifier application, coordinate scaling, optimizer stripping, path incrementing
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel # For PyTorch utility functions like device selection, classifier loading, time synchronization, model tracing

from sort import * # For the SORT tracker implementation

# Function to Draw Bounding boxes
def draw_boxes(img, bbox, identities=None, categories=None, confidences=None, names=None, colors=None): # Function to draw bounding boxes on the image
    for i, box in enumerate(bbox): # Loop over all the bounding boxes 
        x1, y1, x2, y2 = [int(i) for i in box] # Get the coordinates of the bounding box
        tl = opt.thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # Get the thickness of the bounding box

        cat = int(categories[i]) if categories is not None else 0 # Get the category and identity of the object 
        id = int(identities[i]) if identities is not None else 0 # Get the category and identity of the object

        color = colors[cat] # Get the color of the bounding box

        if not opt.nobbox: # If the option to not show bounding box is not selected
            cv2.rectangle(img, (x1, y1), (x2, y2), color, tl) # Draw the bounding box

        if not opt.nolabel: # If the option to not show label is not selected
            label = str(id) + ":" + names[cat] if identities is not None else f'{names[cat]} {confidences[i]:.2f}' # Get the label to be displayed
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0] # Get the size of the text to be displayed
            c2 = x1 + t_size[0], y1 - t_size[1] - 3 # Get the coordinates of the text to be displayed
            cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled # Draw the rectangle to display the text
            cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA) # Display the text

    return img # Return the image with the bounding boxes

def detect(save_img=False): # Function to detect objects in the image
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace # Get the source, weights, view image, save text, image size and trace options
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images # Save the images if the source is not a text file
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://')) # Is the source a webcam or a video file
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run # Get the save directory
    if not opt.nosave: # if save_dir is not empty then create the directory
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging() # Set the logging for the model to print the logs on the console 
    device = select_device(opt.device) # Select the device to run the model on 
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model, use 32-bit floating point precision
    stride = int(model.stride.max())  # model stride,  Loads a pre-trained model from the weights file and sets the model to evaluation mode
    imgsz = check_img_size(imgsz, s=stride)  # check img_size, Ensures that the input image size (imgsz) is compatible with the model's stride (s).

    if trace: # Trace the model if the option is selected
        model = TracedModel(model, device, opt.img_size) # Trace the model 

    if half: # If the model is to be run in half precision only supported on CUDA
        model.half()  # to FP16 # Convert the model to half precision

    # Second-stage classifier
    classify = False # Initialize the classifier to False
    if classify: # If the classifier is to be used
        modelc = load_classifier(name='resnet101', n=2)  # initialize second-stage classifier # Load the classifier model 
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval() # Load the classifier model and set it to evaluation mode

    # Set Dataloader
    vid_path, vid_writer = None, None # Initialize the video path and video writer
    if webcam: # Initialize the video path and video writer if the source is a webcam
        view_img = check_imshow() # Check if the image is to be displayed
        #cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride) # Set the dataset to load the stream
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride) # Set the dataset to load the images

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names # Get the names of the classes
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names] # Get the colors for the bounding boxes

    # Run inference
    if device.type != 'cpu': # If the device is not CPU
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once # Run the model once to initialize the model and the device 
    old_img_w = old_img_h = imgsz # Initialize the old image width and height
    old_img_b = 1 # Initialize the old image batch size

    t0 = time.time() # Get the current time
    startTime = 0 # Initialize the start time

    for path, img, im0s, vid_cap in dataset: # Loop over the images in the dataset
        img = torch.from_numpy(img).to(device) # Convert the image to a tensor and move it to the device
        img = img.half() if half else img.float()  # uint8 to fp16/32 # Convert the image to half precision if the model is to be run in half precision
        img /= 255.0  # 0 - 255 to 0.0 - 1.0 # Normalize the image
        if img.ndimension() == 3: # If the image has 3 dimensions
            img = img.unsqueeze(0) # Add a dimension to the image

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]): # If the device is not CPU and the image dimensions have changed
            old_img_b = img.shape[0] # Update the old image batch size
            old_img_h = img.shape[2] # Update the old image height
            old_img_w = img.shape[3] # Update the old image width
            for i in range(3): # 3 warmup passes without saving # Run the model for 3 warmup passes without saving the results
                model(img, augment=opt.augment)[0] # Run the model on the image

        # Inference
        t1 = time_synchronized() # Get the current time
        pred = model(img, augment=opt.augment)[0] # Run the model on the image
        t2 = time_synchronized() # Get the current time

        # Apply NMS,  NMS helps filter out overlapping or redundant predictions. It ensures that each object in the image is identified accurately without duplication.
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms) # Apply non-maximum suppression to the predictions
        t3 = time_synchronized() # Get the current time

        # Apply Classifier, It's like double-checking the first guess to make sure it's correct.
        if classify: # second-stage classifier
            pred = apply_classifier(pred, modelc, img, im0s) # Apply the classifier to the predictions

        # Process detections
        for i, det in enumerate(pred):  # detections per image # Loop over the detections
            if webcam:  # batch_size >= 1 # If the batch size is greater than or equal to 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count # Get the path, string, image and frame 
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0) # Get the path, string, image and frame

            p = Path(p)  # to Path # Convert the path to a Path object
            save_path = str(save_dir / p.name)  # img.jpg # Get the save path
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique(): # check unique classes # Loop over the unique classes
                    n = (det[:, -1] == c).sum()  # detections per class # Get the number of detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string # Add the number of detections to the string

                dets_to_sort = np.empty((0, 6)) # Initialize the detections to sort array
                # NOTE: We send in detected object class too
                for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy(): # Loop over the detections to get the coordinates, confidence and class
                    dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass]))) # Add the detections to the detections to sort array
 
                if opt.track: # If the tracking option is selected 
                    tracked_dets = sort_tracker.update(dets_to_sort, opt.unique_track_color) # Update the tracker with the detections
                    tracks = sort_tracker.getTrackers()  # Get the tracks from the tracker

                    # draw boxes for visualization
                    if len(tracked_dets) > 0: # If there are tracked detections 
                        bbox_xyxy = tracked_dets[:, :4] # Get the bounding box coordinates
                        identities = tracked_dets[:, 8] # Get the identities
                        categories = tracked_dets[:, 4] # Get the categories
                        confidences = None # Get the confidences 

                        if opt.show_track: # If the show track option is selected
                            # loop over tracks and draw the path
                            for t, track in enumerate(tracks): # Loop over the tracks
                                track_color = colors[int(track.detclass)] if not opt.unique_track_color else sort_tracker.color_list[t] # Get the color of the track

                                [cv2.line(im0, (int(track.centroidarr[i][0]), # Draw the path of the track
                                                int(track.centroidarr[i][1])), 
                                                (int(track.centroidarr[i + 1][0]), 
                                                int(track.centroidarr[i + 1][1])),
                                                track_color, thickness=opt.thickness)
                                 for i, _ in enumerate(track.centroidarr) # Loop over the centroids of the track
                                 if i < len(track.centroidarr) - 1] # If the index is less than the length of the centroid array - 1
                else:
                    bbox_xyxy = dets_to_sort[:, :4] # Get the bounding box coordinates
                    identities = None # Get the identities
                    categories = dets_to_sort[:, 5] # Get the categories
                    confidences = dets_to_sort[:, 4] # Get the confidences

                im0 = draw_boxes(im0, bbox_xyxy, identities, categories, confidences, names, colors) # Draw the bounding boxes on the image 

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS') # Print the time taken for inference and NMS 

            # Stream results
            if dataset.mode != 'image' and opt.show_fps: # If the mode is not image and the show fps option is selected 
                currentTime = time.time() # Get the current time 
                fps = 1 / (currentTime - startTime) # Calculate the frames per second
                startTime = currentTime # Update the start time
                cv2.putText(im0, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2) # Display the frames per second

            if view_img:
                cv2.imshow(str(p), im0) # Display the image
                cv2.waitKey(1)  # 1 millisecond  

            # Save results (image with detections)
            if save_img: # Save the image with the detections 
                if dataset.mode == 'image': 
                    cv2.imwrite(save_path, im0) # Save the image with the detections
                    print(f" The image with the result is saved in: {save_path}") # Print the path where the image is saved
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video # If the video path is not the same as the save path
                        vid_path = save_path # Update the video path
                        if isinstance(vid_writer, cv2.VideoWriter): # release previous video writer # If the video writer is an instance of the cv2 VideoWriter
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video # If the video capture is not empty
                            fps = vid_cap.get(cv2.CAP_PROP_FPS) # Get the frames per second
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Get the frame width
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Get the frame height
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0] # Set the frames per second, width and height
                            save_path += '.mp4' # Add the mp4 extension to the save path
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)) # Initialize the video writer
                    vid_writer.write(im0) # Write the image to the video writer

    if save_txt or save_img: # Save results # If the save text or save image option is selected
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else '' # Get the number of labels saved to the directory
        print(f"Results saved to {save_dir}{s}") # Print the results saved to the directory

    print(f'Done. ({time.time() - t0:.3f}s)') # Print the time taken to complete the process


if __name__ == '__main__': 
    parser = argparse.ArgumentParser() # Create an argument parser
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)') # Get the weights of the model
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)') # Get the image size 
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold') # Get the object confidence threshold
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS') # Get the IOU threshold for NMS
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu') # Get the device to run the model on
    parser.add_argument('--view-img', action='store_true', help='display results') # Display the results
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt') # Save the results to a text file 
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels') # Save the confidences in the labels
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos') # Do not save the images or videos
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3') # Filter by class
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS') # Class agnostic NMS
    parser.add_argument('--augment', action='store_true', help='augmented inference') # Augmented inference
    parser.add_argument('--update', action='store_true', help='update all models') # Update all models
    parser.add_argument('--project', default='runs/detect', help='save results to project/name') # Save the results to the project
    parser.add_argument('--name', default='exp', help='save results to project/name') # Save the results to the name
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment') # Existing project name ok
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model') # Don't trace the model

    parser.add_argument('--track', action='store_true', help='run tracking') # Run tracking
    parser.add_argument('--show-track', action='store_true', help='show tracked path')  # show tracked path
    parser.add_argument('--show-fps', action='store_true', help='show fps') # show fps
    parser.add_argument('--thickness', type=int, default=2, help='bounding box and font size thickness') # Get the thickness of the bounding box and font
    parser.add_argument('--seed', type=int, default=1, help='random seed to control bbox colors') # Get the random seed to control the bounding box colors
    parser.add_argument('--nobbox', action='store_true', help='don`t show bounding box') # don't show bounding box
    parser.add_argument('--nolabel', action='store_true', help='don`t show label') # don't show label
    parser.add_argument('--unique-track-color', action='store_true', help='show each track in unique color') # show each track in unique color

    opt = parser.parse_args() # Parse the arguments 
    print(opt) # Print the arguments
    np.random.seed(opt.seed) # Set the random seed to control the bounding box colors


    sort_tracker = Sort(max_age=5, # The maximum number of frames to keep the object in the memory
                       min_hits=2, # The minimum number of hits to consider the object
                       iou_threshold=0.2) # The IOU threshold to consider the object

    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad(): # Turn off the gradient calculation to speed up the inference
        if opt.update:  # update all models (to fix SourceChangeWarning) # If the update option is selected
            for opt.weights in ['yolov7.pt']: # Loop over the weights
                detect() # Detect the objects
                strip_optimizer(opt.weights) # Strip the optimizer
        else:
            detect() # Detect the objects

#python detect_or_track.py --weights yolov7.pt --img-size 416 --conf-thres 0.3 --iou-thres 0.4 --view-img --nosave --source people2.mp4 --track --show-track
