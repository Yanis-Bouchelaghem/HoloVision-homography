import json
import cv2
import numpy as np
import os.path
import pathlib
from pathlib import Path


# This function tests if a folder is in a path
def is_folder_in_path_string(folder, path):
    # Normalize the strings to avoid case-sensitivity issues or slash/backslash confusion
    folder_normalized = folder.lower().replace('\\', '/')
    path_normalized = path.lower().replace('\\', '/')

    # Check if the folder name is in the path string
    return ('/' + folder_normalized + '/') in ('/' + path_normalized + '/')


def apply_homography(path_to_img, path_to_json_file, path_to_save_new_image, template, type_of_img):
    # Load the original image
    src_image = cv2.imread(path_to_img)

    # Open the jason markup file, and extract the exact coordinates of the passport
    f = open(path_to_json_file, "r")
    markup = json.load(f)
    f.close()

    # Load the coordinates of the ROI of the passport/ID
    try:
        key_to_lst = None
        for key in markup["document"]["templates"]:
            key_to_lst = key
        coords = markup["document"]["templates"][key_to_lst]["template_quad"]
    except Exception as e:
        print(e)
        print(path_to_json_file)
        return

    # Display the result
    if not Path(path_to_save_new_image).exists():
        # Create a np.array for the coordinates of the ROI
        src_points = np.array(coords)

        #INSERT HERE : Template's height and width
        h, w = template.shape[:2]
        #print(f"height: {h}, width:{w}")
        # The destination image
        dst_points = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)

        # Get the homography matrix
        h, _ = cv2.findHomography(src_points, dst_points)

        # THIS IS BECAUSE I GET AN ERROR with H, W
        dst_w, dst_h = None, None
        if type_of_img == "ID":
            dst_w = 421
            dst_h = 265
        else:
            dst_w = 615
            dst_h = 432
        # Apply the transformation
        rectangular_canvas = cv2.warpPerspective(src_image, h, (dst_w, dst_h))

        # Save the new image
        cv2.imwrite(path_to_save_new_image, rectangular_canvas)


def apply_homography_to_dataset(path_to_project):
    # Read all the files of images
    p = pathlib.Path(path_to_project + "/dataset/images")
    images = list(p.rglob("*.jpg"))

    # Iterate over the images to mask their background
    for i in range(0, len(images)):
        #print(f"{i}/{len(images)}")
        path_to_image = images[i]

        # -------------- NAME OF THE CORRESPONDING JSON FILE ----------------------------
        # To find the corresponding json file to our current image, we just need the latter's path
        # By adding json extension to the name of the file and change images to markup
        # This will lead us to find json file, since the dataset is well-structured and ordered

        path_element = str(path_to_image).split("\\")  # Convert os.path to str and break it into elements
        # Add json extension to the last element
        path_element[len(path_element) - 1] = path_element[len(path_element) - 1] + ".json"
        # find the extension where "images" is
        index_images = path_element.index("images")
        # change it to markup to go to the json files folder instead of images
        path_element[index_images] = 'markup'
        # Reconstitute the new path as a string
        path_to_json_file = "".join(path_element[i] + "\\" for i in range(0, len(path_element) - 1))
        path_to_json_file = path_to_json_file + path_element[len(path_element) - 1]

        # If we do find the jason file corresponding to our image, we mask the background and save the new image
        if Path(path_to_json_file).exists():

            # Determine the file on which the new image will be saved
            # A WHOLE NEW DATASET IS CREATED UNDER THE NAME OF "new_dataset"
            new_path_img = str(path_to_image).split("\\dataset\\")

            new_path = os.path.join(path_to_project, "new_dataset", new_path_img[1])
            head_path = os.path.split(new_path)[0]

            # Create the directories if they didn't exist already
            if not os.path.isdir(head_path):
                os.makedirs(head_path)

            # Choose the template, either ID or passport
            temp = None
            type_of_img = None
            if is_folder_in_path_string("ID", new_path):
                path_to_temp = path_to_project+"/new_dataset/templates/hologram_masks/id_hologram_mask.png"
                temp = cv2.imread(path_to_temp)
                type_of_img = "ID"
            else:
                path_to_temp = path_to_project + "/new_dataset/templates/hologram_masks/passport_hologram_mask.png"
                temp = cv2.imread(path_to_temp)
                type_of_img = "PASSPORT"

            # Create the new image (erased background)+ save it under the pathname specified in new_path
            try:
                apply_homography(str(path_to_image), path_to_json_file, new_path, template=temp, type_of_img=type_of_img)
                #print(path_to_image)
            except Exception as e:
                print(e)
                print(path_to_image)

        else:
            print("------------------------------------ Could NOT FIND JSON FILE!!!!!!!!!!!-------------")
            print(path_to_image)
            continue


def adjust_templates(path_to_project):
    mask_id_path = path_to_project + "/dataset/templates/hologram_masks/id_hologram_mask.png"
    mask_id = cv2.imread(mask_id_path)
    mask_passport_path = path_to_project + "/dataset/templates/hologram_masks/passport_hologram_mask.png"
    mask_passport = cv2.imread(mask_passport_path)

    # APPLY DILATION
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((5, 5), np.uint8)
    mask_id = cv2.dilate(mask_id, kernel)
    mask_passport = cv2.dilate(mask_passport, kernel)

    # REDUCE THE DIMENSIONS
    height, width, clr_channels = mask_id.shape
    mask_id = cv2.resize(mask_id, (int(width/8), int(height/8)))
    height, width, clr_channels = mask_passport.shape
    mask_passport = cv2.resize(mask_passport, (int(width / 8), int(height / 8)))

    # THIS IS WHERE THE NEW MASKS ARE SAVED
    new_path = pathlib.Path(path_to_project+"/new_dataset/templates/hologram_masks")
    # Create the directories if they didn't exist already
    if not os.path.isdir(new_path):
        os.makedirs(new_path)

    # SAVE THE NEW MASKS
    new_path = path_to_project+"/new_dataset/templates/hologram_masks/id_hologram_mask.png"
    cv2.imwrite(new_path, mask_id)
    new_path = path_to_project + "/new_dataset/templates/hologram_masks/passport_hologram_mask.png"
    cv2.imwrite(new_path, mask_passport)


if __name__ == "__main__":
    adjust_templates("C:\\paris_cite\\m1\\S2\\TER")
    apply_homography_to_dataset("C:\\paris_cite\\m1\\S2\\TER")

