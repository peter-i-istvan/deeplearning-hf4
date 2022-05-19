import cv2
import os
import os.path as osp
import shutil


def jpeg_compress_demo():
    img = cv2.imread("db/CUB_200_2011/images/009.Brewer_Blackbird/Brewer_Blackbird_0064_2290.jpg")
    cv2.imshow("Before", img)
    cv2.waitKey(0)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    decoded_img = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    cv2.imshow("After", decoded_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_train_test_split(db_path, images_foldername, train_quality, validation_quality):
    """
    db_path: ex. 'db/Birds'
    images_foldername: ex. 'images'
    train_quality: int between 0 and 100; train images are taken from 'images-{train_quality}', if exists
    validation_quality: int between 0 and 100; validation images are taken from 'images-{validation_quality}', if exists

    Result: a 'train-{train_quality}' and 'validation-{validation_quality}' folder is created in db_path
    """
    # Assertions:
    assert (isinstance(train_quality, int) and 0 <= train_quality <= 100), "Parameter test_quality must be an int"
    assert (isinstance(validation_quality, int) and 0 <= validation_quality <= 100), "Parameter validation_quality must be an int"
    # Build a LUT of image_id -> filename (excluding "db/Birds/images") associations inside 'filenames'
    filenames = [-1]  # indexing starts from 1
    with open(osp.join(db_path, "images.txt")) as f:
        lines = f.readlines()
        filenames.extend([l.split(" ")[1] for l in lines])
    # Build a LUT for image_id -> class_id
    class_ids = [-1]
    with open(osp.join(db_path, "image_class_labels.txt")) as f:
        lines = f.readlines()
        class_ids.extend([int(l.split(" ")[1]) for l in lines])
    # Source and destination folder paths:
    train_input_path = osp.join(db_path, f"{images_foldername}-{train_quality}")
    validation_input_path = osp.join(db_path, f"{images_foldername}-{validation_quality}")
    train_output_path = osp.join(db_path, f"train-{train_quality}")
    validation_output_path = osp.join(db_path, f"validation-{validation_quality}")
    if not osp.exists(train_output_path): os.makedirs(train_output_path)
    if not osp.exists(validation_output_path): os.makedirs(validation_output_path)
    # create class folders too:
    for i in range(1, 7):
        if not osp.exists(osp.join(train_output_path, f"{i}")):
            os.makedirs(osp.join(train_output_path, f"{i}"))
        if not osp.exists(osp.join(validation_output_path, f"{i}")):
            os.makedirs(osp.join(validation_output_path, f"{i}"))
    # Create train-test split:
    last_image_id = 323  # we only use the first 6 classes, so the last image ID is 323
    with open(osp.join(db_path, "train_test_split.txt")) as traintest_file:
        lines = traintest_file.readlines()
        for l in lines:
            image_id, is_train = l.split(" ")
            image_id = int(image_id)
            is_train = bool(is_train)
            if image_id > last_image_id:
                break
            if is_train:
                shutil.copyfile(
                    src=osp.join(train_input_path, images_foldername, filenames[image_id]),
                    dst=osp.join(train_output_path, f"{class_ids[image_id]}", filenames[image_id].split("/")[1])
                )
            else:
                shutil.copyfile(
                    src=osp.join(validation_input_path, images_foldername, filenames[image_id]),
                    dst=osp.join(validation_output_path, f"{class_ids[image_id]}", filenames[image_id].split("/")[1])
                )


def compress_db(quality):
    """
    Compresses the whole "Birds" database with the given quality
    :param quality: is passed to cv2.imencode as cv2.IMWRITE_JPEG_QUALITY parameter, int between 0 and 100
    """
    assert (isinstance(quality, int) and 0 <= quality <= 100), "Parameter quality must be an int"
    db_folder_path = "db"
    db_path = "Birds"
    images_foldername = f"images-{quality}"
    full_path = osp.join(db_folder_path, db_path, images_foldername)
    if not osp.exists(full_path):
        os.makedirs(full_path)


def main():
    # jpeg_compress_demo()
    train_quality = 100
    validation_quality = 100
    # compress_db(train_quality)
    # compress_db(validation_quality)
    create_train_test_split(
        db_path=osp.join("db", "Birds"),
        images_foldername="images",
        train_quality=train_quality,
        validation_quality=validation_quality
    )


if __name__ == "__main__":
    main()
