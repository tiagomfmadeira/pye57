import numpy as np
import pye57
import cv2
from scipy.spatial.transform import Rotation

if __name__ == "__main__":

    e57 = pye57.E57("datasets/file.e57")
    imf = e57.image_file
    root = imf.root()

    print("File loaded successfully!")
    if not root['images2D']:
        print("File contains no 2D images. Exiting...")

    for image_idx, image2D in enumerate(root['images2D']):
        print("\n\n#########################################")
        print("## Camera " + str(image_idx))

        # Get extrinsic matrix
        tx = image2D['pose']['translation']['x'].value()
        ty = image2D['pose']['translation']['y'].value()
        tz = image2D['pose']['translation']['z'].value()

        t = np.array([tx, ty, tz])

        rx = image2D['pose']['rotation']['x'].value()
        ry = image2D['pose']['rotation']['y'].value()
        rz = image2D['pose']['rotation']['z'].value()
        rw = image2D['pose']['rotation']['w'].value()

        r = Rotation.from_quat(np.array([rx, ry, rz, rw]))

        cam_matrix = np.zeros((4, 4))
        cam_matrix[3, 3] = 1
        cam_matrix[:-1, -1] = t
        cam_matrix[:3, :3] = r.as_matrix()

        print('\nCamera matrix= ')
        print(cam_matrix)

        # Get intrinsic matrix
        pinhole = image2D['pinholeRepresentation']

        focal_length = pinhole['focalLength'].value()
        principal_point_x = pinhole['principalPointX'].value()
        principal_point_y = pinhole['principalPointY'].value()

        K = np.zeros((3, 3))
        K[2, 2] = 1
        K[0, 0] = focal_length
        K[1, 1] = focal_length
        K[0, 2] = principal_point_x
        K[1, 2] = principal_point_y

        print('\nK= ')
        print(K)

        # Get picture from blob
        jpeg_image = pinhole['jpegImage']
        jpeg_image_data = np.zeros(shape=jpeg_image.byteCount(), dtype=np.uint8)
        jpeg_image.read(jpeg_image_data, 0, jpeg_image.byteCount())
        image = cv2.imdecode(jpeg_image_data, cv2.IMREAD_COLOR)

        cv2.namedWindow("Image2D", cv2.WINDOW_NORMAL)
        cv2.imshow("Image2D", image)

        print("\n\nUsage: ")
        print("\t's':\t\tSave current image")
        print("\t'spce':\t\tNext image")
        print("\t'q':\t\tExit")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                cv2.imwrite('images2d' + str(image_idx) + '.jpg', image)
            if key == ord(" "):
                break
            if key == ord("q"):
                exit()
