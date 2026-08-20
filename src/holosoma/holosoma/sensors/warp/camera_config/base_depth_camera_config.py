from dataclasses import MISSING


class BaseSensorConfigCamera:
    num_sensors = MISSING
    sensor_type = MISSING

    randomize_placement = False
    min_translation = MISSING
    max_translation = MISSING
    min_euler_rotation_deg = MISSING
    max_euler_rotation_deg = MISSING


class BaseDepthCameraConfig(BaseSensorConfigCamera):
    num_sensors = MISSING  # number of sensors of this type

    sensor_type = "camera"  # sensor type

    # camera params VFOV is calcuated from the aspect ratio and HFOV
    # VFOV = 2 * atan(tan(HFOV/2) / aspect_ratio)

    height = MISSING
    width = MISSING
    horizontal_fov_deg = MISSING
    max_range = MISSING
    min_range = MISSING

    # Border crop applied before resizing the depth image. Keeping this with
    # the camera geometry prevents a generic observation term from silently
    # hard-coding one sensor's field of view.
    crop_top = 0
    crop_bottom = 0
    crop_left = 0
    crop_right = 0

    # Type of camera (depth, range, pointcloud, segmentation)
    # You can combine: (depth+segmentation), (range+segmentation), (pointcloud+segmentation)
    # Other combinations are trivial and you can add support for them in the code if you want.

    calculate_depth = True  # Get a depth image and not a range image. False will result in a range image
    return_pointcloud = (
        False  # Return a pointcloud instead of an image. Above depth option will be ignored if this is set to True
    )
    pointcloud_in_world_frame = False
    segmentation_camera = False

    # transform from sensor element coordinate frame to sensor_base_link frame
    base_offset_pos = MISSING
    base_offset_rot = MISSING

    # randomize placement of the sensor
    randomize_placement = False
    min_translation = MISSING
    max_translation = MISSING
    min_euler_rotation_deg = MISSING
    max_euler_rotation_deg = MISSING
