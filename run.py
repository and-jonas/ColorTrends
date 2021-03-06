
# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Project: LesionZoo
# Date: 15.03.2021
# Sample training data blue background removal
# ======================================================================================================================


from roi_selector import TrainingPatchSelector


def run():
    dir_to_process = "Z:/Public/Jonas/003_ESWW/ColorTrends/testing_segmentation_outdoor/cropped"
    dir_positives = "Z:/Public/Jonas/003_ESWW/ColorTrends/testing_segmentation_outdoor/foreground"
    dir_negatives = "Z:/Public/Jonas/003_ESWW/ColorTrends/testing_segmentation_outdoor/background"
    dir_control = "Z:/Public/Jonas/003_ESWW/ColorTrends/testing_segmentation_outdoor/checkers"
    roi_selector = TrainingPatchSelector(dir_to_process, dir_positives, dir_negatives, dir_control)
    roi_selector.iterate_images()


if __name__ == '__main__':
    run()
