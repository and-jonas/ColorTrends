
from testing_color_checker import ColorCorrector


def run():
    base_dir = "Z:/Public/Jonas/003_ESWW/ColorTrends/Testing_color_corr/"
    output_dir = "Z:/Public/Jonas/003_ESWW/ColorTrends/Testing_color_corr/corrected_images"
    color_corrector = ColorCorrector(base_dir=base_dir,
                                     output_dir=output_dir)
    color_corrector.run()

if __name__ == '__main__':
    run()