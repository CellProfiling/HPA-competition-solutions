import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np


class Gamma(iaa.Augmenter):
    """
    Adjust gamma for all pixels in an image with a specific value.

    This augmenter can be used to make images lighter or darker.

    Parameters
    ----------
    mul : float or tuple of two floats or StochasticParameter, optional(default=1.0)
        The value with which to adjust the pixel values gamma in each
        image.
            * If a float, then that value will always be used.
            * If a tuple (a, b), then a value from the range a <= x <= b will
              be sampled per image and used for all pixels.
            * If a StochasticParameter, then that parameter will be used to
              sample a new value per image.

    per_channel : bool or float, optional(default=False)
        Whether to use the same multiplier per pixel for all channels (False)
        or to sample a new value for each channel (True).
        If this value is a float p, then for p percent of all images
        `per_channel` will be treated as True, otherwise as False.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.Gamma(1.0)

    would adjust all images gamma by a factor of 2**1, making the images
    significantly brighter.

    >>> aug = iaa.Gamma((-0.5, 0.5))

    would adjust images gamma by a random value from the range 2**-0.5 <= x <= 2**0.5,
    making some images darker and others brighter.

    """

    def __init__(self, gamma_log2=0.0, per_channel=False, name=None,
                 deterministic=False, random_state=None):
        super(Gamma, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(gamma_log2):
            self.gamma_log2 = iaa.Deterministic(gamma_log2)
        elif ia.is_iterable(gamma_log2):
            assert len(gamma_log2) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(gamma_log2),)
            self.gamma_log2 = iaa.Uniform(gamma_log2[0], gamma_log2[1])
        elif isinstance(gamma_log2, iaa.StochasticParameter):
            self.gamma_log2 = gamma_log2
        else:
            raise Exception("Expected float or int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(gamma_log2),))

        if per_channel in [True, False, 0, 1, 0.0, 1.0]:
            self.per_channel = iaa.Deterministic(int(per_channel))
        elif ia.is_single_number(per_channel):
            assert 0 <= per_channel <= 1.0
            self.per_channel = iaa.Binomial(per_channel)
        else:
            raise Exception("Expected per_channel to be boolean or number or StochasticParameter")

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i in ia.sm.xrange(nb_images):
            image = images[i].astype(np.float32)
            rs_image = ia.new_random_state(seeds[i])
            per_channel = self.per_channel.draw_sample(random_state=rs_image)
            if per_channel == 1:
                nb_channels = image.shape[2]
                samples = self.gamma_log2.draw_samples((nb_channels,), random_state=rs_image)
                for c, sample in enumerate(samples):
                    image[..., c] = np.power(image[..., c]/255.0,  2.0**sample)*255.0
                # np.clip(image, 0, 255, out=image)
                result[i] = image.astype(np.float32)
            else:
                sample = self.gamma_log2.draw_sample(random_state=rs_image)
                image = np.power(image/255.0,  2.0**sample)*255.0
                # np.clip(image, 0, 255, out=image)
                result[i] = image.astype(np.float32)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.gamma_log2]
