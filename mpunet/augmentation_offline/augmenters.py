import random
from .augment_funcs import flip, rotate, blur, add_noise, translate


class Flipping:
    def __init__(self, apply_prob=1.0):
        if apply_prob > 1 or apply_prob < 0:
            raise ValueError("Apply probability is invalid with value %3.f" % apply_prob)
        self.apply_prob = apply_prob
        self.__name__ = "Flipping"

    def __str__(self):
        return "%s(apply_prob=%.3f)" % (
            self.__name__, self.apply_prob
        )

    def __repr__(self):
        return str(self)

    def __call__(self, image, label):
        is_apply = random.random() < self.apply_prob
        if is_apply:
            flip_axis = random.choice([0, 1, 2])
            image, label = flip(image, label, flip_axis)
        return image, label
    
class Rotation:
    def __init__(self, apply_prob=1.0):
        if apply_prob > 1 or apply_prob < 0:
            raise ValueError("Apply probability is invalid with value %3.f" % apply_prob)
        self.apply_prob = apply_prob
        self.__name__ = "Rotation"

    def __str__(self):
        return "%s(apply_prob=%.3f)" % (
            self.__name__, self.apply_prob
        )

    def __repr__(self):
        return str(self)

    def __call__(self, image, label):
        is_apply = random.random() < self.apply_prob
        if is_apply:
            rotate_axis = random.choice([0, 1, 2])
            rotate_angle = random.choice(range(-15, 15))
            image, label = rotate(image, label, rotate_axis, rotate_angle)
        return image, label
    
class Blur:
    def __init__(self, apply_prob=1.0):
        if apply_prob > 1 or apply_prob < 0:
            raise ValueError("Apply probability is invalid with value %3.f" % apply_prob)
        self.apply_prob = apply_prob
        self.__name__ = "Blur"

    def __str__(self):
        return "%s(apply_prob=%.3f)" % (
            self.__name__, self.apply_prob
        )

    def __repr__(self):
        return str(self)

    def __call__(self, image, sigma):
        is_apply = random.random() < self.apply_prob
        if is_apply:
            image = blur(image, sigma)
        return image
    
class Noise:
    def __init__(self, apply_prob=1.0):
        if apply_prob > 1 or apply_prob < 0:
            raise ValueError("Apply probability is invalid with value %3.f" % apply_prob)
        self.apply_prob = apply_prob
        self.__name__ = "Noise"

    def __str__(self):
        return "%s(apply_prob=%.3f)" % (
            self.__name__, self.apply_prob
        )

    def __repr__(self):
        return str(self)

    def __call__(self, image, mean=1.0, std=1.0):
        is_apply = random.random() < self.apply_prob
        if is_apply:
            image = add_noise(image, mean, std)
        return image
    
class Translation:
    def __init__(self, apply_prob=1.0):
        if apply_prob > 1 or apply_prob < 0:
            raise ValueError("Apply probability is invalid with value %3.f" % apply_prob)
        self.apply_prob = apply_prob
        self.__name__ = "Translation"

    def __str__(self):
        return "%s(apply_prob=%.3f)" % (
            self.__name__, self.apply_prob
        )

    def __repr__(self):
        return str(self)

    def __call__(self, image, label, max_trans=20):
        is_apply = random.random() < self.apply_prob
        if is_apply:
            translating_axis = random.choice([0, 1, 2])
            image, label = translate(image, label, translating_axis, max_trans)
        return image, label