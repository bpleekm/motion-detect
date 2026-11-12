"""Explicit CPU reference of the migrated MOG/Motion/spatial semantics."""
import math
import cv2
import numpy as np


def area(reference, width, height):
    return max(1, math.floor(reference * width * height / (768 * 432) + 0.5))


def denoise(raw, strength):
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    value = cv2.erode(raw, rect, iterations=strength, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    value = cv2.erode(value, cross, iterations=strength, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    value = cv2.dilate(value, cross, iterations=strength, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.dilate(value, rect, iterations=strength, borderType=cv2.BORDER_CONSTANT, borderValue=0)


def components(mask, minimum, strict=False):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4, ltype=cv2.CV_32S)
    accepted = np.zeros(count, np.uint8)
    regions = []
    for label in range(1, count):
        x, y, w, h, size = map(int, stats[label])
        if size > minimum if strict else size >= minimum:
            accepted[label] = 255
            regions.append({"box": [x, y, w, h], "area": size})
    regions.sort(key=lambda r: (-r["area"], r["box"]))
    return accepted[labels], regions


class ReferenceCore:
    """Single ordered state per input session; CPU double feedback is labelled."""

    def __init__(self, width, height, watermark=(0, 0, 0, 0)):
        cv2.setNumThreads(2)
        cv2.ocl.setUseOpenCL(False)
        self.width, self.height = width, height
        self.watermark = watermark
        self.shape = ((height + 7) // 8 * 8, (width + 7) // 8 * 8)
        self.mog = cv2.createBackgroundSubtractorMOG2(150, 16, True)
        self.reference = None
        self.age = np.zeros(self.shape, np.uint8)
        self.noise = 32.0
        self.updates = 0
        self.low, self.high = area(50, width, height), area(100, width, height)

    def process(self, gray):
        if gray.shape != (self.height, self.width):
            raise ValueError("visible geometry changed")
        content = cv2.GaussianBlur(gray, (9, 9), 0, borderType=cv2.BORDER_REFLECT_101)
        x, y, w, h = self.watermark
        content[y:y+h, x:x+w] = 0
        prepared = np.zeros(self.shape, np.uint8)
        prepared[:self.height, :self.width] = content
        self.updates += 1
        foreground = self.mog.apply(prepared, learningRate=1 if self.updates == 1 else .006644493744965563)
        mog_raw = np.uint8(foreground[:self.height, :self.width] == 255) * 255
        mog_raw[y:y+h, x:x+w] = 0
        before = self.noise
        raw = np.zeros((self.height, self.width), np.uint8)
        offset = None
        if self.reference is None:
            self.reference = prepared.copy()
            mog_raw[:] = 0
        else:
            residual = prepared.astype(np.int16) - self.reference.astype(np.int16)
            difference = np.abs(residual)
            active = difference > before
            self.age[active] = np.minimum(self.age[active].astype(np.uint16) + 1, 255).astype(np.uint8)
            self.age[~active] = 0
            low = difference <= max(1, int(before * .2))
            partial = ~active & ~low
            self.reference[low] = prepared[low]
            self.reference[partial] = ((self.reference[partial].astype(np.uint16) + prepared[partial]) // 2).astype(np.uint8)
            absorbed = self.age >= 50
            self.reference[absorbed] = prepared[absorbed]
            self.age[absorbed] = 0
            if self.updates % 5 == 0:
                quiet = difference[~active]
                if quiet.size and int(np.count_nonzero(active)) < self.high:
                    self.noise = float(np.clip(.8 * before + .2 * (4 + 2 * float(quiet.mean())), 12, 48))
            visible = residual[:self.height, :self.width]
            offset = float(np.median(visible[::16, ::16]))
            raw = np.uint8(np.abs(visible.astype(np.float32) - offset) > np.float32(before)) * 255
            raw[y:y+h, x:x+w] = 0
        _, mog_regions = components(denoise(mog_raw, 1), self.low)
        motion_low, low_regions = components(denoise(raw, 1), self.low)
        _, high_regions = components(denoise(raw, 2) & motion_low, self.high, True)
        return {"mog_pass": bool(mog_regions), "motion_pass": bool(high_regions),
                "has_candidate": bool(mog_regions or low_regions), "warming": self.updates <= 5,
                "regions": {"mog": mog_regions, "motion_low": low_regions, "motion_high": high_regions},
                "model": {"noise_used": before, "noise_after": self.noise, "offset": offset}}
