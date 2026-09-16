import argparse
import cv2


def detector():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(
        cv2.HOGDescriptor_getDefaultPeopleDetector()
    )
    return hog


def detect_image(path, output="detected.png"):
    img = cv2.imread(path)

    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    hog = detector()

    boxes, weights = hog.detectMultiScale(
        img,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )

    for (x, y, w, h), weight in zip(boxes, weights):
        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            "Pedestrian",
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imwrite(output, img)

    print("Saved:", output)
    print("Detections:", len(boxes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="detected.png")

    args = parser.parse_args()

    detect_image(args.image, args.output)