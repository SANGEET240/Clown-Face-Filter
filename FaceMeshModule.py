import cv2
import mediapipe as mp
import time


class FaceMeshDetector:
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode,
                                                max_num_faces=self.maxFaces,
                                                min_detection_confidence=self.minDetectionCon,
                                                min_tracking_confidence=self.minTrackCon
                                                )

        self.drawSpec = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        self.drawSpec2 = self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)

        # For FPS calculation
        self.previousTime = 0
        self.currentTime = 0

    def findFaceMesh(self, img, draw=True):
        """Finds face mesh in the given image and returns faces with landmarks."""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                               self.drawSpec, self.drawSpec2)

                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([id, x, y])
                faces.append(face)

        return img, faces

    def calculateFPS(self, img, draw=True):
        """Calculates and returns FPS, also draws it if draw=True."""
        self.currentTime = time.time()
        fps = 1 / (self.currentTime - self.previousTime) if (self.currentTime - self.previousTime) != 0 else 0
        self.previousTime = self.currentTime

        if draw:
            cv2.putText(img, f'FPS: {int(fps)}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 2)
        return img, fps


def main():
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)

        # Show FPS
        img, fps = detector.calculateFPS(img)

        cv2.imshow("FaceMesh", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
