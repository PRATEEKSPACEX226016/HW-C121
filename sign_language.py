import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Define fingertip and thumb tip landmarks
finger_tips = [8, 12, 16, 20]  # Indices of fingertips in the landmark list
thumb_tip = 4  # Index of thumb tip in the landmark list

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)  # Flip horizontally for intuitive display
    h, w, c = img.shape
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert to RGB for Mediapipe

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            # Draw landmarks and connections
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec((0, 0, 255), 2, 2),
                                   mp_draw.DrawingSpec((0, 255, 0), 4, 2))

            # Draw blue circles around fingertips and check finger fold status
            finger_fold_status = []
            for tip_id in finger_tips:
                tip_x = int(lm_list[tip_id].x * w)
                tip_y = int(lm_list[tip_id].y * h)
                cv2.circle(img, (tip_x, tip_y), 10, (255, 0, 0), -1)

                # Check if finger is folded
                if tip_x < int(lm_list[tip_id - 1].x * w):  # Compare with previous landmark
                    cv2.circle(img, (tip_x, tip_y), 10, (0, 255, 0), -1)
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            # Check thumb gesture for LIKE or DISLIKE
            thumb_tip_y = int(lm_list[thumb_tip].y * h)
            if thumb_tip_y < int(lm_list[thumb_tip - 1].y * h):  # Thumb raised up
                cv2.putText(img, "LIKE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif thumb_tip_y > int(lm_list[thumb_tip - 1].y * h):  # Thumb lowered
                cv2.putText(img, "DISLIKE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
