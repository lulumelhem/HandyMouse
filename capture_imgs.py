import cv2

webcam = cv2.VideoCapture(2)
ges = 1
var = 1
count = 0;

while(1):

    (grabbed, frame) = webcam.read()
    frame = cv2.flip(frame, 1)
    roi = frame[100:600, 400:700]

    cv2.rectangle(frame, (700, 100), (400, 600), (0, 255, 0), 2)
    cv2.putText(frame, "Press space to begin", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "Gesture: " + str(ges) + " Var: " + str(var), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow("img", frame)

    buttonPressed = cv2.waitKey(1)

    # if letter 'n' pressed get the background
    if buttonPressed == 110:
        for i in range(1, 31):

            (grabbed, frame) = webcam.read()
            frame = cv2.flip(frame, 1)

            cv2.imwrite('images/background_' + str(i) + '.png', roi)


    # if spacebar pressed capture images of gesture
    if buttonPressed == 32:

        for i in range(1, 11):

            (grabbed, frame) = webcam.read()
            frame = cv2.flip(frame, 1)

            if ges == 1:
                cv2.imwrite('images/up_' + str(i+count) + '.png', roi)
            elif ges == 2:
                cv2.imwrite('images/left_' + str(i+count) + '.png', roi)
            elif ges == 3:
                cv2.imwrite('images/down_' + str(i+count) + '.png', roi)
            elif ges == 4:
                cv2.imwrite('images/right_' + str(i+count) + '.png', roi)

        if var > 9:
            ges += 1
            var = 1
            count = 0
        else:
            var += 1
            count += 10

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        cv2.destroyAllWindows()
        break

webcam.release()