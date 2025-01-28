import cv2
import mediapipe as mp
import numpy as np
import time
from tensorflow.keras.models import load_model

HAND_STATES = {0: "rock", 1: "paper", 2: "scissors"}

model = load_model('rock_paper_scissors_cnn.h5')
# Mediapipe settings
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def determine_winner(player1_choice, player2_choice):
    if player1_choice == player2_choice:
        return 0  # Draw
    elif (player1_choice == 0 and player2_choice == 2) or \
            (player1_choice == 1 and player2_choice == 0) or \
            (player1_choice == 2 and player2_choice == 1):
        return 1  # Player 1 wins
    else:
        return 2  # Player 2 wins

def predict_state(hand_image):
    hand_image = cv2.resize(hand_image, (300, 300))
    if len(hand_image.shape) == 3:
        hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)
    hand_image = hand_image.reshape(1, 300, 300, 1).astype('float32') / 255
    prediction = model.predict(hand_image)
    return np.argmax(prediction, axis=1)[0]


def wait_for_both_players_to_get_ready(cap, hands, frame_height, frame_width):
    player1_choice, player2_choice = None, None
    player1_moves, player2_moves = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        both_ready = False
        if results.multi_hand_landmarks:
            hand_states = []
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                landmarks = hand_landmarks.landmark
                hand_state = estimate_hand_state(landmarks)
                hand_states.append(hand_state)

                # Draw hand annotations and green bounding box
                x_min = int(min([landmark.x for landmark in landmarks]) * frame_width)
                y_min = int(min([landmark.y for landmark in landmarks]) * frame_height)
                x_max = int(max([landmark.x for landmark in landmarks]) * frame_width)
                y_max = int(max([landmark.y for landmark in landmarks]) * frame_height)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                if idx == 0:
                    player1_choice = hand_state
                    player1_moves.append(hand_state)  # Track Player 1 moves
                    label = f"Player 1: {HAND_STATES.get(hand_state, 'Unknown')}"
                else:
                    player2_choice = hand_state
                    player2_moves.append(hand_state)  # Track Player 2 moves
                    label = f"Player 2: {HAND_STATES.get(hand_state, 'Unknown')}"

                cv2.putText(frame, label, (10, 50 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if len(hand_states) == 2 and all(state == 0 for state in hand_states):  # Both players in ROCK
                both_ready = True

        cv2.putText(frame, "Waiting for both players to show ROCK position...",
                    (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow('Rock Paper Scissors', frame)

        if both_ready:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return player1_choice, player2_choice, player1_moves, player2_moves


def countdown_before_round(cap, hands, frame_height, frame_width):
    countdown_cheat1 = False
    countdown_cheat2 = False
    prev_positions = {0: None, 1: None}  # Track previous positions for each player

    for countdown in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)


        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                landmarks = hand_landmarks.landmark
                current_position = np.array([[lmk.x, lmk.y, lmk.z] for lmk in landmarks])

                # Track hand movement
                if prev_positions[idx] is not None:
                    displacement = np.linalg.norm(current_position - prev_positions[idx], axis=1).mean()
                    if displacement < 0.02:  # Threshold for minimal movement
                        if idx == 0:
                            countdown_cheat1 = True
                        elif idx == 1:
                            countdown_cheat2 = True

                prev_positions[idx] = current_position

        cv2.putText(frame, f"{countdown}",
                    (frame_width // 2 - 20, frame_height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    4, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.imshow('Rock Paper Scissors', frame)
        cv2.waitKey(1000)

    return countdown_cheat1, countdown_cheat2


def play_round_logic(cap, hands, frame_height, frame_width, player1_moves, player2_moves, player1_score, player2_score):
    player1_choice, player2_choice = None, None
    start_time = time.time()

    while time.time() - start_time < 5:  # 5 seconds per turn
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                landmarks = hand_landmarks.landmark
                hand_state = estimate_hand_state(landmarks)

                if idx == 0:
                    player1_choice = hand_state
                    player1_moves.append(hand_state)  # Track Player 1 moves
                    label = f"Player 1: {HAND_STATES.get(hand_state, 'Unknown')}"
                else:
                    player2_choice = hand_state
                    player2_moves.append(hand_state)  # Track Player 2 moves
                    label = f"Player 2: {HAND_STATES.get(hand_state, 'Unknown')}"

                # Draw hand annotations and green bounding box
                x_min = int(min([landmark.x for landmark in landmarks]) * frame_width)
                y_min = int(min([landmark.y for landmark in landmarks]) * frame_height)
                x_max = int(max([landmark.x for landmark in landmarks]) * frame_width)
                y_max = int(max([landmark.y for landmark in landmarks]) * frame_height)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                cv2.putText(frame, label, (10, 50 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display countdown and scores
        cv2.putText(frame, "Time Left: {:.1f}s".format(5 - (time.time() - start_time)),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Player 1 Score: {player1_score}", (10, frame_height - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"Player 2 Score: {player2_score}", (10, frame_height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('Rock Paper Scissors', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return player1_choice, player2_choice

def estimate_hand_state(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    thumb_mcp = landmarks[2]
    palm_base = landmarks[0]  # Wrist/palm base

    def distance(point1, point2):
        return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5

    # Measure distances to detect if fingers are folded or extended
    thumb_folded = distance(thumb_tip, palm_base) < distance(thumb_mcp, palm_base) * 0.8
    index_extended = distance(index_tip, palm_base) > distance(thumb_mcp, palm_base) * 1.5
    middle_extended = distance(middle_tip, palm_base) > distance(thumb_mcp, palm_base) * 1.5
    ring_folded = distance(ring_tip, palm_base) < distance(thumb_mcp, palm_base) * 1.1
    pinky_folded = distance(pinky_tip, palm_base) < distance(thumb_mcp, palm_base) * 1.1

    if all([thumb_folded, not index_extended, not middle_extended, ring_folded, pinky_folded]):
        return 0
    elif all([not thumb_folded, index_extended, middle_extended, not ring_folded, not pinky_folded]):
        return 1
    elif all([not thumb_folded, index_extended, middle_extended, ring_folded, pinky_folded]):
        return 2
    return 0
#
# def display_red_and_emoji_on_faces(frame, emoji_path='emoji.png'):
#     # Load the emoji image
#     emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)  # Attempt to load with alpha channel
#     if emoji is None:
#         raise FileNotFoundError(f"Emoji image not found at '{emoji_path}'. Ensure the path is correct and the image exists.")
#
#     # Load the Haar Cascade for face detection
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#     # Process each face
#     for idx, (x, y, w, h) in enumerate(faces):
#         if idx == 0:
#             # Apply a red mask to the first detected face
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), -1)
#         elif idx == 1:
#             # Resize the emoji to match the second face dimensions
#             resized_emoji = cv2.resize(emoji, (w, h))
#
#             # Create a circular mask for the emoji
#             mask = np.zeros((h, w), dtype=np.uint8)
#             center = (w // 2, h // 2)
#             radius = min(w, h) // 2
#             cv2.circle(mask, center, radius, 255, -1)
#
#             # If the emoji has an alpha channel
#             if resized_emoji.shape[2] == 4:
#                 emoji_alpha = resized_emoji[:, :, 3] / 255.0
#                 emoji_rgb = resized_emoji[:, :, :3]
#             else:
#                 emoji_alpha = np.ones((h, w), dtype=np.float32)
#                 emoji_rgb = resized_emoji
#
#             # Combine the circular mask with the alpha channel
#             combined_alpha = emoji_alpha * (mask / 255.0)
#
#             # Overlay the emoji onto the frame
#             for c in range(3):  # Loop over color channels
#                 frame[y:y+h, x:x+w, c] = (
#                     combined_alpha * emoji_rgb[:, :, c] + (1 - combined_alpha) * frame[y:y+h, x:x+w, c]
#                 )
#
#     return frame

def display_red_and_emoji_on_faces(frame, emoji_path='emoji.png'):
    # Load the emoji image
    emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)  # Attempt to load with alpha channel
    if emoji is None:
        raise FileNotFoundError(f"Emoji image not found at '{emoji_path}'. Ensure the path is correct and the image exists.")

    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) < 2:
        print("Less than two faces detected. Cannot apply both effects.")
        return frame

    # Sort faces by their x-coordinates (left to right)
    faces = sorted(faces, key=lambda f: f[0])

    # Get the left face and right face
    (x_left, y_left, w_left, h_left) = faces[0]  # Face on the left
    (x_right, y_right, w_right, h_right) = faces[1]  # Face on the right

    # Apply red mask to the left face
    cv2.rectangle(frame, (x_left, y_left), (x_left + w_left, y_left + h_left), (0, 0, 255), -1)

    # Resize the emoji to match the right face dimensions
    resized_emoji = cv2.resize(emoji, (w_right, h_right))

    # Create a circular mask for the emoji
    mask = np.zeros((h_right, w_right), dtype=np.uint8)
    center = (w_right // 2, h_right // 2)
    radius = min(w_right, h_right) // 2
    cv2.circle(mask, center, radius, 255, -1)

    # If the emoji has an alpha channel
    if resized_emoji.shape[2] == 4:
        emoji_alpha = resized_emoji[:, :, 3] / 255.0
        emoji_rgb = resized_emoji[:, :, :3]
    else:
        emoji_alpha = np.ones((h_right, w_right), dtype=np.float32)
        emoji_rgb = resized_emoji

    # Combine the circular mask with the alpha channel
    combined_alpha = emoji_alpha * (mask / 255.0)

    # Overlay the emoji onto the right face
    for c in range(3):  # Loop over color channels
        frame[y_right:y_right+h_right, x_right:x_right+w_right, c] = (
            combined_alpha * emoji_rgb[:, :, c] + (1 - combined_alpha) * frame[y_right:y_right+h_right, x_right:x_right+w_right, c]
        )

    return frame

def display_red_mask_on_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), -1)  # Draw a filled red rectangle

    return frame

def display_emoji_on_faces(frame, emoji_path='emoji.png'):
    # Load the emoji image
    emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)  # Attempt to load with alpha channel
    if emoji is None:
        raise FileNotFoundError(f"Emoji image not found at '{emoji_path}'. Ensure the path is correct and the image exists.")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Resize the emoji to match the face dimensions
        resized_emoji = cv2.resize(emoji, (w, h))

        # Create a circular mask for the emoji
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        radius = min(w, h) // 2
        cv2.circle(mask, center, radius, 255, -1)

        # If the emoji has an alpha channel
        if resized_emoji.shape[2] == 4:
            emoji_alpha = resized_emoji[:, :, 3] / 255.0
            emoji_rgb = resized_emoji[:, :, :3]
        else:
            emoji_alpha = np.ones((h, w), dtype=np.float32)
            emoji_rgb = resized_emoji

        # Combine the circular mask with the alpha channel
        combined_alpha = emoji_alpha * (mask / 255.0)

        # Overlay the emoji onto the frame
        for c in range(3):  # Loop over color channels
            frame[y:y+h, x:x+w, c] = (
                combined_alpha * emoji_rgb[:, :, c] + (1 - combined_alpha) * frame[y:y+h, x:x+w, c]
            )

    return frame