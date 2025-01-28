import cv2
import mediapipe as mp
import numpy as np
import time
from tensorflow.keras.models import load_model
from utilities import estimate_hand_state, play_round_logic, countdown_before_round, wait_for_both_players_to_get_ready, determine_winner

# Hand states map
HAND_STATES = {0: "rock", 1: "paper", 2: "scissors"}

model = load_model('rock_paper_scissors_model.h5')

# Mediapipe settings
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

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

        # Check if the emoji has an alpha channel
        if resized_emoji.shape[2] == 4:
            # Extract the alpha channel for transparency
            emoji_alpha = resized_emoji[:, :, 3] / 255.0  # Normalize to 0-1 range
            emoji_rgb = resized_emoji[:, :, :3]  # Get the BGR channels
        else:
            # If no alpha channel, treat emoji as fully opaque
            emoji_alpha = np.ones((h, w))  # Fully opaque
            emoji_rgb = resized_emoji

        # Overlay the emoji onto the frame
        for c in range(3):  # Loop over color channels
            frame[y:y+h, x:x+w, c] = (
                emoji_alpha * emoji_rgb[:, :, c] + (1 - emoji_alpha) * frame[y:y+h, x:x+w, c]
            )
    return frame



def detect_hands():
    cap = cv2.VideoCapture(0)  # Open the camera
    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as hands:
        player1_score, player2_score = 0, 0
        frame_height, frame_width, _ = cap.read()[1].shape

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Cannot read from the camera.")
                break

            # Wait until both players are in position
            player1_choice, player2_choice, player1_moves, player2_moves = wait_for_both_players_to_get_ready(
                cap, hands, frame_height, frame_width)

            # Countdown before the round begins
            countdown_cheat1, countdown_cheat2 = countdown_before_round(cap, hands, frame_height, frame_width)

            if countdown_cheat1:
                print("Player 1 cheated during countdown!")
                player1_score -= 1
            if countdown_cheat2:
                print("Player 2 cheated during countdown!")
                player2_score -= 1

            # Round logic
            player1_choice, player2_choice = play_round_logic(
                cap, hands, frame_height, frame_width, player1_moves,
                player2_moves, player1_score, player2_score)

            # Detect cheating
            if len(set(player1_moves)) > 1:
                print("Player 1 cheated!")
                player1_score -= 1

            if len(set(player2_moves)) > 1:
                print("Player 2 cheated!")
                player2_score -= 1

            # Determine round winner
            if player1_choice is not None and player2_choice is not None:
                winner = determine_winner(player1_choice, player2_choice)
                if winner == 1 and len(set(player1_moves)) <= 1:
                    player1_score += 1
                elif winner == 2 and len(set(player2_moves)) <= 1:
                    player2_score += 1

            # End the game if a player reaches 5 points
            if player1_score < 1 or player2_score < 1:  # update later
                break

        cap.release()

        # Reinitialize VideoCapture for face masking
        cap = cv2.VideoCapture(0)
        print("Game Over! Displaying the red mask.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply the red mask to detected faces
            # frame_with_mask = display_red_mask_on_faces(frame)
            frame_with_mask = display_emoji_on_faces(frame, emoji_path='imoji.png')

            cv2.imshow("Game Over - Red Mask", frame_with_mask)


            # Exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Final result
        if player1_score == 5:
            print(f"Player 1 Wins! Final Scores - Player 1: {player1_score}, Player 2: {player2_score}")
        elif player2_score == 5:
            print(f"Player 2 Wins! Final Scores - Player 1: {player1_score}, Player 2: {player2_score}")

# Run the game
detect_hands()
