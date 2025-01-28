import cv2
import mediapipe as mp
import numpy as np
import time
from tensorflow.keras.models import load_model

from utilities import estimate_hand_state

# Hand states map
HAND_STATES = {0: "rock", 1: "paper", 2: "scissors"}

model = load_model('rock_paper_scissors_model.h5')

def predict_state(hand_image):
    hand_image = cv2.resize(hand_image, (300, 300))
    if len(hand_image.shape) == 3:
        hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)
    hand_image = hand_image.reshape(1, 300, 300, 1).astype('float32') / 255
    prediction = model.predict(hand_image)
    return np.argmax(prediction, axis=1)[0]

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

def detect_hands():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as hands:
        player1_score, player2_score = 0, 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Cannot read from the camera.")
                break

            frame_height, frame_width, _ = frame.shape

            for i in range(150):  # Display pos message
                ret, frame = cap.read()
                if not ret:
                    break

                cv2.putText(frame, "Place your hands in the ROCK position!",
                            (10, frame_height // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, "Prepare to Play!",
                            (10, frame_height // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('Rock Paper Scissors', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            #1) Wait until both players have their hands in the ROCK position /////////////////////////////////////
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

            #2) Countdown before the round begins ///////////////////////////////////////////////////////////
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

            if countdown_cheat1:
                print("Player 1 cheated during countdown!")
                player1_score -= 1
            if countdown_cheat2:
                print("Player 2 cheated during countdown!")
                player2_score -= 1

            #3) Round logic /////////////////////////////////////////////////////////////////////////////////////////
            player1_choice, player2_choice = None, None
            player1_moves, player2_moves = [], []  # Track moves during the round
            start_time = time.time()

            while time.time() - start_time < 7:  # 5 seconds per turn
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
            if player1_score == 5 or player2_score == 5:
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
