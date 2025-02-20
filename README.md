# Rock-Paper-Scissors Hand Recognition Game 🎮🤖

This project implements a Rock-Paper-Scissors game using computer vision. It detects hand gestures in real time using MediaPipe and a trained CNN model to classify hand states (rock, paper, or scissors). The game also includes a cheat detection mechanism to ensure fair play.
## Features

- Real-time hand tracking using MediaPipe
- Gesture classification using a CNN model trained on a custom dataset
- A countdown before each round, ensuring players place their hands in the correct position
- Cheat detection: If a player changes their hand position during a round, they lose a point
- Visual feedback with bounding boxes and labels for hand states

## Requirements

- Python 3.x
- OpenCV
- MediaPipe
- TensorFlow
- NumPy

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rock-paper-scissors
cd rock-paper-scissors
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the game:
```bash
python game.py
```

## How It Works

1. Before each round, a message appears asking players to place their hands in the "rock" position.
2. A countdown ensures both players maintain the correct position.
3. Players make their gestures (rock, paper, or scissors) when the round starts.
4. The program captures and classifies each player's gesture using the CNN model.
5. The winner is determined based on the game rules.
6. Cheat detection: If a player changes their hand position during the round, they lose a point.
7. The game continues until one player reaches 5 points.

## CNN Classifier

To improve accuracy, we trained a Convolutional Neural Network (CNN) on a custom dataset created with Roboflow. The dataset consists of images labeled with three hand states: rock, paper, and scissors.

### Model Training
The CNN model was trained using the following steps:
- Collected and preprocessed images from Roboflow
- Applied data augmentation to enhance generalization
- Used a deep learning model to classify hand states
- Achieved high accuracy on validation data

You can find the CNN training code in the classifier.ipynb file.

