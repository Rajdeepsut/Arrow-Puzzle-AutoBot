# 🏹 Arrow Puzzle Autonomous AI Solver (Easybrain)

A computer-vision and automated Android bot built with Python and OpenCV to automatically solve and clear levels (from Level 32 to 500+) in the mobile game **"Arrow Puzzle: Tap Puzzle Games"** by Easybrain.

---

## 🎮 How the Game Works
The game features a grid of directional arrows (UP, DOWN, LEFT, RIGHT):
- When tapped, an arrow flies forward along its pointing direction.
- **Goal:** Clear all arrows to make the screen completely blank.
- **Collision Rule:** If an arrow collides with another arrow in front of it, you lose a star.
- **Safe Rule:** An arrow is 100% safe to tap **only if** there are NO other arrows in front of it along its line of sight to the edge of the board.

---

## 🧠 AI Solver Algorithm (Ray-Casting)
`
[Capture Screen via USB] -> [Detect Grid & Arrow Directions] -> [Cast Rays along Directions]
                                                                        |
                                                 [Tap All Obstacle-Free Arrows Simultaneously]
                                                                        |
                                                    [Repeat Until Screen is Blank (Victory)]
`
1. **Screen Capture:** Captures high-res frames via wired USB using ADB.
2. **Ray-Casting:** For each arrow, shoots a ray in its direction. If the ray reaches the screen border without hitting another arrow, it's marked as SAFE.
3. **Automated Tap:** Dispatches instantaneous screen taps on all safe arrows.
4. **Autonomous Loop:** Automatically taps "Next Level" and loops continuously from Level 32 to 500+.

---

## 🚀 Getting Started

### 1. Requirements
- Python 3.8+
- An Android phone connected to your PC/Laptop via **Wired USB Cable**.

### 2. Enable USB Debugging on Your Phone
1. Go to **Settings > About Phone**.
2. Tap **Build Number** 7 times to enable **Developer Options**.
3. Go to **Settings > Developer Options** and enable **USB Debugging**.
4. Connect phone to laptop with a USB Data cable.
5. In the popup that appears on the phone screen, select **"Always allow from this computer"** and tap **Allow**.

### 3. Installation
Clone the repository:
`ash
git clone https://github.com/Rajdeepsut/Arrow-Puzzle-AutoBot.git
cd Arrow-Puzzle-AutoBot
pip install -r requirements.txt
`

### 4. Run the Bot
Open the Arrow Puzzle game on your mobile phone and run:
`ash
python arrow_puzzle_solver.py
`
Sit back and watch the bot clear all 500+ levels automatically without losing a single star!

---

## 📂 Project Structure
- rrow_puzzle_solver.py: Main ray-casting decision engine and ADB touch controller.
- equirements.txt: Python package dependencies (opencv-python, 
umpy, pure-python-adb).
- README.md: Project documentation and setup guide.