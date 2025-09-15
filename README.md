[README.md](https://github.com/user-attachments/files/22328824/README.md)# Inkball 🎮

Inkball is a modern recreation of the classic puzzle game where players guide balls into goals by drawing barriers and controlling their movement. This project has been implemented in **Java**, with test files (`AppTest.class`, `BallTest.class`) included to ensure game reliability.

---

## ✨ Features
- Physics-based ball movement and collision.  
- Drawn lines guide the balls into target holes.  
- Multiple levels with increasing difficulty.  
- Lightweight design, runs on most systems.  
- Includes automated tests for core game mechanics.  

---

## 📂 Project Structure
```
Inkball/
│── src/           # Main source code files
│── tests/         # Unit test files (e.g., AppTest, BallTest)
│── assets/        # Game assets (levels, images, sounds)
│── README.md      # Project documentation
│── LICENSE        # License file
```

---

## 🕹️ Controls
- **Draw Line**: Left-click and drag  
- **Erase Line**: Right-click  
- **Restart Level**: Press `R`  
- **Pause Game**: Press `P`  

---

## 🚀 Run Instructions
Compile the source code and run the main game file:

```bash
javac src/*.java
java src.Inkball
```

To run tests (example with JUnit):
```bash
javac tests/*.java
java org.junit.runner.JUnitCore AppTest BallTest
```

---

## 🛠️ Tech Stack
- **Language:** Java  
- **Testing:** JUnit (with `AppTest` and `BallTest`)  
- **Graphics:** Java AWT/Swing or Processing (depending on your setup)  

---

## 📜 License
This project is licensed under the MIT License

---

## 👤 Author
Developed by Abhishek  


