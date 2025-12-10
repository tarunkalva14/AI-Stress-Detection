# AI‑Stress‑Detection

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Flask](https://img.shields.io/badge/Flask-Backend-green)
![PyTorch](https://img.shields.io/badge/PyTorch-AI-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Project Overview
AI‑Stress‑Detection is a modern full-stack project that combines computer vision, deep learning, and web development to deliver real-time stress monitoring. Using a pre-trained neural network model, the system analyzes live webcam video to estimate stress levels and provides a dynamic trend visualization. The project leverages PyTorch for AI inference, Flask for backend APIs, and an interactive frontend for visualization.

In today’s world, mental wellness and AI-powered monitoring tools are increasingly important. This project demonstrates how AI can be applied to human-centered applications, providing insights into emotional states, improving productivity, and supporting mental health initiatives.


## Project Structure
AI‑Stress‑Detection/
- **app.py** — Flask backend + model loading + inference APIs  
- **stress_model.pth** — Pre-trained model weights  
- **requirements.txt** — Python dependencies  
- **predictions.log** — (Optional) persistent server log of predictions  
- **static/** — Frontend assets  
  - `main.js`  
- **temaplates/** — HTML templates  
  - `index.html`  
- **README.md** — Project documentation  

## Features
- Live stress detection via webcam  
- Calibration for personalized baseline threshold  
- Trend visualization of stress levels  
- Optional AWS S3 integration for saving frame snapshots  
- Persistent logging in `predictions.log`  


## How to Run Locally (Webcam)
1. Clone the repository.
2. Ensure Python 3.x is installed.
3. (Optional) Create and activate a virtual environment.
4. Install dependencies with `pip install -r requirements.txt`.
5. Run the Flask server using `python app.py`.
6. Open your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000).
7. Allow webcam access → click **Start** to begin live stress detection.
8. Use **Calibrate** to set a personalized baseline threshold.

## How to Run on Remote Server (e.g., EC2)
1. Deploy code on the server and run Flask.
2. On your local PC, create an SSH tunnel:
   `ssh -i "<your-key>.pem" -L 5000:127.0.0.1:5000 ec2-user@<EC2-PUBLIC-IP>`
3. Open your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000).
4. Allow webcam access → everything works as if running locally.

## Output & Logging
- Each prediction (frame or video) can be logged in `predictions.log`.
- Optional frame snapshots can be uploaded to S3.
- Frontend shows live stress percentage, trend chart, and historical logs.

## Use Cases
- Stress monitoring for personal wellness apps.
- Real-time stress tracking in workplaces.
- Research or data collection for psychological studies.
- Prototype for affective computing and mental health tools.

## Why This Project Matters
- Combines full-stack web development with AI and computer vision.
- Easy local and cloud deployment.
- Demonstrates skills in machine learning inference, web apps, and deployment.

## Future Improvements
- Support video streams (RTSP / IP cameras).
- Upgrade model or use multimodal inputs (audio, sensors).
- Containerized deployment (Docker) with HTTPS.
- Multi-user support and historical data storage.
- Enhanced UI/UX and mobile compatibility.

## Conclusion
This project was developed by **TARUN KALVA**, a recent graduate with a Master’s degree in Computer Science. It demonstrates hands-on experience in:

- Frontend and backend web development using **Flask**.
- Machine learning inference and computer vision with **PyTorch**.
- Deploying and running AI-powered applications both locally and on cloud platforms like **AWS EC2**.

This project showcases the practical integration of AI into a real-time web application, combining machine learning, computer vision, and full-stack development skills. As a recent MS in Computer Science graduate, I built this system to demonstrate not only technical proficiency but also the ability to deliver AI solutions that are relevant to today’s digital and health-focused landscape.

By deploying AI models locally or on cloud servers like EC2, the system is flexible and ready for both research and real-world applications, highlighting the power and potential of AI-driven solutions in modern software development.


## License
This project is licensed under the MIT License.

## Author
**TARUN KALVA** — Recent Computer Science graduate, specialized in frontend and AI projects.

*Thank you for exploring AI‑Stress‑Detection! Contributions, forks, and stars are welcome.*
