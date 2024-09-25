FROM python:3.11.5
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Set the working directory to /app
WORKDIR ~/MedicalBot
# Copy the current directory contents into the container at /app
COPY . /MedicalBot

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /MedicalBot/requirements.txt

# Expose the port that Streamlit will run on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["gradio", "run", "/MedicalBot/app.py"]

