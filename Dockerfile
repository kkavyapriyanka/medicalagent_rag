# Use the official Python image from the Docker Hub
FROM python:3.11

# Set the working directory
WORKDIR /code

# Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Set the working directory to the user's home directory
WORKDIR /home/user/app

# Set the USER_AGENT environment variable
ENV USER_AGENT="medical_rag/1.0"

# Set the TRANSFORMERS_CACHE environment variable to a writable directory
ENV TRANSFORMERS_CACHE="/home/user/.cache/huggingface/hub"

# Create the cache directory and ensure it is writable
RUN mkdir -p /home/user/.cache/huggingface/hub && \
    chown -R user:user /home/user/.cache

# Copy the current directory contents into the container at /home/user/app
COPY --chown=user . .

# Find the path of Gradio and download the missing frpc file while in root context
RUN python3 -c "import gradio; print(gradio.__file__)" > /tmp/gradio_path.txt && \
    GRADIO_PATH=$(cat /tmp/gradio_path.txt | sed 's|/[^/]*$||') && \
    curl -L -o /usr/local/lib/python3.11/site-packages/gradio/frpc_linux_amd64 https://cdn-media.huggingface.co/frpc-gradio-0.2/frpc_linux_amd64 && \
    mv /usr/local/lib/python3.11/site-packages/gradio/frpc_linux_amd64 /usr/local/lib/python3.11/site-packages/gradio/frpc_linux_amd64_v0.2

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user after the file operations
USER user

# Expose the port for Gradio
EXPOSE 7860

# Run the application
CMD python3 -u app.py
