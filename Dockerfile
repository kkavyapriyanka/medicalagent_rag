FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
# Switch to the "user" user
USER user
# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Set the USER_AGENT environment variable
ENV USER_AGENT="medical_rag/1.0"

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

# Find the path of Gradio and download the missing frpc file
RUN python3 -c "import gradio; print(gradio.__file__)" > /tmp/gradio_path.txt && \
    GRADIO_PATH=$(cat /tmp/gradio_path.txt | sed 's|/[^/]*$||') && \
    curl -L -o $GRADIO_PATH/frpc_linux_amd64 https://cdn-media.huggingface.co/frpc-gradio-0.2/frpc_linux_amd64 && \
    mv $GRADIO_PATH/frpc_linux_amd64 $GRADIO_PATH/frpc_linux_amd64_v0.2

# Switch to the "user" user after the file operations
USER user

EXPOSE 7860

CMD python3 -u app.py

