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

# Download the missing frpc file and rename it
RUN curl -L -o /home/user/.local/lib/python3.11/site-packages/gradio/frpc_linux_amd64 https://cdn-media.huggingface.co/frpc-gradio-0.2/frpc_linux_amd64 && \
    mv /home/user/.local/lib/python3.11/site-packages/gradio/frpc_linux_amd64 /home/user/.local/lib/python3.11/site-packages/gradio/frpc_linux_amd64_v0.2

EXPOSE 7860

CMD python3 -u app.py

