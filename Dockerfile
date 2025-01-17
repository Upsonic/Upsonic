# Use the official Ubuntu base image
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV USER=docker

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    rm -rf /var/lib/apt/lists/*
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update

RUN apt-get update && apt-get install -y \
    xfce4 \
    xfce4-goodies \
    tightvncserver \
    xterm \
    wget \
    curl \
    xvfb \
    software-properties-common \
    tzdata \
    python3.12 python3.12-dev gcc \
    python3.12-tk libportaudio2 scrot libportaudio2

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

RUN add-apt-repository ppa:mozillateam/ppa
RUN apt-get update && apt-get install -y firefox-esr

RUN apt-get install -y gnome-screenshot




RUN apt-get remove -y xfce4-power-manager


RUN curl -sL https://deb.nodesource.com/setup_18.x -o /tmp/nodesource_setup.sh
RUN bash /tmp/nodesource_setup.sh
RUN apt-get install -y nodejs


# Add symbolic link for uvx
RUN ln -s /home/docker/.local/bin/uvx /usr/local/bin/uvx



RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo

RUN touch /home/docker/.Xauthority
RUN chown docker:docker /home/docker/.Xauthority

USER docker



RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN mkdir /home/docker/.vnc
RUN echo "docker" | vncpasswd -f > /home/docker/.vnc/passwd
RUN chmod 600 /home/docker/.vnc/passwd

RUN echo '#!/bin/bash\nxrdb $HOME/.Xresources\nstartxfce4 &' > /home/docker/.vnc/xstartup
RUN chmod +x /home/docker/.vnc/xstartup



EXPOSE 5901
EXPOSE 7541

RUN mkdir /home/docker/Upsonic
COPY Upsonic /home/docker/Upsonic


RUN python3.12 -m pip install --upgrade pip
RUN python3.12 -m pip install /home/docker/Upsonic[server]


ADD Upsonic/wallpaper.png /home/docker/Pictures/wallpaper.png

# Configure VNC startup script
RUN echo '#!/bin/bash\n\
xrdb $HOME/.Xresources\n\
startxfce4 &\n\
sleep 2\n\
export XAUTHORITY=$HOME/.Xauthority\n\
export DISPLAY=:1\n\
xfconf-query -c xfce4-desktop -p /backdrop/screen0/monitor0/workspace0/last-image --create -t string -s /home/docker/Pictures/wallpaper.png\n' > /home/docker/.vnc/xstartup
RUN chmod +x /home/docker/.vnc/xstartup



CMD /bin/bash -c "export DISPLAY=:1 && /usr/bin/vncserver :1 -geometry 1366x768 -depth 24 && \
    python3.12 -c 'from upsonic.server import run_main_server_internal; run_main_server_internal(reload=False)' & \
    python3.12 -c 'from upsonic.tools_server import run_tools_server_internal; run_tools_server_internal(reload=False)' & \
    wait"
