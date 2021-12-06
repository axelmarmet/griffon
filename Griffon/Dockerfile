# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/pytorch:21.11-py3


COPY  environment_docker.yml .
RUN conda env create -f environment_docker.yml

RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/#*PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

ENV NOTVISIBLE="in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]

# RUN adduser worker
# USER worker
# WORKDIR /home/worker

# ENV PATH="/home/worker/.local/bin:${PATH}"

# COPY --chown=worker:worker . .

# # RUN pip install -e .

# CMD ["tail", "-f", "/dev/null"]