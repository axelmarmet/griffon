# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/pytorch:21.03-py3

RUN adduser worker
USER worker
WORKDIR /home/worker

COPY --chown=worker:worker requirements.txt requirements.txt

ENV PATH="/home/worker/.local/bin:${PATH}"

RUN python -m pip install --upgrade pip
RUN pip install --user -r requirements.txt

COPY --chown=worker:worker . .

RUN pip install -e .

CMD ["tail", "-f", "/dev/null"]