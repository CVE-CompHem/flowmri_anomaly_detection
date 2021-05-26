ARG HPC_PREDICT_IO_DEPLOY

FROM $HPC_PREDICT_IO_DEPLOY

COPY . /src/hpc-predict/flowmri_anomaly_detection

RUN apt-get update  && apt-get install -y --no-install-recommends \
    libsm6 libxext6 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN bash -c "cd /src/hpc-predict/flowmri_anomaly_detection \
    && python3 -m venv venv \
    && source venv/bin/activate \
    && pip install --upgrade pip \
    && pip install -r requirements.txt"

#VOLUME ["/hpc-predict-data"]
