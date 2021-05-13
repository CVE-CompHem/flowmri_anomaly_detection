ARG ANOMALY_DETECTION_DEPLOY

FROM $ANOMALY_DETECTION_DEPLOY

ARG PYDEVD_PYCHARM_VERSION

RUN apt-get update  && apt-get install -y --no-install-recommends \
        inotify-tools \
     && rm -rf /var/lib/apt/lists/*

RUN bash -c "cd /src/hpc-predict/flowmri_anomaly_detection \
    && source venv/bin/activate \
    && pip install pydevd-pycharm~=$PYDEVD_PYCHARM_VERSION"
