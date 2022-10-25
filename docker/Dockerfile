# Image name: landerlini/ml-infn-lab

FROM dodasts/snj-base-lab-gpu:v1.0.5p1-snj

# Install system packages
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && apt-get install -y subversion graphviz libgl1 \
    build-essential libargon2-0-dev libffi-dev libpixman-1-dev libcairo2-dev libpangox-1.0-dev libjpeg-dev \
    libczmq-dev pandoc texlive-xetex texlive-fonts-recommended texlive-base \
    vim tmate snakemake screen \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install python packages
RUN python3 -m pip install --no-cache-dir \
    graphviz==0.17 \
    gwdama==0.5.3 \
    keras==2.7.0 \
    lalsuite==7.0 \
    matplotlib==3.4.3 \
    opencv-python==4.5.4.58 \
    pandoc==1.1.0 \
    pydot==1.4.2 \
    scikit-image==0.18.3 \
    scikit-learn==1.0.1 \
    seaborn==0.11.2 \
    torch==1.10.0 \
    uproot==4.1.7 \
    optuna==3.0.3 \
    iminuit==2.17.0 
  

