# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

WORKDIR /app

# Install dependencies to add PPAs, X11 development libraries, and other utilities
RUN apt-get update && \
    apt-get install -y -qq ffmpeg aria2 libx11-dev software-properties-common python-dev python3-dev python3.9-distutils python3.9-dev python3.9 curl supervisor && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 && \
    curl https://bootstrap.pypa.io/get-pip.py | python3.9

COPY . .

RUN python3 -m pip install --no-cache-dir -r requirements.txt && \
    mkdir -p assets/pretrained_v2 assets/uvr5_weights assets/hubert assets/rmvpe && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
        https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D40k.pth -d assets/pretrained_v2/ -o D40k.pth \
        https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G40k.pth -d assets/pretrained_v2/ -o G40k.pth \
        https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D40k.pth -d assets/pretrained_v2/ -o f0D40k.pth \
        https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth -d assets/pretrained_v2/ -o f0G40k.pth \
        https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP2-人声vocals+非人声instrumentals.pth -d assets/uvr5_weights/ -o HP2-人声vocals+非人声instrumentals.pth \
        https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP5-主旋律人声vocals+其他instrumentals.pth -d assets/uvr5_weights/ -o HP5-主旋律人声vocals+其他instrumentals.pth \
        https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt -d assets/hubert -o hubert_base.pt \
        https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt -d assets/rmvpe -o rmvpe.pt


COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 7865 7866

CMD ["/usr/bin/supervisord"]
