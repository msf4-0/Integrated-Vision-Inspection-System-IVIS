import streamlit as st
import streamlit.components.v1 as components

# st.set_page_config(page_title="Webcam Test Screenshot",
#                    page_icon="random", layout='wide')

components.html(
    """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta
      name="viewport"
      content="width=device-width, user-scalable=no, initial-scale=1.0, maximum-scale=1.0, minimum-scale=1.0"
    />
    <meta http-equiv="X-UA-Compatible" content="ie=edge" />
    <link
      rel="stylesheet"
      href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css"
    />
    <style>
    .screenshot-image {
  width: 150px;
  height: 90px;
  border-radius: 4px;
  border: 2px solid whitesmoke;
  box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.1);
  position: absolute;
  bottom: 5px;
  left: 10px;
  background: white;
}

.display-cover {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 70%;
  margin: 5% auto;
  position: relative;
}

video {
  width: 100%;
  background: rgba(0, 0, 0, 0.2);
}

.video-options {
  position: absolute;
  left: 20px;
  top: 30px;
}

.controls {
  position: absolute;
  right: 20px;
  top: 20px;
  display: flex;
}

.controls > button {
  width: 45px;
  height: 45px;
  text-align: center;
  border-radius: 100%;
  margin: 0 6px;
  background: transparent;
}

.controls > button:hover svg {
  color: white !important;
}

@media (min-width: 300px) and (max-width: 400px) {
  .controls {
    flex-direction: column;
  }

  .controls button {
    margin: 5px 0 !important;
  }
}

.controls > button > svg {
  height: 20px;
  width: 18px;
  text-align: center;
  margin: 0 auto;
  padding: 0;
}

.controls button:nth-child(1) {
  border: 2px solid #d2002e;
}

.controls button:nth-child(1) svg {
  color: #d2002e;
}

.controls button:nth-child(2) {
  border: 2px solid #008496;
}

.controls button:nth-child(2) svg {
  color: #008496;
}

.controls button:nth-child(3) {
  border: 2px solid #00b541;
}

.controls button:nth-child(3) svg {
  color: #00b541;
}

.controls > button {
  width: 45px;
  height: 45px;
  text-align: center;
  border-radius: 100%;
  margin: 0 6px;
  background: transparent;
}

.controls > button:hover svg {
  color: white;
}

    </style>
    <title>Document</title>
  </head>
  <body>
    <div class="display-cover">
      <video autoplay></video>
      <canvas class="d-none"></canvas>

      <div class="video-options">
        <select name="" id="" class="custom-select">
          <option value="">Select camera</option>
        </select>
      </div>

      <img class="screenshot-image d-none" alt="" />

      <div class="controls">
        <button class="btn btn-danger play" title="Play">
          <i data-feather="play-circle"></i>
        </button>
        <button class="btn btn-info pause d-none" title="Pause">
          <i data-feather="pause"></i>
        </button>
        <button
          class="btn btn-outline-success screenshot d-none"
          title="ScreenShot"
        >
          <i data-feather="image"></i>
        </button>
      </div>
    </div>

    <script src="https://unpkg.com/feather-icons"></script>
    <script>
    feather.replace();

const controls = document.querySelector('.controls');
const cameraOptions = document.querySelector('.video-options>select');
const video = document.querySelector('video');
const canvas = document.querySelector('canvas');
const screenshotImage = document.querySelector('img');
const buttons = [...controls.querySelectorAll('button')];
let streamStarted = false;

const [play, pause, screenshot] = buttons;

const constraints = {
  video: {
    width: {
      min: 1280,
      ideal: 1920,
      max: 2560,
    },
    height: {
      min: 720,
      ideal: 1080,
      max: 1440
    },
  }
};

const getCameraSelection = async () => {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const videoDevices = devices.filter(device => device.kind === 'videoinput');
  const options = videoDevices.map(videoDevice => {
    return `<option value="${videoDevice.deviceId}">${videoDevice.label}</option>`;
  });
  cameraOptions.innerHTML = options.join('');
};

play.onclick = () => {
  if (streamStarted) {
    video.play();
    play.classList.add('d-none');
    pause.classList.remove('d-none');
    return;
  }
  if ('mediaDevices' in navigator && navigator.mediaDevices.getUserMedia) {
    const updatedConstraints = {
      ...constraints,
      deviceId: {
        exact: cameraOptions.value
      }
    };
    startStream(updatedConstraints);
  }
};

const startStream = async (constraints) => {
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  handleStream(stream);
};

const handleStream = (stream) => {
  video.srcObject = stream;
  play.classList.add('d-none');
  pause.classList.remove('d-none');
  screenshot.classList.remove('d-none');
  streamStarted = true;
};

getCameraSelection();

cameraOptions.onchange = () => {
    const updatedConstraints = {
      ...constraints,
      deviceId: {
        exact: cameraOptions.value
      }
    };
    startStream(updatedConstraints);
  };
  
  const pauseStream = () => {
    video.pause();
    play.classList.remove('d-none');
    pause.classList.add('d-none');
  };
  
  const doScreenshot = () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    screenshotImage.src = canvas.toDataURL('image/webp');
    screenshotImage.classList.remove('d-none');
  };
  
  pause.onclick = pauseStream;
  screenshot.onclick = doScreenshot;
  </script>
  </body>
</html>
""", height=1000)
