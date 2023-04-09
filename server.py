import argparse
import asyncio
import json
import logging
import math
import os
import ssl
import uuid
import aiohttp_cors as aiohttp_cors
import cv2
import face_recognition
from aiohttp import web
import numpy as np

from av import VideoFrame
import register

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

dem = 1


def face_confidence(face_distance, face_match_threshold=0.6):
  range = (1.0 - face_match_threshold)
  linear_val = (1.0 - face_distance) / (range * 2.0)

  if face_distance > face_match_threshold:
    return str(round(linear_val * 100, 2)) + '%'
  else:
    value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
    return str(round(value, 2)) + '%'


class VideoTransformTrack(MediaStreamTrack):
  """
    A video stream track that transforms frames from an another track.
    """

  kind = "video"
  counter = 0;

  # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
  # landmark_detector = cv2.face.createFacemarkLBF()
  # landmark_detector.loadModel('lbfmodel.yaml')
  def __init__(self, track, transform):
    super().__init__()  # don't forget this!
    self.known_face_names = []
    self.known_faces_encodings = []
    self.face_encodings = []
    self.face_names = []
    self.face_locations = []
    self.track = track
    self.transform = transform
    self.process_current_frame = True
    self.counter = 0

    f = open('data.json', 'r+')
    data = json.load(f)
    for image in data:
      self.known_face_names.append(image["class_name"])
      self.known_faces_encodings.append(image["encode"])
    f.close()

  async def recv(self):
    frame = await self.track.recv()
    self.counter += 1
    # global dem
    img = frame.to_ndarray(format="bgr24")
    img = np.array(img)
    print(self.counter)
    print(self.process_current_frame)
    print(type(img))

    if self.counter % 3 == 0:
      small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
      rgb_small_frame = small_frame[:, :, ::-1]
      self.face_locations = face_recognition.face_locations(small_frame)  # Lấy vị trí khuôn mặt
      if self.counter % 4 == 0:
        self.face_encodings = face_recognition.face_encodings(small_frame,
                                                          self.face_locations) # Mã hóa khuôn mặt về vector 128 chiều

      self.face_names = []
      for face_encoding in self.face_encodings:
        matches = face_recognition.compare_faces(self.known_faces_encodings, face_encoding)
        name = "Unknown"
        confidence = '???'
        # Calculate shortest distance
        face_distances = face_recognition.face_distance(self.known_faces_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = self.known_face_names[best_match_index]
            confidence = face_confidence(face_distances[best_match_index])

        self.face_names.append(f'{name} ({confidence})')

    self.process_current_frame = not self.process_current_frame

    # Display the results
    for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
      # Scale back up face locations since the frame we detected in was scaled to 1/4 size
      top *= 4
      right *= 4
      bottom *= 4
      left *= 4

      # Create the frame with the name
      cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
      cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
      cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    new_frame = VideoFrame.from_ndarray(img, format="bgr24")
    new_frame.pts = frame.pts
    new_frame.time_base = frame.time_base
    return new_frame


async def index(request):
  content = open(os.path.join(ROOT, "index.html"), "r").read()
  return web.Response(content_type="text/html", text=content)


async def javascript(request):
  content = open(os.path.join(ROOT, "client.js"), "r").read()
  return web.Response(content_type="application/javascript", text=content)


async def offer(request):
  params = await request.json()
  offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

  pc = RTCPeerConnection()
  pc_id = "PeerConnection(%s)" % uuid.uuid4()
  pcs.add(pc)

  def log_info(msg, *args):
    logger.info(pc_id + " " + msg, *args)

  log_info("Created for %s", request.remote)

  # prepare local media
  player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
  if args.record_to:
    recorder = MediaRecorder(args.record_to)
  else:
    recorder = MediaBlackhole()

  @pc.on("datachannel")
  def on_datachannel(channel):
    @channel.on("message")
    def on_message(message):
      if isinstance(message, str) and message.startswith("ping"):
        channel.send("pong" + message[4:])

  @pc.on("connectionstatechange")
  async def on_connectionstatechange():
    log_info("Connection state is %s", pc.connectionState)
    if pc.connectionState == "failed":
      await pc.close()
      pcs.discard(pc)

  @pc.on("track")
  def on_track(track):
    log_info("Track %s received", track.kind)

    if track.kind == "audio":
      pc.addTrack(player.audio)
      recorder.addTrack(track)
    elif track.kind == "video":
      pc.addTrack(
        VideoTransformTrack(
          relay.subscribe(track), transform=params["video_transform"]
        )
      )
      if args.record_to:
        recorder.addTrack(relay.subscribe(track))

    @track.on("ended")
    async def on_ended():
      log_info("Track %s ended", track.kind)
      await recorder.stop()

  # handle offer
  await pc.setRemoteDescription(offer)
  await recorder.start()

  # send answer
  answer = await pc.createAnswer()
  await pc.setLocalDescription(answer)

  response = web.Response(
    content_type="application/json",
    text=json.dumps(
      {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    ),
  )
  response.headers['Access-Control-Allow-Origin'] = '*'
  return response


async def on_shutdown(app):
  # close peer connections
  coros = [pc.close() for pc in pcs]
  await asyncio.gather(*coros)
  pcs.clear()


async def handler(request):
  response = web.Response(text="Hello, world!")
  return response


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="WebRTC audio / video / data-channels demo"
  )
  parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
  parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
  parser.add_argument(
    "--host", default="localhost", help="Host for HTTP server (default: 0.0.0.0)"
  )
  parser.add_argument(
    "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
  )
  parser.add_argument("--record-to", help="Write received media to a file."),
  parser.add_argument("--verbose", "-v", action="count")
  args = parser.parse_args()

  if args.verbose:
    logging.basicConfig(level=logging.DEBUG)
  else:
    logging.basicConfig(level=logging.INFO)

  if args.cert_file:
    ssl_context = ssl.SSLContext()
    ssl_context.load_cert_chain(args.cert_file, args.key_file)
  else:
    ssl_context = None

  app = web.Application()
  app.on_shutdown.append(on_shutdown)
  app.router.add_get("/", index)
  app.router.add_get("/client.js", javascript)
  app.router.add_post("/offer", offer)
  cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
      allow_credentials=True,
      expose_headers="*",
      allow_headers="*",
    )
  })
  resource = cors.add(app.router.add_resource("/hello"))
  cors.add(resource.add_route("POST", handler))
  web.run_app(
    app
  )
