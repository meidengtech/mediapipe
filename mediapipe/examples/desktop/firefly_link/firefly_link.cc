// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// A simple example to print out "Hello World!" from a MediaPipe graph.

#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <memory>
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"

#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"

class Vertex {
public:
  Vertex(float x_=0, float y_=0, float z_ = 0): x(x_), y(y_), z(z_) {
  }
  Vertex(const Vertex& v): x(v.x), y(v.y), z(v.z){
  }
  Vertex(const mediapipe::NormalizedLandmark& mark): x(mark.x()), y(mark.y()), z(mark.z()) {
  }

  Vertex& operator /=(float v) {
    x /= v;
    y /= v;
    z /= v;
  }
  Vertex operator / (float v) {
    Vertex ret(*this);
    ret /= v;
    return ret;
  }

  Vertex& operator -=(const Vertex& other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
  }
  Vertex operator -(const Vertex& other) {
    Vertex ret(*this);
    ret -= other;
    return ret;
  }

  float length() {
    return sqrtf(x*x+y*y+z*z);
  }

  float distance(const Vertex& other) {
    return ((*this) - other).length();
  }

  float x, y, z;
};

inline float clamp(float val, float min, float max) {
  return std::max(std::min(val, max), min);
}

inline float remap(float val, float l, float r) {
  return (clamp(val, l, r)- l) / (r - l);
}

std::unique_ptr<mediapipe::CalculatorGraph> graph;
std::unique_ptr<mediapipe::OutputStreamPoller> output_video_poller;
std::unique_ptr<mediapipe::OutputStreamPoller> pose_world_landmarks_poller;
std::unique_ptr<mediapipe::OutputStreamPoller> face_landmarks_poller;

bool getPooler(const char* name, std::unique_ptr<mediapipe::OutputStreamPoller>& ptr) {
  auto statusor =  std::move(graph->AddOutputStreamPoller(name));
  if (!statusor.ok()) {
    LOG(ERROR) << "Poller init failed.";
    graph.reset();
    return false;
  }
  ptr = std::make_unique<mediapipe::OutputStreamPoller>(std::move(std::move(statusor).value()));
  return true;
}

bool initGraph() {
    mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(R"pb(
# CPU image. (ImageFrame)
input_stream: "input_video"

# Whether to run face mesh model with attention on lips and eyes. (bool)
# Attention provides more accuracy on lips and eye regions as well as iris
# landmarks.
input_side_packet: "refine_face_landmarks"

# Whether landmarks on the previous image should be used to help localize
# landmarks on the current image. (bool)
input_side_packet: "use_prev_landmarks"

# CPU image with rendered results. (ImageFrame)
output_stream: "output_video"

# Pose world landmarks.(LandmarkList)
output_stream: "pose_world_landmarks"

# Face landmarks (NormalizedLandmarkList)
output_stream: "face_landmarks"

node {
  calculator: "FlowLimiterCalculator"
  input_stream: "input_video"
  input_stream: "FINISHED:output_video"
  input_stream_info: {
    tag_index: "FINISHED"
    back_edge: true
  }
  output_stream: "throttled_input_video"
  node_options: {
    [type.googleapis.com/mediapipe.FlowLimiterCalculatorOptions] {
      max_in_flight: 1
      max_in_queue: 1
      # Timeout is disabled (set to 0) as first frame processing can take more
      # than 1 second.
      in_flight_timeout: 0
    }
  }
}

node {
  calculator: "HolisticLandmarkCpu"
  input_stream: "IMAGE:throttled_input_video"
  input_side_packet: "REFINE_FACE_LANDMARKS:refine_face_landmarks"
  input_side_packet: "USE_PREV_LANDMARKS:use_prev_landmarks"
  output_stream: "POSE_LANDMARKS:pose_landmarks"
  output_stream: "WORLD_LANDMARKS:pose_world_landmarks"
  output_stream: "POSE_ROI:pose_roi"
  output_stream: "POSE_DETECTION:pose_detection"
  output_stream: "FACE_LANDMARKS:face_landmarks"
  output_stream: "LEFT_HAND_LANDMARKS:left_hand_landmarks"
  output_stream: "RIGHT_HAND_LANDMARKS:right_hand_landmarks"
}

# Gets image size.
node {
  calculator: "ImagePropertiesCalculator"
  input_stream: "IMAGE:throttled_input_video"
  output_stream: "SIZE:image_size"
}

# Converts pose, hands and face landmarks to a render data vector.
node {
  calculator: "HolisticTrackingToRenderData"
  input_stream: "IMAGE_SIZE:image_size"
  input_stream: "POSE_LANDMARKS:pose_landmarks"
  input_stream: "POSE_ROI:pose_roi"
  input_stream: "LEFT_HAND_LANDMARKS:left_hand_landmarks"
  input_stream: "RIGHT_HAND_LANDMARKS:right_hand_landmarks"
  input_stream: "FACE_LANDMARKS:face_landmarks"
  output_stream: "RENDER_DATA_VECTOR:render_data_vector"
}

# Draws annotations and overlays them on top of the input images.
node {
  calculator: "AnnotationOverlayCalculator"
  input_stream: "IMAGE:throttled_input_video"
  input_stream: "VECTOR:render_data_vector"
  output_stream: "IMAGE:output_video"
}

      )pb");

  graph.reset(new mediapipe::CalculatorGraph());
  if (!graph->Initialize(config).ok()) {
    LOG(ERROR) << "Graph init failed.";
    graph.reset();
    return false;
  }

  LOG(INFO) << "Start running the calculator graph.";

  if (!getPooler("output_video", output_video_poller)) {
    return false;
  }

  if (!getPooler("pose_world_landmarks", pose_world_landmarks_poller)) {
    return false;
  }

  if (!getPooler("face_landmarks", face_landmarks_poller)) {
    return false;
  }

  std::map<std::string, mediapipe::Packet> extra_side_packets;
  extra_side_packets["refine_face_landmarks"] = mediapipe::MakePacket<bool>(true);
  extra_side_packets["use_prev_landmarks"] = mediapipe::MakePacket<bool>(true);

  if (!graph->StartRun(extra_side_packets).ok()) {
    LOG(ERROR) << "Graph init failed.";
    graph.reset();
    return false;
  }

  LOG(INFO) << "Start grabbing and processing frames.";
  return true;
}


void calcMouth(const mediapipe::NormalizedLandmarkList& list) {
  Vertex eyeInnerCornerL(list.landmark(133));
  Vertex eyeInnerCornerR(list.landmark(362));
  Vertex eyeOuterCornerL(list.landmark(130));
  Vertex eyeOuterCornerR(list.landmark(263));
  float eyeInnerDistance = eyeInnerCornerL.distance(eyeInnerCornerR);
  float eyeOuterDistance = eyeOuterCornerL.distance(eyeOuterCornerR);

  Vertex upperInnerLip(list.landmark(13));
  Vertex lowerInnerLip(list.landmark(14));
  Vertex mouthCornerLeft(list.landmark(61));
  Vertex mouthCornerRight(list.landmark(291));

  float mouthOpen = upperInnerLip.distance(lowerInnerLip);
  float mouthWidth = mouthCornerLeft.distance(mouthCornerRight);

  // mouth open and mouth shape ratios
  // let ratioXY = mouthWidth / mouthOpen;
  float ratioY = mouthOpen / eyeInnerDistance;
  float ratioX = mouthWidth / eyeOuterDistance;

  // normalize and scale mouth open
  ratioY = remap(ratioY, 0.15, 0.7);

  // normalize and scale mouth shape
  ratioX = remap(ratioX, 0.30, 0.9);
  ratioX = (ratioX - 0.3) * 2;

  float mouthX = ratioX;
  float mouthY = remap(mouthOpen / eyeInnerDistance, 0.17, 0.5);

  float ratioI = clamp(remap(mouthX, 0, 1) * 2 * remap(mouthY, 0.2, 0.7), 0, 1);
  float ratioA = mouthY * 0.4 + mouthY * (1 - ratioI) * 0.6;
  float ratioU = mouthY * remap(1 - ratioI, 0, 0.3) * 0.1;
  float ratioE = remap(ratioU, 0.2, 1) * (1 - ratioI) * 0.3;
  float ratioO = (1 - ratioI) * remap(mouthY, 0.3, 1) * 0.4;

  printf("X: %f, Y:%f, A:%f, E:%f, I:%f, O:%f, U:%f\n", mouthX, mouthY, ratioA, ratioE, ratioI, ratioO, ratioU);

}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  if (!initGraph()) {
    return -1;
  }

  cv::VideoCapture capture;
  capture.open(0);
  cv::namedWindow("Firefly", /*flags=WINDOW_AUTOSIZE*/ 1);
  capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  capture.set(cv::CAP_PROP_FPS, 30);

  bool running = true;
  
  while (running) {
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;

    if (camera_frame_raw.empty()) {
      continue;
    }

    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

      // Send image packet into the graph.
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    auto status = graph->AddPacketToInputStream(
      "input_video", mediapipe::Adopt(input_frame.release())
                        .At(mediapipe::Timestamp(frame_timestamp_us)));
    if (!status.ok()) {
      LOG(ERROR) << "Failed to add packet to input stream." << status;
      return -1;
    }

    if (pose_world_landmarks_poller->QueueSize() > 0) {
      mediapipe::Packet packet;
      if (!pose_world_landmarks_poller->Next(&packet)) {
        LOG(ERROR) << "Failed to poll packet.";
        break;
      }
      auto& output_frame = packet.Get<mediapipe::LandmarkList>();
      // for (int i = 11; i <= 11; i++) {
      //   const auto& landmark = output_frame.landmark(i);
      //   printf("%d: %f %f %f\n", i, landmark.x(), landmark.y(), landmark.z());
      // }
    }

    if (face_landmarks_poller->QueueSize() > 0) {
      mediapipe::Packet packet;
      if (!face_landmarks_poller->Next(&packet)) {
        LOG(ERROR) << "Failed to poll packet.";
        break;
      }
      auto& output_frame = packet.Get<mediapipe::NormalizedLandmarkList>();
      
      calcMouth(output_frame);
    }

    // Get the graph result packet, or stop if that fails.
    {
      mediapipe::Packet packet;
      if (!output_video_poller->Next(&packet)) {
        LOG(ERROR) << "Failed to poll packet.";
        break;
      }
      auto& output_frame = packet.Get<mediapipe::ImageFrame>();

      cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
      cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

      cv::imshow("Firefly", output_frame_mat);
      // Press any key to exit.
      const int pressed_key = cv::waitKey(5);
      if (pressed_key >= 0 && pressed_key != 255) running = false;
    }
  }

  LOG(INFO) << "Shutting down.";
  graph->CloseInputStream("input_video");
  graph->WaitUntilDone();

  return 0;
}
