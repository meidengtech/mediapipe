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
#include <stdlib.h>
#include "firefly_link.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  int cameraId = 0;

  if (argc >= 2) {
    cameraId = atoi(argv[1]);
  }
  printf("Camera ID: %s %d\n",argv[1], cameraId);
  
  if (!firefly::init(cameraId).ok()) {
    return -1;
  }

  bool running = true;
  firefly::ARKitFaceBlendShapes out;
  
  while (running) {
      if (!firefly::run(&out, true, nullptr).ok()) {
        break;
      }
      // Press any key to exit.
      const int pressed_key = cv::waitKey(5);
      if (pressed_key >= 0 && pressed_key != 255) running = false;
  }

  firefly::shutdown();

  return 0;
}
