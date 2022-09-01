# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

cc_binary(
    name = "firefly_link",
    srcs = [
        "firefly_link.cc",
        "mat2euler.cc",
        "face_euler_calculator.cc",
        "face_landmark_to_multi_calculator.cc"
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_graph",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:matrix_data_cc_proto",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/modules/face_geometry/protos:geometry_pipeline_metadata_cc_proto",
        "//mediapipe/modules/face_geometry:env_generator_calculator",
        "//mediapipe/modules/face_geometry:geometry_pipeline_calculator",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",

        "//mediapipe/graphs/holistic_tracking:holistic_tracking_cpu_graph_deps",
    ],
)
