#include "firefly_link.h"
#include <memory>
#include <math.h>
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"

#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/opencv_video_inc.h"


namespace firefly {
    std::unique_ptr<mediapipe::CalculatorGraph> graph;
    std::unique_ptr<mediapipe::OutputStreamPoller> output_video_poller;
    std::unique_ptr<mediapipe::OutputStreamPoller> face_blendshapes_poller;
    std::unique_ptr<cv::VideoCapture> capture;


    absl::Status getPooler(std::unique_ptr<mediapipe::CalculatorGraph>& graph, const char* name, std::unique_ptr<mediapipe::OutputStreamPoller>& ptr) {
    auto statusor =  std::move(graph->AddOutputStreamPoller(name));
    if (!statusor.ok()) {
        LOG(ERROR) << "Poller init failed.";
        graph.reset();
        return statusor.status();
    }
    ptr = std::make_unique<mediapipe::OutputStreamPoller>(std::move(std::move(statusor).value()));
    return absl::OkStatus();
    }

    absl::Status init(int cameraId) {

        std::unique_ptr<mediapipe::CalculatorGraph> graph_;
        std::unique_ptr<mediapipe::OutputStreamPoller> output_video_poller_;
        std::unique_ptr<mediapipe::OutputStreamPoller> face_blendshapes_poller_;
        std::unique_ptr<cv::VideoCapture> capture_;

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

output_stream: "face_blendshapes"

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

node {
  calculator: "LandmarksSmoothingCalculator"
  input_stream: "NORM_LANDMARKS:face_landmarks"
  input_stream: "IMAGE_SIZE:image_size"
  output_stream: "NORM_FILTERED_LANDMARKS:filtered_face_landmarks"
  node_options: {
    [type.googleapis.com/mediapipe.LandmarksSmoothingCalculatorOptions] {
      velocity_filter: {
        window_size: 5
        velocity_scale: 20.0
      }
    }
  }
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
    input_stream: "FACE_LANDMARKS:filtered_face_landmarks"
    output_stream: "RENDER_DATA_VECTOR:render_data_vector"
}

# Draws annotations and overlays them on top of the input images.
node {
    calculator: "AnnotationOverlayCalculator"
    input_stream: "IMAGE:throttled_input_video"
    input_stream: "VECTOR:render_data_vector"
    output_stream: "IMAGE:output_video"
}

node {
    calculator: "FaceGeometryEnvGeneratorCalculator"
    output_side_packet: "ENVIRONMENT:environment"
    node_options: {
        [type.googleapis.com/mediapipe.FaceGeometryEnvGeneratorCalculatorOptions] {
        environment: {
            origin_point_location: TOP_LEFT_CORNER
            perspective_camera: {
            vertical_fov_degrees: 63.0  # 63 degrees
            near: 1.0  # 1cm
            far: 10000.0  # 100m
            }
        }
        }
    }
}

node {
    calculator: "FaceLandmarkToMultiCalculator"
    input_stream: "FACE_LANDMARKS:filtered_face_landmarks"
    output_stream: "MULTI_FACE_LANDMARKS:multi_face_landmarks"
}

node {
    calculator: "FaceGeometryPipelineCalculator"
    input_side_packet: "ENVIRONMENT:environment"
    input_stream: "IMAGE_SIZE:image_size"
    input_stream: "MULTI_FACE_LANDMARKS:multi_face_landmarks"
    output_stream: "MULTI_FACE_GEOMETRY:multi_face_geometry"
    options: {
        [mediapipe.FaceGeometryPipelineCalculatorOptions.ext] {
            metadata_path: "mediapipe/modules/face_geometry/data/geometry_pipeline_metadata_landmarks.binarypb"
        }
    }
}

node {
    calculator: "ArkitBlendshapesCalculator"
    input_stream: "FACE_LANDMARKS:filtered_face_landmarks"
    input_stream: "MULTI_FACE_GEOMETRY:multi_face_geometry"
    output_stream: "FACE_BLEND_SHAPES:face_blendshapes"
}
    )pb");

        graph_.reset(new mediapipe::CalculatorGraph());
        MP_RETURN_IF_ERROR(graph_->Initialize(config)) << "Graph init failed.";

        LOG(INFO) << "Start running the calculator graph.";

        MP_RETURN_IF_ERROR(getPooler(graph_, "output_video", output_video_poller_));
        MP_RETURN_IF_ERROR(getPooler(graph_, "face_blendshapes", face_blendshapes_poller_));
        
        std::map<std::string, mediapipe::Packet> extra_side_packets;
        extra_side_packets["refine_face_landmarks"] = mediapipe::MakePacket<bool>(true);
        extra_side_packets["use_prev_landmarks"] = mediapipe::MakePacket<bool>(true);

        MP_RETURN_IF_ERROR(graph_->StartRun(extra_side_packets)) << "Graph init failed.";

        LOG(INFO) << "Start grabbing and processing frames.";

        capture_.reset(new cv::VideoCapture());
        capture_->open(cameraId, cv::CAP_DSHOW);
        // capture_->set(cv::CAP_PROP_FRAME_WIDTH, 800);
        // capture_->set(cv::CAP_PROP_FRAME_HEIGHT, 450);
        capture_->set(cv::CAP_PROP_FPS, 30);

        graph.reset(graph_.release());
        output_video_poller.reset(output_video_poller_.release());
        face_blendshapes_poller.reset(face_blendshapes_poller_.release());
        capture.reset(capture_.release());

        return absl::OkStatus();
    }

    absl::Status run(ARKitFaceBlendShapes* out, bool showDebug) {
        cv::Mat camera_frame_raw;
        (*capture) >> camera_frame_raw;

        if (camera_frame_raw.empty()) {
            printf("Empty frame\n");
            return absl::OkStatus();
        }

        cv::Mat camera_frame;
        cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
        //cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);

        // Wrap Mat into an ImageFrame.
        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
            mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        camera_frame.copyTo(input_frame_mat);

        // Send image packet into the graph.
        size_t frame_timestamp_us =
            (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
        MP_RETURN_IF_ERROR(graph->AddPacketToInputStream(
        "input_video", mediapipe::Adopt(input_frame.release())
                            .At(mediapipe::Timestamp(frame_timestamp_us))));

        // Get the graph result packet, or stop if that fails.
        {
            mediapipe::Packet packet;
            if (!output_video_poller->Next(&packet)) {
                LOG(ERROR) << "Failed to poll packet.";
                return absl::UnknownError("poller->Next failed.");
            }
            auto& output_frame = packet.Get<mediapipe::ImageFrame>();

            cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
            cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

            if (showDebug) {
                cv::imshow("Firefly", output_frame_mat);
            }
        }

        if (face_blendshapes_poller->QueueSize() > 0)
        {
            mediapipe::Packet packet;
            if (!face_blendshapes_poller->Next(&packet)) {
                return absl::UnknownError("poller->Next failed.");
            }
            auto& output_frame = packet.Get<firefly::ARKitFaceBlendShapes>();

            if (out) {
                *out = output_frame;
            }
        }

        return absl::OkStatus();
    }

    void shutdown() {
        graph->CloseInputStream("input_video");
        graph->WaitUntilDone();
        capture.reset();
        output_video_poller.reset();
        graph.reset();
    }

}