#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/modules/face_geometry/libs/geometry_pipeline.h"
#include "mediapipe/util/resource_util.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
    static constexpr char kFaceLandmarksTag[] = "FACE_LANDMARKS";
    static constexpr char kMultiFaceLandmarksTag[] = "MULTI_FACE_LANDMARKS";

    class FaceLandmarkToMultiCalculator: public CalculatorBase {
    public:
        static absl::Status GetContract(CalculatorContract* cc) {
            cc->Inputs()
                .Tag(kFaceLandmarksTag)
                .Set<NormalizedLandmarkList>();
            cc->Outputs()
                .Tag(kMultiFaceLandmarksTag)
                .Set<std::vector<NormalizedLandmarkList>>();

            return absl::OkStatus();
        }

        absl::Status Process(CalculatorContext* cc) override {
            const auto& face_landmarks =
            cc->Inputs()
                .Tag(kFaceLandmarksTag)
                .Get<NormalizedLandmarkList>();

            auto multi_face_landmarks =
                absl::make_unique<std::vector<NormalizedLandmarkList>>();

            multi_face_landmarks->resize(1);
            for (int i = 0;i < 468; i++) {
                *((*multi_face_landmarks)[0].add_landmark()) = face_landmarks.landmark(i);
            }

            cc->Outputs()
                .Tag(kMultiFaceLandmarksTag)
                .AddPacket(mediapipe::Adopt<std::vector<NormalizedLandmarkList>>(
                        multi_face_landmarks.release())
                        .At(cc->InputTimestamp()));
            return absl::OkStatus();
        }
    };

    REGISTER_CALCULATOR(FaceLandmarkToMultiCalculator);

}

