#include <map>
#include <memory>
#include <algorithm>

#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/modules/face_geometry/libs/geometry_pipeline.h"
#include "mediapipe/util/resource_util.h"
#include "mediapipe/framework/port/ret_check.h"
#include "firefly_link.h"

namespace mediapipe {
    static constexpr char kFaceLandmarksTag[] = "FACE_LANDMARKS";
    static constexpr char kMultiFaceGeometryTag[] = "MULTI_FACE_GEOMETRY";
    static constexpr char kFaceBlendshapes[] = "FACE_BLEND_SHAPES";

    class ArkitBlendshapesCalculator: public CalculatorBase {
    public:
        static absl::Status GetContract(CalculatorContract* cc) {
            cc->Inputs()
                .Tag(kFaceLandmarksTag)
                .Set<NormalizedLandmarkList>();
            cc->Inputs()
                .Tag(kMultiFaceGeometryTag)
                .Set<std::vector<face_geometry::FaceGeometry>>();
            cc->Outputs()
                .Tag(kFaceBlendshapes)
                .Set<firefly::ARKitFaceBlendShapes>();

            return absl::OkStatus();
        }

        absl::Status Process(CalculatorContext* cc) override {
            const auto& mutiple_face_geometry =
            cc->Inputs()
                .Tag(kMultiFaceGeometryTag)
                .Get<std::vector<face_geometry::FaceGeometry>>();

            const auto& face_landmarks =
            cc->Inputs()
                .Tag(kFaceLandmarksTag)
                .Get<NormalizedLandmarkList>();

            std::unique_ptr<firefly::ARKitFaceBlendShapes> blendShapes = std::make_unique<firefly::ARKitFaceBlendShapes>();

            // reset to zero.
            std::fill(blendShapes->bs, blendShapes->bs + 61, 0);
            firefly::geometry2blendshape(mutiple_face_geometry[0], *blendShapes);
            firefly::iris2blendshape(face_landmarks, *blendShapes);

            cc->Outputs()
                .Tag(kFaceBlendshapes)
                .AddPacket(mediapipe::Adopt<firefly::ARKitFaceBlendShapes>(
                        blendShapes.release())
                        .At(cc->InputTimestamp()));
            
            return absl::OkStatus();
        }
    };

    REGISTER_CALCULATOR(ArkitBlendshapesCalculator);
}

