#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/modules/face_geometry/libs/geometry_pipeline.h"
#include "mediapipe/util/resource_util.h"
#include "mediapipe/framework/port/ret_check.h"
#include "firefly_link.h"
#include <memory>

namespace mediapipe {
    static constexpr char kMultiFaceGeometryTag[] = "MULTI_FACE_GEOMETRY";
    static constexpr char kFaceEulerTag[] = "FACE_EULER";

    class FaceEulerCalculator: public CalculatorBase {
    public:
        static absl::Status GetContract(CalculatorContract* cc) {
            cc->Inputs()
                .Tag(kMultiFaceGeometryTag)
                .Set<std::vector<face_geometry::FaceGeometry>>();
            cc->Outputs()
                .Tag(kFaceEulerTag)
                .Set<firefly::Euler>();

            return absl::OkStatus();
        }

        absl::Status Process(CalculatorContext* cc) override {
            const auto& mutiple_face_geometry =
            cc->Inputs()
                .Tag(kMultiFaceGeometryTag)
                .Get<std::vector<face_geometry::FaceGeometry>>();
            
            auto& geo = mutiple_face_geometry[0];

            firefly::Euler euler = firefly::mat2euler(geo.pose_transform_matrix());

            cc->Outputs()
                .Tag(kFaceEulerTag)
                .AddPacket(mediapipe::Adopt<firefly::Euler>(new firefly::Euler(euler))
                        .At(cc->InputTimestamp()));
            return absl::OkStatus();
        }
    };

    REGISTER_CALCULATOR(FaceEulerCalculator);
}

