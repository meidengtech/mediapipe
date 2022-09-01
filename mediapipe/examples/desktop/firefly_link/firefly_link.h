
namespace mediapipe {
    class MatrixData;
    class NormalizedLandmarkList;
}

namespace firefly {
    struct Euler {
        float Pitch;
        float Roll;
        float Yaw;

        Euler(float p = 0.f, float r = 0.f, float y = 0.f)
            : Pitch(p), Roll(r), Yaw(y)
        {
        }
    };

    Euler mat2euler(const mediapipe::MatrixData& data);
}
