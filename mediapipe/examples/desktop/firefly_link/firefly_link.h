#include <algorithm>
#include <functional>
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
    class MatrixData;
    class NormalizedLandmark;
    class NormalizedLandmarkList;
    namespace face_geometry {
        class FaceGeometry;
    }
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

    namespace ARKit {
        enum FaceBlendShape {
            EyeBlinkLeft = 0,
            EyeLookDownLeft = 1,
            EyeLookInLeft = 2,
            EyeLookOutLeft = 3,
            EyeLookUpLeft = 4,
            EyeSquintLeft = 5,
            EyeWideLeft = 6,
            EyeBlinkRight = 7,
            EyeLookDownRight = 8,
            EyeLookInRight = 9,
            EyeLookOutRight = 10,
            EyeLookUpRight = 11,
            EyeSquintRight = 12,
            EyeWideRight = 13,
            JawForward = 14,
            JawLeft = 15,
            JawRight = 16,
            JawOpen = 17,
            MouthClose = 18,
            MouthFunnel = 19,
            MouthPucker = 20,
            MouthLeft = 21,
            MouthRight = 22,
            MouthSmileLeft = 23,
            MouthSmileRight = 24,
            MouthFrownLeft = 25,
            MouthFrownRight = 26,
            MouthDimpleLeft = 27,
            MouthDimpleRight = 28,
            MouthStretchLeft = 29,
            MouthStretchRight = 30,
            MouthRollLower = 31,
            MouthRollUpper = 32,
            MouthShrugLower = 33,
            MouthShrugUpper = 34,
            MouthPressLeft = 35,
            MouthPressRight = 36,
            MouthLowerDownLeft = 37,
            MouthLowerDownRight = 38,
            MouthUpperUpLeft = 39,
            MouthUpperUpRight = 40,
            BrowDownLeft = 41,
            BrowDownRight = 42,
            BrowInnerUp = 43,
            BrowOuterUpLeft = 44,
            BrowOuterUpRight = 45,
            CheekPuff = 46,
            CheekSquintLeft = 47,
            CheekSquintRight = 48,
            NoseSneerLeft = 49,
            NoseSneerRight = 50,
            TongueOut = 51,
            HeadYaw = 52,
            HeadPitch = 53,
            HeadRoll = 54,
            LeftEyeYaw = 55,
            LeftEyePitch = 56,
            LeftEyeRoll = 57,
            RightEyeYaw = 58,
            RightEyePitch = 59,
            RightEyeRoll = 60,
        };

        enum PoseLandmark {
            NOSE = 0,
            LEFT_SHOULDER = 11,
            RIGHT_SHOULDER = 12,
            LEFT_ELBOW =13,
            RIGHT_ELBOW = 14,
            LEFT_WRIST = 15,
            RIGHT_WRIST = 16,
            LEFT_PINKY = 17,
            RIGHT_PINKY = 18,
            LEFT_INDEX = 19,
            RIGHT_INDEX = 20,
            LEFT_THUMB = 21,
            RIGHT_THUMB = 22,
            LEFT_HIP = 23,
            RIGHT_HIP = 24,
            LEFT_KNEE = 25,
            RIGHT_KNEE = 26,
            LEFT_ANKLE = 27,
            RIGHT_ANKLE = 28,
            LEFT_FOOT_INDEX = 31,
            RIGHT_FOOT_INDEX = 32,
        };

        enum HandLandMark {
            WRIST = 0,
            THUMB_1 = 1,
            THUMB_2 = 2,
            THUMB_3 = 3,
            THUMB_4 = 4,
            INDEX_1 = 5,
            INDEX_2 = 6,
            INDEX_3 = 7,
            INDEX_4 = 8,
            MIDDLE_1 = 9,
            MIDDLE_2 = 10,
            MIDDLE_3 = 11,
            MIDDLE_4 = 12,
            RING_1 = 13,
            RING_2 = 14,
            RING_3 = 15,
            RING_4 = 16,
            LITTLE_1 = 17,
            LITTLE_2 = 18,
            LITTLE_3 = 19,
            LITTLE_4 = 20,
        };
    }

    struct ARKitFaceBlendShapes {
        float data[61];
    };

    struct PosePositions {
        float data[99];
    };

    struct HandPositions {
        float data[63];
    };

    struct AllOutput {
        ARKitFaceBlendShapes bs;
        PosePositions        poses;
        HandPositions        leftHand, rightHand;
    };

    Euler mat2euler(const mediapipe::MatrixData& data);

    class Vertex {
        public:
        Vertex(float x_=0, float y_=0, float z_ = 0): x(x_), y(y_), z(z_) {
        }
        Vertex(const Vertex& v): x(v.x), y(v.y), z(v.z){
        }

        Vertex& operator /=(float v) {
            x /= v;
            y /= v;
            z /= v;
            return *this;
        }
        Vertex operator / (float v) const {
            Vertex ret(*this);
            ret /= v;
            return ret;
        }
        
        Vertex& operator +=(const Vertex& other) {
            x += other.x;
            y += other.y;
            z += other.z;
            return *this;
        }
        Vertex operator +(const Vertex& other) const {
            Vertex ret(*this);
            ret += other;
            return ret;
        }


        Vertex& operator -=(const Vertex& other) {
            x -= other.x;
            y -= other.y;
            z -= other.z;
            return *this;
        }
        Vertex operator -(const Vertex& other) const {
            Vertex ret(*this);
            ret -= other;
            return ret;
        }

        float length() const {
            return sqrtf(x*x+y*y+z*z);
        }

        float x, y, z;
    };

    inline float distance(const Vertex& a, const Vertex& b) {
        return (a - b).length();
    }

    inline float distance2d(const Vertex& a, const Vertex& b) {
        auto v = (a - b);
        v.z = 0.0f;
        return v.length();
    }

    inline float cross2d(const Vertex&a, const Vertex& b) {
        return a.x * b.y - a.y * b.x;
    }

    template <typename T> int sign(T val) {
        return (T(0) < val) - (val < T(0));
    }

    void geometry2blendshape(
        const mediapipe::face_geometry::FaceGeometry& geometry,
        ARKitFaceBlendShapes& out
    );

    void iris2blendshape(
        const mediapipe::NormalizedLandmarkList& face_landmarks,
        ARKitFaceBlendShapes& out
    );

    absl::Status init(int cameraId);
    absl::Status run(AllOutput* out, bool showDebug, std::function<void (const void* data, int width, int height)> preview);
    void shutdown();
}
