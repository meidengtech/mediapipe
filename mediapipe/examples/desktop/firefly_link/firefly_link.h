#include <algorithm>

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
    }

    struct ARKitFaceBlendShapes {
        float bs[61];
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

    void geometry2blendshape(
        const mediapipe::face_geometry::FaceGeometry& geometry,
        ARKitFaceBlendShapes& out
    );
}
