// adapt and modified from https://github.com/JimWest/MeFaMo/blob/main/mefamo/blendshapes/blendshape_calculator.py 

#include "firefly_link.h"
#include <algorithm>
#include <type_traits>
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/modules/face_geometry/protos/face_geometry.pb.h"
#include "mediapipe/modules/face_geometry/protos/mesh_3d.pb.h"

static const int eye_right[] = {33, 133, 160, 159, 158, 144, 145, 153};
static const int eye_left[] = {263, 362, 387, 386, 385, 373, 374, 380};
static const int head[] = {10, 152};
static const int nose_tip = 1;
static const int upper_lip = 13;
static const int lower_lip = 14;
static const int upper_outer_lip = 12;
static const int mouth_corner_left = 291;
static const int mouth_corner_right = 61;
static const int lowest_chin = 175;
static const int upper_head = 10;
static const int mouth_frown_left = 422;
static const int mouth_frown_right = 202;
static const int mouth_left_stretch = 287;
static const int mouth_right_stretch = 57;
static const int lowest_lip = 17;
static const int under_lip = 18;
static const int over_upper_lip = 164;
static const int left_upper_press[] = {40, 80};
static const int left_lower_press[] = {88, 91};
static const int right_upper_press[] = {270, 310};
static const int right_lower_press[] = {318, 321};
static const int squint_left[] = {253, 450};
static const int squint_right[] = {23, 230};
static const int right_brow = 27;
static const int right_brow_lower[] = {53, 52, 65};
static const int left_brow = 257;
static const int left_brow_lower[] = {283, 282, 295};
static const int inner_brow = 9;
static const int upper_nose = 6;
static const int cheek_squint_left[] = {359, 342};
static const int cheek_squint_right[] = {130, 113};
static const int iris_right = 468;
static const int iris_left = 473;

static const firefly::ARKit::FaceBlendShape eyeDirBSLeft[] = {
    firefly::ARKit::EyeLookInLeft,
    firefly::ARKit::EyeLookOutLeft,
    firefly::ARKit::EyeLookUpLeft,
    firefly::ARKit::EyeLookDownLeft,
};

static const firefly::ARKit::FaceBlendShape eyeDirBSRight[] = {
    firefly::ARKit::EyeLookInRight,
    firefly::ARKit::EyeLookOutRight,
    firefly::ARKit::EyeLookUpRight,
    firefly::ARKit::EyeLookDownRight,
};

static std::map<firefly::ARKit::FaceBlendShape, std::pair<float, float> > remap_config= {
    {firefly::ARKit::EyeBlinkLeft , std::make_pair(0.40, 0.70) },
    {firefly::ARKit::EyeSquintLeft , std::make_pair(0.37, 0.44) },
    {firefly::ARKit::EyeWideLeft , std::make_pair(0.9, 1.2) },
    {firefly::ARKit::EyeBlinkRight , std::make_pair(0.40, 0.70) },
    {firefly::ARKit::EyeSquintRight , std::make_pair(0.37, 0.44) },
    {firefly::ARKit::EyeWideRight , std::make_pair(0.9, 1.2) },
    {firefly::ARKit::JawLeft , std::make_pair(-0.4, 0.0) },
    {firefly::ARKit::JawRight , std::make_pair(0.0, 0.4) },
    {firefly::ARKit::JawOpen , std::make_pair(1.05, 1.15) },
    {firefly::ARKit::MouthClose , std::make_pair(0, 0.7) },
    {firefly::ARKit::MouthFunnel , std::make_pair(3.5, 4) },
    {firefly::ARKit::MouthPucker , std::make_pair(3.26, 4.1) },
    {firefly::ARKit::MouthLeft , std::make_pair(-3.4, -2.3) },
    {firefly::ARKit::MouthRight , std::make_pair( 1.5, 3.0) },
    {firefly::ARKit::MouthSmileLeft , std::make_pair(-0.25, 0.25) },
    {firefly::ARKit::MouthSmileRight , std::make_pair(-0.25, 0.25) },
    {firefly::ARKit::MouthSmileLeft , std::make_pair(-0.25, 0.25) },
    {firefly::ARKit::MouthSmileRight , std::make_pair(-0.25, 0.0) },
    {firefly::ARKit::MouthFrownLeft , std::make_pair(0.4, 0.9) },
    {firefly::ARKit::MouthFrownRight , std::make_pair(0.4, 0.9) },
    {firefly::ARKit::MouthStretchLeft , std::make_pair(-0.4, 0.0) },
    {firefly::ARKit::MouthStretchRight , std::make_pair(-0.4, 0.0) },
    {firefly::ARKit::MouthRollLower , std::make_pair(0.4, 0.7) },
    {firefly::ARKit::MouthRollUpper , std::make_pair(0.31, 0.34) },
    {firefly::ARKit::MouthShrugLower , std::make_pair(1.9, 2.3) },
    {firefly::ARKit::MouthShrugUpper , std::make_pair(1.4, 2.4) },
    {firefly::ARKit::MouthPressLeft , std::make_pair(0.4, 0.5) },
    {firefly::ARKit::MouthPressRight , std::make_pair(0.4, 0.5) },
    {firefly::ARKit::MouthLowerDownLeft , std::make_pair(1.7, 2.1) },
    {firefly::ARKit::MouthLowerDownRight , std::make_pair(1.7, 2.1) },
    {firefly::ARKit::BrowDownLeft , std::make_pair(1.0, 1.2) },
    {firefly::ARKit::BrowDownRight , std::make_pair(1.0, 1.2) },
    {firefly::ARKit::BrowInnerUp , std::make_pair(2.2, 2.6) },
    {firefly::ARKit::BrowOuterUpLeft , std::make_pair(1.25, 1.5) },
    {firefly::ARKit::BrowOuterUpRight , std::make_pair(1.25, 1.5) },
    {firefly::ARKit::CheekSquintLeft , std::make_pair(0.55, 0.63) },
    {firefly::ARKit::CheekSquintRight , std::make_pair(0.55, 0.63) },
};

namespace firefly {
    inline float clamp(float val, float min, float max) {
        return std::max(std::min(val, max), min);
    }

    inline float remap(float val, float min, float max) {
        return (clamp(val, min, max) - min ) / (max - min);
    }

    inline float remap(ARKit::FaceBlendShape id, float val) {
        auto itor = remap_config.find(id);
        if (itor == remap_config.end()) {
            return val;
        }
        auto& config = itor->second;
        return remap(val, config.first, config.second);
    }

    Vertex getVertexPosition(const float* data, int index) {
        index *= 5;
        return Vertex(data[index], data[index+1], data[index+2]);
    }

    #define get(i) getVertexPosition(data, i)

    static void _calculate_mouth_landmarks(const float* data, ARKitFaceBlendShapes& out) {
        auto upper_lip = get(::upper_lip);
        auto upper_outer_lip = get(::upper_outer_lip);
        auto lower_lip = get(::lower_lip);

        auto mouth_corner_left = get(::mouth_corner_left);
        auto mouth_corner_right = get(::mouth_corner_right);
        auto lowest_chin = get(::lowest_chin);
        auto nose_tip = get(::nose_tip);
        auto upper_head = get(::upper_head);

        auto mouth_center = (lower_lip + upper_lip) / 2;
        auto mouth_width = distance(mouth_corner_left, mouth_corner_right);
        auto mouth_open_dist = distance(upper_lip, lower_lip);
        auto mouth_top_nose_dist = distance(upper_lip, nose_tip);

        auto jaw_open_ratio = fabs(lowest_chin.y) / fabs(upper_head.y);

        out.bs[ARKit::JawOpen] = remap(ARKit::JawOpen, jaw_open_ratio);
        out.bs[ARKit::MouthClose] = clamp(out.bs[ARKit::JawOpen] - remap(ARKit::MouthClose, mouth_open_dist / mouth_top_nose_dist), 0, 1);

        auto mouth_smile_left = out.bs[ARKit::MouthSmileLeft] = remap(ARKit::MouthSmileLeft, mouth_corner_left.y - mouth_center.y);
        auto mouth_smile_right = out.bs[ARKit::MouthSmileRight] = remap(ARKit::MouthSmileRight, mouth_corner_right.y - mouth_center.y);

        out.bs[ARKit::MouthDimpleLeft] = out.bs[ARKit::MouthSmileLeft] / 2;
        out.bs[ARKit::MouthDimpleRight] = out.bs[ARKit::MouthDimpleRight] / 2;

        out.bs[ARKit::MouthFrownLeft] = 1 - remap(ARKit::MouthFrownLeft, mouth_corner_left.y - get(::mouth_frown_left).y);
        out.bs[ARKit::MouthFrownRight] = 1 - remap(ARKit::MouthFrownRight, mouth_corner_right.y - get(::mouth_frown_right).y);


        auto mouth_left_stretch_point = get(::mouth_left_stretch);
        auto mouth_right_stretch_point = get(::mouth_right_stretch);

        auto mouth_left_stretch = mouth_corner_left.x - mouth_left_stretch_point.x;
        auto mouth_right_stretch = mouth_right_stretch_point.x - mouth_corner_right.x;
        auto mouth_center_left_stretch = mouth_center.x - mouth_left_stretch_point.x;
        auto mouth_center_right_stretch = mouth_center.x - mouth_right_stretch_point.x;

        auto mouth_left = out.bs[ARKit::MouthLeft] = remap(ARKit::MouthLeft, mouth_center_left_stretch);
        auto mouth_right = out.bs[ARKit::MouthRight] = remap(ARKit::MouthRight, mouth_center_right_stretch);

        auto stretch_normal_left = -0.7 + 
            (0.42 * mouth_smile_left) + (0.36 * mouth_left);
        auto stretch_max_left = -0.45 + 
            (0.45 * mouth_smile_left) + (0.36 * mouth_left);

        auto stretch_normal_right = -0.7 + 0.42 * 
            mouth_smile_right + (0.36 * mouth_right);
        auto stretch_max_right = -0.45 + 
            (0.45 * mouth_smile_right) + (0.36 * mouth_right );

        out.bs[ARKit::MouthStretchLeft] =  remap(mouth_left_stretch, stretch_normal_left, stretch_max_left);
        out.bs[ARKit::MouthStretchRight] =  remap(mouth_right_stretch, stretch_normal_right, stretch_max_right);

        auto uppest_lip = get(0);
        auto jaw_right_left = nose_tip.x - lowest_chin.x;

        out.bs[ARKit::JawLeft] = 1 - remap(ARKit::JawLeft, jaw_right_left);
        out.bs[ARKit::JawRight] = remap(ARKit::JawRight, jaw_right_left);

        auto lowest_lip = get(::lowest_lip);
        auto under_lip = get(::under_lip);

        auto outer_lip_dist = distance(lower_lip, lowest_lip);
        auto upper_lip_dist = distance(upper_lip, upper_outer_lip);

        out.bs[ARKit::MouthPucker] = 1 - remap(ARKit::MouthPucker, mouth_width);
        out.bs[ARKit::MouthRollLower] = 1 - remap(ARKit::MouthRollLower, outer_lip_dist);
        out.bs[ARKit::MouthRollUpper] = 1 - remap(ARKit::MouthRollUpper, upper_lip_dist);

        auto upper_lip_nose_dist = nose_tip.y - uppest_lip.y;
        out.bs[ARKit::MouthShrugUpper] = 1 - remap(ARKit::MouthShrugUpper, upper_lip_nose_dist);

        auto over_upper_lip = get(::over_upper_lip);
        auto mouth_shrug_lower = distance(lowest_lip, over_upper_lip);

        out.bs[ARKit::MouthShrugLower] = 1 - remap(ARKit::MouthShrugLower, mouth_shrug_lower);

        auto lower_down_left = distance(get(424), get(319)) + mouth_open_dist * 0.5;
        auto lower_down_right = distance(get(204), get(89)) + mouth_open_dist * 0.5;

        out.bs[ARKit::MouthLowerDownLeft] = 1 - remap(ARKit::MouthLowerDownLeft, lower_down_left);
        out.bs[ARKit::MouthLowerDownRight] = 1 - remap(ARKit::MouthLowerDownRight, lower_down_right);

        // mouth funnel only can be seen if mouth pucker is really small
        if (out.bs[ARKit::MouthPucker] < 0.5) {
            out.bs[ARKit::MouthFunnel] = 1 - remap(ARKit::MouthFunnel, mouth_width);
        } 

        auto left_upper_press = distance(
            get(::left_upper_press[0]), 
            get(::left_upper_press[1])
        );
        auto left_lower_press = distance(
            get(::left_lower_press[0]), 
            get(::left_lower_press[1])
        );
        auto mouth_press_left = (left_upper_press + left_lower_press) / 2;

        auto right_upper_press = distance(
            get(::right_upper_press[0]), 
            get(::right_upper_press[1])
        );
        auto right_lower_press = distance(
            get(::right_lower_press[0]), 
            get(::right_lower_press[1])
        );
        auto mouth_press_right = (right_upper_press + right_lower_press) / 2;

        out.bs[ARKit::MouthPressLeft] = 1 - remap(ARKit::MouthPressLeft, mouth_press_left);
        out.bs[ARKit::MouthPressRight] = 1 - remap(ARKit::MouthPressRight, mouth_press_right);
    }

    static float _eye_lid_distance(const float* data, const int* eye_points) {
        auto eye_width = distance(get(
            eye_points[0]), get(eye_points[1]));
        auto eye_outer_lid = distance(get(
            eye_points[2]), get(eye_points[5]));
        auto eye_mid_lid = distance(get(
            eye_points[3]), get(eye_points[6]));
        auto eye_inner_lid = distance(get(
            eye_points[4]), get(eye_points[7]));
        auto eye_lid_avg = (eye_outer_lid + eye_mid_lid + eye_inner_lid) / 3;
        auto ratio = eye_lid_avg / eye_width;
        return ratio;
    }

    static float get_eye_open_ration(const float* data, const int* points) {
        auto eye_distance = _eye_lid_distance(data, points);
        auto max_ratio = 0.285;
        auto ratio = clamp(eye_distance / max_ratio, 0, 2);
        return ratio;
    }

    static void _calculate_eye_landmarks(const float* data, ARKitFaceBlendShapes& out) {
        auto eye_open_ratio_left = get_eye_open_ration(data, ::eye_left);
        auto eye_open_ratio_right = get_eye_open_ration(data, ::eye_right);

        out.bs[ARKit::EyeBlinkLeft] = 1 - remap(ARKit::EyeBlinkLeft, eye_open_ratio_left);
        out.bs[ARKit::EyeBlinkRight] = 1 - remap(ARKit::EyeBlinkRight, eye_open_ratio_right);

        out.bs[ARKit::EyeWideLeft] = remap(ARKit::EyeWideLeft, eye_open_ratio_left);
        out.bs[ARKit::EyeWideRight] = remap(ARKit::EyeWideRight, eye_open_ratio_right);

        auto squint_left = distance(
            get(::squint_left[0]), 
            get(::squint_left[1])
        );

        out.bs[ARKit::EyeSquintLeft] = 1 - remap(ARKit::EyeSquintLeft, squint_left);

         auto squint_right = distance(
            get(::squint_right[0]), 
            get(::squint_right[1])
        );

        out.bs[ARKit::EyeSquintRight] = 1 - remap(ARKit::EyeSquintRight, squint_right);

        auto right_brow_lower = (
            get(::right_brow_lower[0]) +
            get(::right_brow_lower[1]) +
            get(::right_brow_lower[2])
        ) / 3;
        auto right_brow_dist = distance(get(::right_brow), right_brow_lower);

        auto left_brow_lower = (
            get(::left_brow_lower[0]) +
            get(::left_brow_lower[1]) +
            get(::left_brow_lower[2])
        ) / 3;
        auto left_brow_dist = distance(get(::left_brow), left_brow_lower);

        out.bs[ARKit::BrowDownLeft] = 1 - remap(ARKit::BrowDownLeft, left_brow_dist);
        out.bs[ARKit::BrowOuterUpLeft] = remap(ARKit::BrowOuterUpLeft, left_brow_dist);

        out.bs[ARKit::BrowDownRight] = 1 - remap(ARKit::BrowDownRight, right_brow_dist);
        out.bs[ARKit::BrowOuterUpRight] = remap(ARKit::BrowOuterUpRight, right_brow_dist);

        auto inner_brow = get(::inner_brow);
        auto upper_nose = get(::upper_nose);
        auto inner_brow_dist = distance(upper_nose, inner_brow);

        out.bs[ARKit::BrowInnerUp] = remap(ARKit::BrowInnerUp, inner_brow_dist);

        auto cheek_squint_left = distance(
            get(::cheek_squint_left[0]), 
            get(::cheek_squint_left[1])
        );

        auto cheek_squint_right = distance(
            get(::cheek_squint_right[0]), 
            get(::cheek_squint_right[1])
        );

        out.bs[ARKit::CheekSquintLeft] = 1 - remap(ARKit::CheekSquintLeft, cheek_squint_left);
        out.bs[ARKit::CheekSquintRight] = 1 - remap(ARKit::CheekSquintRight, cheek_squint_right);

        // just use the same values for cheeksquint for nose sneer, mediapipe deosn't seem to have a separate value for nose sneer
        out.bs[ARKit::NoseSneerLeft] =  out.bs[ARKit::CheekSquintLeft];
        out.bs[ARKit::NoseSneerRight] =  out.bs[ARKit::CheekSquintRight];
    }
    
    void geometry2blendshape(
        const mediapipe::face_geometry::FaceGeometry& geometry,
        ARKitFaceBlendShapes& out
    ) {
        const float* data = geometry.mesh().vertex_buffer().data();

        _calculate_mouth_landmarks(data, out);
        _calculate_eye_landmarks(data, out);

        // calc head rotation
        auto euler = mat2euler(geometry.pose_transform_matrix());
        out.bs[ARKit::HeadYaw] = euler.Yaw;
        out.bs[ARKit::HeadPitch] = euler.Pitch;
        out.bs[ARKit::HeadRoll] = euler.Roll;
    }

    inline Vertex v2d(const mediapipe::NormalizedLandmark& landmark) {
        return Vertex(landmark.x(), landmark.y(), 0);
    }

    float _iris_offset(const Vertex& a, const Vertex& b, const Vertex& c, const Vertex& d, const Vertex& i) {
        auto cd = (d - c);
        cd /= cd.length();

        float da = cross2d(cd, a - d);
        float db = cross2d(cd, b - d);
        float di = cross2d(cd, i - d);
        //printf("%f %f %f ", da, db, di);
        return clamp((di - da)/ (db - da), 0, 1) - 0.5f;
    }

    void _calculate_eye_iris(
        const mediapipe::NormalizedLandmarkList& face_landmarks,
        ARKitFaceBlendShapes& out,
        const int* points,
        int irisPoint,
        const firefly::ARKit::FaceBlendShape* bsList
    ) {
        auto a = v2d(face_landmarks.landmark(points[0]));
        auto b = v2d(face_landmarks.landmark(points[1]));
        auto c = v2d(face_landmarks.landmark(points[4]));
        auto d = v2d(face_landmarks.landmark(points[6]));
        auto i = v2d(face_landmarks.landmark(irisPoint));

        float hd = _iris_offset(a,b,c,d,i) * -5 + 0.2;
        float vd = _iris_offset(d,c,b,a,i) * 3 - 0.3;
        
        if (hd < 0) {
            out.bs[bsList[1]] = std::min(1.0f, -hd);
        } else {
            out.bs[bsList[0]] = std::min(1.0f, hd);
        }

        if (vd < 0) {
            out.bs[bsList[2]] = std::min(1.0f, -hd);
        } else {
            out.bs[bsList[3]] = std::min(1.0f, hd);
        }
    }

    void iris2blendshape(
        const mediapipe::NormalizedLandmarkList& face_landmarks,
        ARKitFaceBlendShapes& out
    ) {
        _calculate_eye_iris(face_landmarks, out, eye_right, iris_right, eyeDirBSRight);
        _calculate_eye_iris(face_landmarks, out, eye_left, iris_left, eyeDirBSLeft);
    }
}
