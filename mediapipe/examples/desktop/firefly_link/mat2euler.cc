#include <math.h>
#include <type_traits>

#include "firefly_link.h"
#include "mediapipe/framework/formats/matrix_data.pb.h"

static const float _EPS4 = std::numeric_limits<float>::epsilon() * 4.0f;

namespace firefly {

    inline float get(const mediapipe::MatrixData& data, int col, int row) {
        if (data.layout() == ::mediapipe::MatrixData_Layout_ROW_MAJOR) {
            return data.packed_data(col * data.rows() + row);
        } else {
            return data.packed_data(col + row * data.cols());
        }
    }

    #define M(i, j) get(data, i, j)

    Euler mat2euler(const mediapipe::MatrixData& data) {
        int i = 0;
        int j = 1;
        int k = 2;

        float cy = sqrtf(M(i,i)*M(i,i) + M(j,i)*M(j,i));
        if (cy > _EPS4) {
            float ax = atan2f(M(k,j), M(k,k));
            float ay = atan2f(-M(k,i), cy);
            float az = atan2f(M(j,i), M(i,i));
            return Euler(ax, ay, az);
        } else {
            float ax = atan2f(-M(j,k), M(j, j));
            float ay = atan2f(-M(k, i), cy);
            return Euler(ax, ay, 0.0f);
        }
    }
}