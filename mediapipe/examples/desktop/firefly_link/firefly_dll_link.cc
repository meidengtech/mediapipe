#include "firefly_link.h"

extern "C" __declspec(dllexport) bool firefly_init(int cameraId) {
    return firefly::init(cameraId);
}

extern "C" __declspec(dllexport) bool firefly_run(firefly::ARKitFaceBlendShapes* out) {
    return firefly::run(out);
}

extern "C" __declspec(dllexport) void firefly_shutdown() {
    firefly::shutdown();
}
