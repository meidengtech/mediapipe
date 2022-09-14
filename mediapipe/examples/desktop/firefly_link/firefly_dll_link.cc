#include "firefly_link.h"
#include "mediapipe/util/resource_util_custom.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/file_helpers.h"
#include <Windows.h>

BOOL APIENTRY DllMain(HMODULE hModule,
                   DWORD  ul_reason_for_call,
                   LPVOID lpReserved
                 ) {
    switch (ul_reason_for_call)
    { 
        case DLL_PROCESS_ATTACH:
        {
            char dllFilePath[512 + 1] = { 0 };
            GetModuleFileNameA(hModule, dllFilePath, 512);
            for (size_t l = strlen(dllFilePath); l >= 0; l--) {
                if (dllFilePath[l] == '/' || dllFilePath[l] == '\\') {
                    break;
                }
                dllFilePath[l] = 0;
            }
            std::string basePath = dllFilePath;
            OutputDebugStringA((std::string("Base Path: ") + basePath + "\n").c_str());
            
            mediapipe::SetCustomGlobalResourceProvider([=](const std::string& resource, std::string* out) {
                std::string realPath;
                if (resource[0] == '/' || resource[0] == '\\') {
                    realPath =  basePath + resource.substr(1);
                } else {
                    realPath =  basePath + resource;
                }
                OutputDebugStringA((*out + "\n").c_str());
                return mediapipe::file::GetContents(realPath, out, true);
            });
        }
        break;
    }
    return TRUE;  // Successful DLL_PROCESS_ATTACH.
}

extern "C" __declspec(dllexport) bool firefly_init(int cameraId) {
    auto s =  firefly::init(cameraId);
    if (!s.ok()) {
        OutputDebugStringA(s.ToString().c_str());
        return false;
    }
    return true;
}

extern "C" __declspec(dllexport) bool firefly_run(firefly::ARKitFaceBlendShapes* out, std::function<void (const void* data, int width, int height)> preview) {
    auto s = firefly::run(out, false, preview);
    if (!s.ok()) {
        OutputDebugStringA(s.ToString().c_str());
        return false;
    }
    return true;
}

extern "C" __declspec(dllexport) void firefly_shutdown() {
    firefly::shutdown();
}
