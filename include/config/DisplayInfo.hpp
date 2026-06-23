#ifndef DISPALYINFO_H
#define DISPLAYINFO_H
#include <iostream>
#include <windows.h>

namespace defl {

    struct DisplayInfo {
        int posx = 0;
        int posy = 0;
        int width = 1920;
        int height = 1080;

        float width_mm = 527.04f;
        float height_mm = 296.46f;
        float diagonal_mm = 605.0f;
        float pixel_pitch_mm = 0.2745f;

        static DisplayInfo queryPrimaryOrSecondMonitor() {
            DisplayInfo info;

            DISPLAY_DEVICE dd;
            ZeroMemory(&dd, sizeof(dd));
            dd.cb = sizeof(dd);

            bool foundSecond = false;

            for (DWORD i = 0; EnumDisplayDevices(nullptr, i, &dd, 0); ++i) {
                if (dd.StateFlags & DISPLAY_DEVICE_ACTIVE) {
                    if (i == 1) {
                        foundSecond = true;
                        break;
                    }
                }
            }

            DEVMODE dm;
            ZeroMemory(&dm, sizeof(dm));
            dm.dmSize = sizeof(dm);

            if (foundSecond && EnumDisplaySettings(dd.DeviceName, ENUM_CURRENT_SETTINGS, &dm)) {
                info.posx = dm.dmPosition.x;
                info.posy = dm.dmPosition.y;
                info.width = dm.dmPelsWidth;
                info.height = dm.dmPelsHeight;
            }

            return info;
        }
    };

} // namespace defl

#endif
