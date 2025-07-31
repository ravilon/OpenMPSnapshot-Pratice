#include <windows.h>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>

using std::vector;

static const int PERM[512] = {
        151,160,137, 91, 90,15,131, 13,201, 95, 96,53,194,233,  7,225,
        140, 36,103, 30, 69,142,  8, 99, 37,240, 21, 10, 23,190,  6,148,
        247,120,234, 75,  0, 26,197, 62, 94,252,219,203,117, 35, 11, 32,
         57,177, 33, 88,237,149, 56, 87,174, 20,125,136,171,168, 68,175,
         74,165, 71,134,139, 48, 27,166, 77,146,158,231, 83,111,229,122,
         60,211,133,230,220,105, 92, 41, 55, 46,245, 40,244,102,143, 54,
         65, 25, 63,161,  1,216, 80, 73,209, 76,132,187,208, 89, 18,169,
        200,196,135,130,116,188,159, 86,164,100,109,198,173,186,  3, 64,
         52,217,226,250,124,123,  5,202, 38,147,118,126,255, 82, 85,212,
        207,206, 59,227, 47, 16, 58, 17,182,189, 28, 42,223,183,170,213,
        119,248,152,  2, 44,154,163, 70,221,153,101,155,167, 43,172,  9,
        129, 22, 39,253, 19, 98,108,110, 79,113,224,232,178,185,112,104,
        218,246, 97,228,251, 34,242,193,238,210,144, 12,191,179,162,241,
         81, 51,145,235,249, 14,239,107, 49,192,214, 31,181,199,106,157,
        184, 84,204,176,115,121, 50, 45,127,  4,150,254,138,236,205, 93,
        222,114, 67, 29, 24, 72,243,141,128,195, 78, 66,215, 61,156,180,
        // repeat so we can index with &255 without mod
        151,160,137, 91, 90,15,131, 13,201, 95, 96,53,194,233,  7,225,
        140, 36,103, 30, 69,142,  8, 99, 37,240, 21, 10, 23,190,  6,148,
        247,120,234, 75,  0, 26,197, 62, 94,252,219,203,117, 35, 11, 32,
         57,177, 33, 88,237,149, 56, 87,174, 20,125,136,171,168, 68,175,
         74,165, 71,134,139, 48, 27,166, 77,146,158,231, 83,111,229,122,
         60,211,133,230,220,105, 92, 41, 55, 46,245, 40,244,102,143, 54,
         65, 25, 63,161,  1,216, 80, 73,209, 76,132,187,208, 89, 18,169,
        200,196,135,130,116,188,159, 86,164,100,109,198,173,186,  3, 64,
         52,217,226,250,124,123,  5,202, 38,147,118,126,255, 82, 85,212,
        207,206, 59,227, 47, 16, 58, 17,182,189, 28, 42,223,183,170,213,
        119,248,152,  2, 44,154,163, 70,221,153,101,155,167, 43,172,  9,
        129, 22, 39,253, 19, 98,108,110, 79,113,224,232,178,185,112,104,
        218,246, 97,228,251, 34,242,193,238,210,144, 12,191,179,162,241,
         81, 51,145,235,249, 14,239,107, 49,192,214, 31,181,199,106,157,
        184, 84,204,176,115,121, 50, 45,127,  4,150,254,138,236,205, 93,
        222,114, 67, 29, 24, 72,243,141,128,195, 78, 66,215, 61,156,180
};

// ---- Configuration ----
const int ROWS = 300;
const int COLS = 400;
const int CELL_SIZE = 2;
const int MARGIN = 5;
const int BORDER = 2;

#define ID_TIMER        102

// ---- Application State ----
struct AppState {
    HINSTANCE hInst;
    vector<uint8_t> grid;
    vector<uint8_t> nextGrid;

    // Back-buffer via DIB
    HDC     memDC;
    HBITMAP memBmp;
    void* pixels;
    int     dibW;
    int     dibH;
    int     pitch;

    AppState(HINSTANCE h)
        : hInst(h)
    {
        grid.assign(ROWS * COLS, 0);
        nextGrid.assign(ROWS * COLS, 0);
        dibW = COLS * CELL_SIZE + 2 * BORDER;
        dibH = ROWS * CELL_SIZE + 2 * BORDER;
        pitch = ((dibW * 3 + 3) / 4) * 4;
        pixels = nullptr;
        memDC = nullptr;
        memBmp = nullptr;
    }

    void InitializeGridUniform() {
        std::mt19937_64 rng(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()
        );
        std::uniform_int_distribution<int> coin(0, 1);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < ROWS * COLS; ++i) {
            grid[i] = coin(rng);
        }
    }

    void InitializeGridUniform_LeapFrog() {
        unsigned long long seed =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int> coin(0, 1);

        // Precompute all random numbers sequentially
        std::vector<int> random_numbers(ROWS * COLS);
        for (int i = 0; i < ROWS * COLS; ++i) {
            random_numbers[i] = coin(rng);
        }

#pragma omp parallel for schedule(static)
        for (int i = 0; i < ROWS * COLS; ++i) {
            grid[i] = random_numbers[i];
        }
    }

    // inside AppState:

    /// Returns true if perlin noise at (x,y) > threshold.
    void InitializeGridPerlin(double frequency, double threshold)
    {
        // Parallel over flat array
#pragma omp parallel for schedule(static)
        for (int i = 0; i < ROWS * COLS; ++i) {
            int r = i / COLS;
            int c = i % COLS;

            // sample coordinates scaled by frequency
            double x = c * frequency;
            double y = r * frequency;

            // get noise in [0,1]
            double n = perlin(x, y);

            // alive if above threshold
            grid[i] = (n > threshold) ? 1 : 0;
        }
    }
    // --- Perlin noise support ---

    // Fade (6t^5 - 15t^4 + 10t^3)
    static double fade(double t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }
    // Linear interpolation
    static double lerp(double a, double b, double t) {
        return a + t * (b - a);
    }
    // Gradient: pick one of 8 directions & dot with (x,y)
    static double grad(int hash, double x, double y) {
        int h = hash & 7;           // lower 3 bits
        double u = (h < 4 ? x : y);
        double v = (h < 4 ? y : x);
        return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
    }

    // 2D Perlin noise in [0,1]
    double perlin(double x, double y) {
        int xi = int(floor(x)) & 255;
        int yi = int(floor(y)) & 255;
        double xf = x - floor(x), yf = y - floor(y);
        double u = fade(xf), v = fade(yf);

        int aa = PERM[PERM[xi] + yi];
        int ab = PERM[PERM[xi] + yi + 1];
        int ba = PERM[PERM[xi + 1] + yi];
        int bb = PERM[PERM[xi + 1] + yi + 1];

        double x1 = lerp(grad(aa, xf, yf),grad(ba, xf - 1, yf), u);
        double x2 = lerp(grad(ab, xf, yf - 1), grad(bb, xf - 1, yf - 1), u);

        double value = lerp(x1, x2, v);
        return (value + 1.0) * 0.5;  // scale to [0,1]
    }



    void UpdateGrid() {
#pragma omp parallel for schedule(static)
        for (int idx = 0; idx < ROWS * COLS; ++idx) {
            int r = idx / COLS;
            int c = idx % COLS;
            int n = 0;
            for (int dr = -1; dr <= 1; ++dr) {
                int rr = r + dr;
                if (rr < 0 || rr >= ROWS) continue;
                for (int dc = -1; dc <= 1; ++dc) {
                    int cc = c + dc;
                    if ((dr | dc) == 0 || cc < 0 || cc >= COLS) continue;
                    n += grid[rr * COLS + cc];
                }
            }
            bool alive = grid[idx];
            if (alive) {
                nextGrid[idx] = (n == 2 || n == 3) ? 1 : 0;
            }
            else {
                nextGrid[idx] = (n == 3) ? 1 : 0;
            }
        }
        grid.swap(nextGrid);
    }

    void RenderToBackBuffer() {
        uint8_t* px = static_cast<uint8_t*>(pixels);
        // fill white background
#pragma omp parallel for schedule(static)
        for (int y = 0; y < dibH; ++y) {
            uint8_t* row = px + y * pitch;
            memset(row, 255, dibW * 3);
        }
        // vertical borders
#pragma omp parallel for schedule(static)
        for (int y = 0; y < dibH; ++y) {
            uint8_t* row = px + y * pitch;
            for (int b = 0; b < BORDER; ++b) {
                row[b * 3 + 0] = 0; row[b * 3 + 1] = 0; row[b * 3 + 2] = 0;
                int base = (dibW - 1 - b) * 3;
                row[base + 0] = 0; row[base + 1] = 0; row[base + 2] = 0;
            }
        }
        // horizontal borders
#pragma omp parallel for schedule(static)
        for (int x = 0; x < dibW; ++x) {
            uint8_t* top = px + 0 * pitch + x * 3;
            uint8_t* bot = px + (dibH - 1) * pitch + x * 3;
            top[0] = top[1] = top[2] = 0;
            bot[0] = bot[1] = bot[2] = 0;
        }
        // draw cells
#pragma omp parallel for schedule(static)
        for (int idx = 0; idx < ROWS * COLS; ++idx) {
            if (!grid[idx]) continue;
            int r = idx / COLS;
            int c = idx % COLS;
            int y0 = BORDER + r * CELL_SIZE;
            int x0 = BORDER + c * CELL_SIZE;
            for (int dy = 0; dy < CELL_SIZE; ++dy) {
                uint8_t* row = px + (y0 + dy) * pitch + x0 * 3;
                for (int dx = 0; dx < CELL_SIZE; ++dx) {
                    row[0] = row[1] = row[2] = 0;
                    row += 3;
                }
            }
        }
    }
};

inline AppState* GetState(HWND hwnd) {
    return reinterpret_cast<AppState*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_CREATE: {
        AppState* state = reinterpret_cast<AppState*>(((CREATESTRUCT*)lParam)->lpCreateParams);
        SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(state));
        //state->InitializeGridPerlin(0.1, 0.5);
        state->InitializeGridUniform_LeapFrog();
        HDC hdc = GetDC(hwnd);
        state->memDC = CreateCompatibleDC(hdc);
        BITMAPINFO bmi = {};
        bmi.bmiHeader.biSize = sizeof(bmi.bmiHeader);
        bmi.bmiHeader.biWidth = state->dibW;
        bmi.bmiHeader.biHeight = -state->dibH;
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 24;
        bmi.bmiHeader.biCompression = BI_RGB;
        state->memBmp = CreateDIBSection(state->memDC, &bmi, DIB_RGB_COLORS,
            &state->pixels, nullptr, 0);
        SelectObject(state->memDC, state->memBmp);
        ReleaseDC(hwnd, hdc);
        SetTimer(hwnd, ID_TIMER, 100, nullptr);
        break;
    }
    case WM_TIMER:
        if (wParam == ID_TIMER) {
            AppState* state = GetState(hwnd);
            state->UpdateGrid();
            state->RenderToBackBuffer();
            InvalidateRect(hwnd, nullptr, FALSE);
        }
        break;
    case WM_PAINT: {
        AppState* state = GetState(hwnd);
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);
        BitBlt(
            hdc,
            MARGIN,
            MARGIN,
            state->dibW,
            state->dibH,
            state->memDC,
            0, 0,
            SRCCOPY
        );
        EndPaint(hwnd, &ps);
        break;
    }
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hwnd, msg, wParam, lParam);
    }
    return 0;
}

int APIENTRY wWinMain(HINSTANCE hInst, HINSTANCE, LPWSTR, int nCmdShow) {
    AppState* state = new AppState(hInst);
    WNDCLASSEX wc = { sizeof(wc) };
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = L"GameOfLifeClass";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
    RegisterClassEx(&wc);
    int clientW = COLS * CELL_SIZE + 2 * BORDER + 2 * MARGIN;
    int clientH = ROWS * CELL_SIZE + 2 * BORDER + 2 * MARGIN;
    RECT wr = { 0,0,clientW,clientH };
    DWORD style = WS_OVERLAPPEDWINDOW & ~(WS_THICKFRAME | WS_MAXIMIZEBOX);
    AdjustWindowRect(&wr, style, FALSE);
    HWND hwnd = CreateWindowEx(
        0, wc.lpszClassName,
        L"Game of Life",
        style,
        CW_USEDEFAULT, CW_USEDEFAULT,
        wr.right - wr.left,
        wr.bottom - wr.top,
        nullptr, nullptr, hInst, state);
    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);
    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    delete state;
    return (int)msg.wParam;
}