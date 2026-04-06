struct GrayCodeSet {
    // index 0 corresponds to 1 (Scr01) if your files start at 01
    std::vector<cv::Mat> x;
    std::vector<cv::Mat> y;

    // Keep original names so we can save with identical filenames
    std::vector<std::string> xNames;
    std::vector<std::string> yNames;
};

static bool ensureDirExists(const std::filesystem::path& p) {
    std::error_code ec;
    if (std::filesystem::exists(p, ec)) return std::filesystem::is_directory(p, ec);
    return std::filesystem::create_directories(p, ec);
}

GrayCodeSet loadGrayCodeFromFolder(const std::filesystem::path& folder, bool asGray = true) {
    if (!std::filesystem::exists(folder) || !std::filesystem::is_directory(folder))
        throw std::runtime_error("Folder does not exist or is not a directory: " + folder.string());

    // Matches:
    // GrayCodeXScr01.bmp
    // GrayCodeYScr11.bmp
    // also allows .png/.jpg if you want to extend (currently bmp only, change if needed)
    const std::regex re(R"(GrayCode([XY])Scr(\d+)\.bmp$)", std::regex::icase);

    struct Entry { char axis; int idx; std::filesystem::path path; std::string name; };
    std::vector<Entry> entries;

    for (const auto& de : std::filesystem::directory_iterator(folder)) {
        if (!de.is_regular_file()) continue;

        const std::string fname = de.path().filename().string();
        std::smatch m;
        if (std::regex_match(fname, m, re)) {
            char axis = static_cast<char>(std::toupper(m[1].str()[0])); // 'X' or 'Y'
            int idx = std::stoi(m[2].str());                           // e.g. 1..11
            entries.push_back({ axis, idx, de.path(), fname });
        }
    }

    if (entries.empty())
        throw std::runtime_error("No GrayCode[X|Y]ScrXX.bmp files found in: " + folder.string());

    // Sort by axis, then idx
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        if (a.axis != b.axis) return a.axis < b.axis;
        return a.idx < b.idx;
        });

    // Determine max index for X and Y (so we can size vectors)
    int maxX = 0, maxY = 0;
    for (const auto& e : entries) {
        if (e.axis == 'X') maxX = std::max(maxX, e.idx);
        else               maxY = std::max(maxY, e.idx);
    }

    GrayCodeSet set;
    set.x.resize(maxX);
    set.y.resize(maxY);
    set.xNames.resize(maxX);
    set.yNames.resize(maxY);

    const int imreadFlag = asGray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_UNCHANGED;

    for (const auto& e : entries) {
        cv::Mat img = cv::imread(e.path.string(), imreadFlag);
        if (img.empty())
            throw std::runtime_error("Failed to read image: " + e.path.string());

        // idx is 1-based in filename -> store at idx-1
        const int pos = e.idx - 1;
        if (pos < 0) throw std::runtime_error("Invalid index in filename: " + e.name);

        if (e.axis == 'X') {
            if (pos >= static_cast<int>(set.x.size()))
                throw std::runtime_error("Index out of range for X: " + e.name);
            set.x[pos] = img;
            set.xNames[pos] = e.name;
        }
        else {
            if (pos >= static_cast<int>(set.y.size()))
                throw std::runtime_error("Index out of range for Y: " + e.name);
            set.y[pos] = img;
            set.yNames[pos] = e.name;
        }
    }


    return set;
}

static std::string makeName(char axis, size_t i) {
    std::ostringstream oss;
    oss << "GrayCode" << axis << "Scr"
        << std::setw(2) << std::setfill('0') << (i + 1)
        << ".bmp";
    return oss.str();
}

void saveGrayCodeToFolder(const GrayCodeSet& set, const std::filesystem::path& outFolder) {
    if (!ensureDirExists(outFolder))
        throw std::runtime_error("Could not create output directory: " + outFolder.string());

    // Save X
    for (size_t i = 0; i < set.x.size(); ++i) {
        if (set.x[i].empty()) continue;

        const std::filesystem::path outPath = outFolder /
            (set.xNames[i].empty() ? makeName('X', i) : set.xNames[i]);

        if (!cv::imwrite(outPath.string(), set.x[i]))
            throw std::runtime_error("Failed to write: " + outPath.string());
    }

    // Save Y
    for (size_t i = 0; i < set.y.size(); ++i) {
        if (set.y[i].empty()) continue;

        const std::filesystem::path outPath = outFolder /
            (set.yNames[i].empty() ? makeName('Y', i) : set.yNames[i]);

        if (!cv::imwrite(outPath.string(), set.y[i]))
            throw std::runtime_error("Failed to write: " + outPath.string());
    }
}