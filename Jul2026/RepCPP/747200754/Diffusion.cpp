// TODO: remove the following rule disable and fix the issue
// ReSharper disable CppDFAUnusedValue CppDFAUnreadVariable CppDFALoopConditionNotUpdated
#include "Diffusion.hpp"
#include "ExplicitScheme.hpp"

#include <cmath>
#include <iostream>

Diffusion::Diffusion(const InputFile& input, const std::shared_ptr<Mesh>& m) : mesh{m} {
    if (const std::string scheme_str = input.getString("scheme", "explicit"); scheme_str == "explicit") {
        scheme = std::make_unique<ExplicitScheme>(mesh);
    } else {
        std::cerr << "Error: unknown scheme \"" << scheme_str << "\"" << '\n';
        std::exit(1);
    }

    subregion = input.getDoubleList("subregion", std::vector<double>{});

    if (!subregion.empty() && subregion.size() != 4) {
        std::cerr << "Error:  region must have 4 entries (xmin, ymin, xmax, ymax)" << '\n';
        std::exit(1);
    }

    init();
}

void Diffusion::init() const {
    if (subregion.empty()) {
        return;
    }

    auto& u0 = mesh->getU0();

    const int nx = mesh->getNx()[0] + 2;

    const int x_min_pos = static_cast<int>(std::ceil(subregion[0]));
    const int x_max_pos = static_cast<int>(std::floor(subregion[2]));
    const int y_min_pos = static_cast<int>(std::ceil(subregion[1]));
    const int y_max_pos = static_cast<int>(std::floor(subregion[3]));

#pragma omp parallel default(none) shared(u0) firstprivate(nx, x_min_pos, x_max_pos, y_min_pos, y_max_pos)
    {
#pragma omp for collapse(2) schedule(static) nowait
        for (int j = y_min_pos; j <= y_max_pos; ++j) {
            for (int i = x_min_pos; i <= x_max_pos; ++i) {
                u0[i + j * nx] = 10.0;
            }
        }
    }

    scheme->init();
}

void Diffusion::doCycle(const double& dt) const {
    scheme->doAdvance(dt);
}
