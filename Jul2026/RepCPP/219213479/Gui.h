/**
 *    Copyright 2020 Jannik Bamberger
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <QFileDialog>
#include <QImage>
#include <QLabel>
#include <QMainWindow>
#include <QMenuBar>
#include <QPushButton>
#include <QStatusBar>
#include <QToolBar>
#include <utility>

#include "PathTracer.h"
#include "Scene.h"
#include "Viewer.h"

/**
 * Host window of the ray tracing application.
 */
class Gui : public QMainWindow {
    /**
     * Viewer widget that hosts the ray tracer.
     */
    Viewer* viewer_;

  public:
    Gui() = delete;

    /**
     * Creates a new instance of the main window.
     * @param width window width
     * @param height window height
     * @param raytracer raytracer object
     * @param scene scene object
     */
    Gui(int width,
        int height,
        std::shared_ptr<PathTracer> raytracer,
        std::shared_ptr<Scene> scene,
        QWindow* = nullptr);

    ~Gui() override;
};
