// ======================================================================== //
// Copyright 2017-2018 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <random>
#include <algorithm>
#include <array>
#include <chrono>
#include <random>
#include <GLFW/glfw3.h>
#include <mpiCommon/MPICommon.h>
#include <mpi.h>
#include <ospcommon/utility/SaveImage.h>
#include <ospray/ospray_cpp/Camera.h>
#include <ospray/ospray_cpp/Data.h>
#include <ospray/ospray_cpp/Device.h>
#include <ospray/ospray_cpp/FrameBuffer.h>
#include <ospray/ospray_cpp/Geometry.h>
#include <ospray/ospray_cpp/Renderer.h>
#include <ospray/ospray_cpp/TransferFunction.h>
#include <ospray/ospray_cpp/Volume.h>
#include <ospray/ospray_cpp/Model.h>
#include <ospcommon/containers/AlignedVector.h>
//#include "widgets/transferFunction.h"
//#include "common/sg/transferFunction/TransferFunction.h"
#include "imgui.h"
#include "imgui_impl_glfw_gl3.h"
#include "gensv/generateSciVis.h"
#include "arcball.h"
#include "gensv/llnlrm_reader.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

/* This app demonstrates how to write an distributed scivis style
 * interactive renderer using the distributed MPI device. Note that because
 * OSPRay uses sort-last compositing it is up to the user to ensure
 * that the data distribution across the nodes is suitable. Specifically,
 * each nodes' data must be convex and disjoint. This renderer only
 * supports multiple volumes and geometries per-node, to ensure they're
 * composited correctly you specify a list of bounding regions to the
 * model, within these regions can be arbitrary volumes/geometrys
 * and each rank can have as many regions as needed. As long as the
 * regions are disjoint/convex the data will be rendered correctly.
 * In this example we set two regions on certain ranks just to produce
 * a gap in the ranks volume to demonstrate how they work.
 *
 * In the case that you have geometry crossing the boundary of nodes
 * and are replicating it on both nodes to render (ghost zones, etc.)
 * the region will be used by the renderer to clip rays against allowing
 * to split the object between the two nodes, with each rendering half.
 * This will keep the regions rendered by each rank disjoint and thus
 * avoid any artifacts. For example, if a sphere center is on the border
 * between two nodes, each would render half the sphere and the halves
 * would be composited to produce the final complete sphere in the image.
 *
 * See gensv::loadVolume for an example of how to properly load a volume
 * distributed across ranks with correct specification of brick positions
 * and ghost voxels. If no volume file data is passed a volume will be
 * generated instead, in that case see gensv::makeVolume.
 */

using namespace ospray::cpp;
using namespace ospcommon;

// Commandline params
std::string volumeFile = "";
std::string dtype = "";
vec3i dimensions = vec3i(-1);
vec2f valueRange = vec2f(-1);
size_t nSpheres = 0;
float varianceThreshold = 0.0f;
FileName transferFcnFile;
bool appInitMPI = false;
size_t nlocalBricks = 1;
float sphereRadius = 0.005;
bool transparentSpheres = false;
int aoSamples = 0;
int nBricks = -1;
int bricksPerRank = 1;
bool llnlrm = false;
bool fbnone = false;
vec2i imgSize(512, 512);
int framesToRender = std::numeric_limits<int>::max();
std::string outputFilename;
std::vector<float> isovalues;
bool no_volume = false;
bool autoRotate = false;
bool autoRandomCamera = false;
bool transparentIsosurfaces = false;
std::string cosmicWebPath = "";
vec3i cosmicWebGrid = vec3i(8);
vec3f bgColor = vec3f(0.02, 0.02, 0.02);

bool cmdlineViewParams = false;
vec3f cmdlineEye = vec3f(0);
vec3f cmdlineTarget = vec3f(0);
vec3f cmdlineUp = vec3f(0);
float opacityScaling = 1.0;
int shadowsEnabled = 0;

std::vector<std::string> replicatedObjs;
std::vector<std::string> osp_bricks;
bool showGUI = true;

// Struct for bcasting out the camera change info and general app state
struct AppState {
  // eye pos, look dir, up dir
  std::array<vec3f, 3> v;
  vec2i fbSize;
  bool cameraChanged, quit, fbSizeChanged, tfcnChanged;

  AppState() : fbSize(imgSize), cameraChanged(false), quit(false),
    fbSizeChanged(false)
  {}
};

// Extra stuff we need in GLFW callbacks
struct WindowState {
  Arcball &camera;
  vec2f prevMouse;
  bool cameraChanged;
  AppState &app;
  bool isImGuiHovered;

  WindowState(AppState &app, Arcball &camera)
    : camera(camera), prevMouse(-1), cameraChanged(false), app(app),
    isImGuiHovered(false)
  {}
};

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
  WindowState *state = static_cast<WindowState*>(glfwGetWindowUserPointer(window));
  if (action == GLFW_PRESS) {
    switch (key) {
      case GLFW_KEY_ESCAPE:
        glfwSetWindowShouldClose(window, true);
        break;
      case GLFW_KEY_P: {
          const vec3f eye = state->camera.eyePos();
          const vec3f center = state->camera.center();
          const vec3f up = state->camera.upDir();
          std::cout << "-vp " << eye.x << " " << eye.y << " " << eye.z
            << "\n-vu " << up.x << " " << " " << up.y << " " << up.z
            << "\n-vi " << center.x << " " << center.y << " " << center.z
            << "\n";
        }
        break;
      case GLFW_KEY_G: showGUI = !showGUI;
      default:
        break;
    }
  }
  // Forward on to ImGui
  ImGui_ImplGlfwGL3_KeyCallback(window, key, scancode, action, mods);
}
void cursorPosCallback(GLFWwindow *window, double x, double y) {
  WindowState *state = static_cast<WindowState*>(glfwGetWindowUserPointer(window));
  if ((showGUI && state->isImGuiHovered) || autoRotate || autoRandomCamera) {
    return;
  }
  const vec2f mouse(x, y);
  if (state->prevMouse != vec2f(-1)) {
    const bool leftDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    const bool rightDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    const bool middleDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
    const vec2f prev = state->prevMouse;
    state->cameraChanged = leftDown || rightDown || middleDown;

    if (leftDown) {
      const vec2f mouseFrom(clamp(prev.x * 2.f / state->app.fbSize.x - 1.f,  -1.f, 1.f),
                            clamp(prev.y * 2.f / state->app.fbSize.y - 1.f,  -1.f, 1.f));
      const vec2f mouseTo(clamp(mouse.x * 2.f / state->app.fbSize.x - 1.f,  -1.f, 1.f),
                          clamp(mouse.y * 2.f / state->app.fbSize.y - 1.f,  -1.f, 1.f));
      state->camera.rotate(mouseFrom, mouseTo);
    } else if (rightDown) {
      state->camera.zoom(mouse.y - prev.y);
    } else if (middleDown) {
      state->camera.pan(vec2f(prev.x - mouse.x, prev.y - mouse.y));
    }
  }
  state->prevMouse = mouse;
}
void framebufferSizeCallback(GLFWwindow *window, int width, int height) {
  WindowState *state = static_cast<WindowState*>(glfwGetWindowUserPointer(window));
  state->app.fbSize = vec2i(width, height);
  state->app.fbSizeChanged = true;
}
void charCallback(GLFWwindow *, unsigned int c) {
  ImGuiIO& io = ImGui::GetIO();
  if (c > 0 && c < 0x10000) {
    io.AddInputCharacter((unsigned short)c);
  }
}

void parseArgs(int argc, char **argv)
{
  for (int i = 0; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-f") {
      volumeFile = argv[++i];
    } else if (arg == "-dtype") {
      dtype = argv[++i];
    } else if (arg == "-dims") {
      dimensions.x = std::atoi(argv[++i]);
      dimensions.y = std::atoi(argv[++i]);
      dimensions.z = std::atoi(argv[++i]);
    } else if (arg == "-range") {
      valueRange.x = std::atof(argv[++i]);
      valueRange.y = std::atof(argv[++i]);
    } else if (arg == "-spheres") {
      nSpheres = std::atol(argv[++i]);
    } else if (arg == "-variance") {
      varianceThreshold = std::atof(argv[++i]);
    } else if (arg == "-tfn") {
      transferFcnFile = argv[++i];
    } else if (arg == "-appMPI") {
      appInitMPI = true;
    } else if (arg == "-nlocal-bricks") {
      nlocalBricks = std::stol(argv[++i]);
    } else if (arg == "-radius") {
      sphereRadius = std::stof(argv[++i]);
    } else if (arg == "-transparent-spheres") {
      transparentSpheres = true;
    } else if (arg == "-ao") {
      aoSamples = std::stoi(argv[++i]);
    } else if (arg == "-nbricks") {
      nBricks = std::stoi(argv[++i]);
    } else if (arg == "-bpr") {
      bricksPerRank = std::stoi(argv[++i]);
    } else if (arg == "-bob") {
      llnlrm = true;
    } else if (arg == "-osp-brick") {
      ++i;
      for (; i < argc && argv[i][0] != '-'; ++i) {
        osp_bricks.push_back(argv[i]);
      }
      // "Unread" the param if we hit a following param, to not lose it
      if (i != argc && argv[i][0] == '-') {
        --i;
      }
    } else if (arg == "-fb-none") {
      fbnone = true;
    } else if (arg == "-o" ) {
      outputFilename = argv[++i];
    } else if (arg == "-nf") {
      framesToRender = std::atoi(argv[++i]);
    } else if (arg == "-isovals") {
      ++i;
      for (; i < argc && argv[i][0] != '-'; ++i) {
        isovalues.push_back(std::atof(argv[i]));
      }
      // "Unread" the param if we hit a following param, to not lose it
      if (i != argc && argv[i][0] == '-') {
        --i;
      }
    } else if (arg == "-no-volume") {
      no_volume = true;
    } else if (arg == "-rotate") {
      autoRotate = true;
    } else if (arg == "-random-cam") {
      autoRandomCamera = true;
    } else if (arg == "-cosmic-web") {
      cosmicWebPath = argv[++i];
      cosmicWebGrid.x = std::atoi(argv[++i]);
      cosmicWebGrid.y = std::atoi(argv[++i]);
      cosmicWebGrid.z = std::atoi(argv[++i]);
    } else if (arg == "-transparent-iso") {
      transparentIsosurfaces = true;
    } else if (arg == "-w") {
      imgSize.x = std::atoi(argv[++i]);
    } else if (arg == "-h") {
      imgSize.y = std::atoi(argv[++i]);
    } else if (arg == "--bgColor") {
      bgColor.x = std::atof(argv[++i]);
      bgColor.y = std::atof(argv[++i]);
      bgColor.z = std::atof(argv[++i]);
    } else if (arg == "-vp") {
      cmdlineEye.x = std::atof(argv[++i]);
      cmdlineEye.y = std::atof(argv[++i]);
      cmdlineEye.z = std::atof(argv[++i]);
      cmdlineViewParams = true;
    } else if (arg == "-vi") {
      cmdlineTarget.x = std::atof(argv[++i]);
      cmdlineTarget.y = std::atof(argv[++i]);
      cmdlineTarget.z = std::atof(argv[++i]);
      cmdlineViewParams = true;
    } else if (arg == "-vu") {
      cmdlineUp.x = std::atof(argv[++i]);
      cmdlineUp.y = std::atof(argv[++i]);
      cmdlineUp.z = std::atof(argv[++i]);
      cmdlineViewParams = true;
    } else if (arg == "--opacity-scaling") {
      opacityScaling = std::atof(argv[++i]);
    } else if (arg == "-shadows") {
      shadowsEnabled = 1;
    } else if (arg == "-repl-objs") {
      ++i;
      for (; i < argc && argv[i][0] != '-'; ++i) {
        replicatedObjs.push_back(argv[i]);
      }
      // "Unread" the param if we hit a following param, to not lose it
      if (i != argc && argv[i][0] == '-') {
        --i;
      }
    }
  }
  if (!volumeFile.empty() && !llnlrm) {
    if (dtype.empty()) {
      std::cerr << "Error: -dtype (uchar|char|float|double) is required\n";
      std::exit(1);
    }
    if (dimensions == vec3i(-1)) {
      std::cerr << "Error: -dims X Y Z is required to pass volume dims\n";
      std::exit(1);
    }
    if (valueRange == vec2f(-1)) {
      std::cerr << "Error: -range X Y is required to set transfer function range\n";
      std::exit(1);
    }
    if (nlocalBricks != 1) {
      std::cerr << "Error: -nlocal-bricks only makes supported for generated volumes\n";
      std::exit(1);
    }
  }
}

void runApp()
{
  ospLoadModule("mpi");
  Device device("mpi_distributed");
  device.set("masterRank", 0);
  ospDeviceSetStatusFunc(device.handle(),
                         [](const char *msg) {
                           std::cout << "OSP Status: " << msg << "\n";
                         });
  ospDeviceSetErrorFunc(device.handle(),
                        [](OSPError err, const char *msg) {
                          std::cout << "OSP Error: " <<  msg << "\n";
                        });
  device.commit();
  device.setCurrent();

  const int rank = mpicommon::world.rank;
  const int worldSize = mpicommon::world.size;
  std::cout << "Rank " << rank << "/" << worldSize << "\n";

  const std::vector<vec3f> isosurfaceColors = {
    vec3f(0.17647058823529413, 0.3176470588235294, 0.6392156862745098),
    vec3f(0.19215686274, 0.43529411764, 0.20392156862),
    vec3f(0.20392156862, 0.76078431372, 0.86666666666),
    vec3f(0.65098039215, 0.8431372549, 0.42352941176)
  };
  std::vector<Material> isosurfaceMats;
  for (const auto &c : isosurfaceColors) {
    Material m("scivis", "OBJMaterial");
    m.set("Kd", c);
    m.set("Ks", vec3f(0.5));
    m.set("Ns", 30.f);
    if (transparentIsosurfaces) {
      m.set("d", 0.3f);
    }
    m.commit();
    isosurfaceMats.push_back(m);
  }

  AppState app;
  containers::AlignedVector<gensv::LoadedVolume> volumes;
  box3f worldBounds;
  containers::AlignedVector<Model> models, ghostModels;
  if (cosmicWebPath.empty()) {
    if (!osp_bricks.empty()) {
      if (osp_bricks.size() != worldSize) {
        throw std::runtime_error("OSP Brick count must match number of ranks");
      }
      const std::string my_brick = osp_bricks[rank];
      std::cout << "Rank " << rank << " loading brick " << my_brick << "\n";
      volumes.push_back(gensv::loadOSPBrick(my_brick, valueRange));

      MPI_Allreduce(&volumes[0].bounds.lower, &worldBounds.lower, 3,
          MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&volumes[0].bounds.upper, &worldBounds.upper, 3,
          MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

      std::cout << "World bounds: " << worldBounds << "\n";

    } else if (!volumeFile.empty()) {
      if (nBricks == -1) {
        nBricks = worldSize;
      }
      if (!llnlrm) {
        using namespace std::chrono;
        auto beginLoad = high_resolution_clock::now();
        volumes = gensv::loadBrickedVolume(volumeFile,
                                           dimensions,
                                           dtype,
                                           valueRange,
                                           nBricks,
                                           bricksPerRank,
                                           isovalues,
                                           no_volume);
        auto endLoad = high_resolution_clock::now();
        std::cout << "Loading on rank " << rank << " took "
          << duration_cast<seconds>(endLoad - beginLoad).count()
          << "s\n";
      } else {
        volumes = gensv::loadRMBricks(volumeFile, bricksPerRank);
        dimensions = gensv::LLNLRMReader::dimensions();
      }

      worldBounds = box3f(vec3f(0), vec3f(dimensions));

      // Pick a nice sphere radius for a consisten voxel size to
      // sphere size ratio
      sphereRadius *= dimensions.x;
    } else {
      volumes = gensv::makeVolumes(rank * nlocalBricks, nlocalBricks,
          worldSize * nlocalBricks);

      const vec3f upperBound = vec3f(128) * gensv::computeGrid(worldSize * nlocalBricks);
      worldBounds = box3f(vec3f(0), upperBound);

      for (size_t i = 0; i < volumes.size(); ++i) {
        auto &v = volumes[i];
        v.id = rank * nlocalBricks + i;
      }
    }

    for (auto &v : volumes) {
      Model m;
      if (!no_volume) {
        v.volume.commit();
        m.addVolume(v.volume);
      }
      // All ranks generate the same sphere data to mimic rendering a distributed
      // shared dataset
      if (nSpheres != 0) {
        auto spheres = gensv::makeSpheres(worldBounds, nSpheres,
            sphereRadius, transparentSpheres);
        m.addGeometry(spheres);

        Model g;
        g.addGeometry(spheres);
        ghostModels.push_back(g);
      }
      for (size_t i = 0; i < v.isosurfaces.size(); ++i) {
        auto &iso = v.isosurfaces[i];
        iso.setMaterial(isosurfaceMats[v.isosurfaceValueIndex[i] % isosurfaceMats.size()]);
        iso.set("geom.materialID", 0);
        iso.commit();

        m.addGeometry(iso);
        m.set("compactMode", 1);

        Model g;
        g.addGeometry(iso);
        g.set("compactMode", 1);
        g.set("id", v.id);
        ghostModels.push_back(g);
      }
      // Clip off any ghost voxels or triangles from the isosurface
      m.set("region.lower", v.bounds.lower);
      m.set("region.upper", v.bounds.upper);
      m.set("id", v.id);
      models.push_back(m);
    }
  } else {
    auto web = gensv::loadCosmicWeb(cosmicWebPath, cosmicWebGrid);
    Model m;
    for (auto &b : web.bricks) {
      m.addGeometry(b);
    }
    m.set("compactMode", 1);
    m.set("id", mpicommon::globalRank());
    m.set("region.lower", web.localBounds.lower);
    m.set("region.upper", web.localBounds.upper);
    models.push_back(m);
    worldBounds = box3f(vec3f(0), cosmicWebGrid * 768);
  }

  // Load any replicated OBJ files we're given
  for (const auto &obj : replicatedObjs) {
    std::cout << "Loading OBJ: " << obj << "\n";
		const bool binaryOBJ = obj.substr(obj.size() - 4) == "bobj";
    std::vector<vec3i> indexBuf;
    std::vector<float> verts;
    if (!binaryOBJ) {
      // Load the OBJ file
      tinyobj::attrib_t attrib;
      std::vector<tinyobj::shape_t> shapes;
      std::vector<tinyobj::material_t> materials;
      std::string err, warn;
      bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, obj.c_str());
      if (!ret) {
        std::cout << "Error loading mesh: " << err << std::endl;
        std::exit(1);
      }
      if (shapes.size() > 1) {
        std::cout << "Error: OBJ file must contain a single object/group\n" << std::flush;
        std::exit(1);
      }

      // Need to build the index buffer ourselves
      for (size_t f = 0; f < shapes[0].mesh.num_face_vertices.size(); ++f) {
        int fv = shapes[0].mesh.num_face_vertices[f];
        if (fv != 3) {
          std::cout << "Error: only triangle meshes are supported\n" << std::flush;
          std::exit(1);
        }
        // Loop over vertices in the face.
        vec3i indices;
        for (size_t v = 0; v < 3; ++v) {
          tinyobj::index_t idx = shapes[0].mesh.indices[f * 3 + v];
          indices[v] = idx.vertex_index;
        }
        indexBuf.push_back(indices);
      }
      verts = std::move(attrib.vertices);
    } else {
      std::ifstream fin(obj.c_str(), std::ios::binary);
      uint64_t header[2] = {0};
      fin.read(reinterpret_cast<char*>(header), sizeof(header));
      verts.resize(header[0] * 3, 0.f);
      fin.read(reinterpret_cast<char*>(verts.data()), sizeof(float) * 3 * header[0]);
      std::vector<uint64_t> indices(header[1] * 3, 0);
      fin.read(reinterpret_cast<char*>(indices.data()), sizeof(uint64_t) * 3 * header[1]);
      indexBuf.reserve(header[1]);
      for (size_t i = 0; i < indices.size(); i += 3) {
        indexBuf.push_back(vec3i(indices[i], indices[i + 1], indices[i + 2]));
      }
    }
    std::cout << "Loaded mesh with " << verts.size() / 3 << " verts and "
       << indexBuf.size() << " indices\n";

    // We assume there's a single shape in each OBJ file, since that's
    // what my mesh gridder and isosurface to obj tools output.
    ospray::cpp::Geometry triMesh("triangles");
    ospray::cpp::Data vertsData(verts.size() / 3, OSP_FLOAT3, verts.data());
    ospray::cpp::Data indexData(indexBuf.size(), OSP_INT3, indexBuf.data());
    triMesh.set("vertex", vertsData);
    triMesh.set("index", indexData);
    triMesh.setMaterial(isosurfaceMats[0]);
    triMesh.set("geom.materialID", 0);
    triMesh.commit();

    for (auto &m : models) {
      m.set("compactMode", 1);
      m.addGeometry(triMesh);
    }
  }

  for (auto &m : models) {
    m.commit();
  }
  for (auto &m : ghostModels) {
    m.commit();
  }

  Arcball arcballCamera(worldBounds, imgSize);

  Camera camera("perspective");
  if (!cmdlineViewParams) {
    camera.set("pos", arcballCamera.eyePos());
    camera.set("dir", arcballCamera.lookDir());
    camera.set("up", arcballCamera.upDir());
  } else {
    vec3f dir = cmdlineTarget - cmdlineEye;
    camera.set("pos", cmdlineEye);
    camera.set("dir", dir);
    camera.set("up", cmdlineUp);
  }
  camera.set("aspect", static_cast<float>(app.fbSize.x) / app.fbSize.y);
  camera.commit();

  Renderer renderer("mpi_raycast");

  std::vector<OSPModel> modelHandles;
  std::transform(models.begin(), models.end(), std::back_inserter(modelHandles),
                 [](const Model &m) { return m.handle(); });
  Data modelsData(modelHandles.size(), OSP_OBJECT, modelHandles.data());

  std::vector<OSPModel> ghostModelHandles;
  std::transform(ghostModels.begin(), ghostModels.end(), std::back_inserter(ghostModelHandles),
                 [](const Model &m) { return m.handle(); });
  Data ghostModelsData(ghostModelHandles.size(), OSP_OBJECT, ghostModelHandles.data());

  renderer.set("model", modelsData);
  renderer.set("ghostModel", ghostModelsData);
  renderer.set("camera", camera);
  renderer.set("bgColor", vec4f(bgColor.x, bgColor.y, bgColor.z, 0.0));
  renderer.set("varianceThreshold", varianceThreshold);
  renderer.set("aoSamples", aoSamples);
  renderer.set("shadowsEnabled", shadowsEnabled);
  renderer.commit();
  assert(renderer);

  int fbFlags = OSP_FB_COLOR | OSP_FB_ACCUM;
  if (varianceThreshold != 0.0f) {
    fbFlags |= OSP_FB_VARIANCE;
  }

  OSPFrameBufferFormat fbColorFormat = OSP_FB_SRGBA;
  if (fbnone) {
    fbColorFormat = OSP_FB_NONE;
  }
  FrameBuffer fb(app.fbSize, fbColorFormat, fbFlags);
  if (fbnone) {
    PixelOp pixelOp("debug");
    pixelOp.set("prefix", "distrib-viewer");
    pixelOp.commit();
    fb.setPixelOp(pixelOp);
  }
  fb.commit();
  fb.clear(fbFlags);

  mpicommon::world.barrier();

  std::mt19937 rng;
  std::uniform_real_distribution<float> randomCamDistrib;

  containers::AlignedVector<vec3f> tfcnColors;
  containers::AlignedVector<float> tfcnAlphas;

  //std::shared_ptr<ospray::sg::TransferFunction> transferFcn = nullptr;
  //std::shared_ptr<ospray::imgui3D::TransferFunction> tfnWidget = nullptr;
  std::shared_ptr<WindowState> windowState;
  GLFWwindow *window = nullptr;
  if (rank == 0) {
	/*
    transferFcn = std::make_shared<ospray::sg::TransferFunction>();

    tfnWidget = std::make_shared<ospray::imgui3D::TransferFunction>(transferFcn);
    tfnWidget->loadColorMapPresets();

    if (!transferFcnFile.str().empty()) {
      tfnWidget->load(transferFcnFile);
      tfn::TransferFunction loaded(transferFcnFile);
      transferFcn->child("valueRange").setValue(valueRange);

      auto &colorCP = *transferFcn->child("colorControlPoints").nodeAs<ospray::sg::DataVector4f>();
      auto &opacityCP = *transferFcn->child("opacityControlPoints").nodeAs<ospray::sg::DataVector2f>();
      opacityCP.clear();
      colorCP.clear();

      for (size_t i = 0; i < loaded.rgbValues.size(); ++i) {
        colorCP.push_back(vec4f(static_cast<float>(i) / loaded.rgbValues.size(),
                                loaded.rgbValues[i].x,
                                loaded.rgbValues[i].y,
                                loaded.rgbValues[i].z));
      }
      for (size_t i = 0; i < loaded.opacityValues.size(); ++i) {
        const float x = (loaded.opacityValues[i].x - loaded.dataValueMin)
                      / (loaded.dataValueMax - loaded.dataValueMin);
        opacityCP.push_back(vec2f(x, loaded.opacityValues[i].y * opacityScaling));
      }

      opacityCP.markAsModified();
      colorCP.markAsModified();

      transferFcn->updateChildDataValues();
      auto &colors    = *transferFcn->child("colors").nodeAs<ospray::sg::DataBuffer>();
      auto &opacities = *transferFcn->child("opacities").nodeAs<ospray::sg::DataBuffer>();

      colors.markAsModified();
      opacities.markAsModified();

      transferFcn->commit();

      tfcnColors = transferFcn->child("colors").nodeAs<ospray::sg::DataVector3f>()->v;
      const auto &ospAlpha = transferFcn->child("opacities").nodeAs<ospray::sg::DataVector1f>()->v;
      tfcnAlphas.clear();
      std::copy(ospAlpha.begin(), ospAlpha.end(), std::back_inserter(tfcnAlphas));
      app.tfcnChanged = true;
    }
	  */

    if (!glfwInit()) {
      std::cerr << "Failed to init GLFW" << std::endl;
      std::exit(1);
    }
    window = glfwCreateWindow(app.fbSize.x, app.fbSize.y,
        "Sample Distributed OSPRay Viewer", nullptr, nullptr);
    if (!window) {
      glfwTerminate();
      std::cerr << "Failed to create window" << std::endl;
      std::exit(1);
    }
    glfwMakeContextCurrent(window);

    windowState = std::make_shared<WindowState>(app, arcballCamera);

	ImGui::CreateContext();
    ImGui_ImplGlfwGL3_Init(window, false);

    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetWindowUserPointer(window, windowState.get());
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

    glfwSetMouseButtonCallback(window, ImGui_ImplGlfwGL3_MouseButtonCallback);
    glfwSetScrollCallback(window, ImGui_ImplGlfwGL3_ScrollCallback);
    glfwSetCharCallback(window, charCallback);
  }

  int frameNumber = 0;
  while (!app.quit) {
    using namespace std::chrono;

    if (app.cameraChanged) {
      camera.set("pos", app.v[0]);
      camera.set("dir", app.v[1]);
      camera.set("up", app.v[2]);
      camera.commit();

      fb.clear(fbFlags);
      app.cameraChanged = false;
    }
    auto startFrame = high_resolution_clock::now();
    renderer.renderFrame(fb, OSP_FB_COLOR);
    auto endFrame = high_resolution_clock::now();

    const int renderTime = duration_cast<milliseconds>(endFrame - startFrame).count();

    if (rank == 0) {
      glClear(GL_COLOR_BUFFER_BIT);
      if (!fbnone && frameNumber > 0) {
        uint32_t *img = (uint32_t*)fb.map(OSP_FB_COLOR);
        glDrawPixels(app.fbSize.x, app.fbSize.y, GL_RGBA, GL_UNSIGNED_BYTE, img);
        if (!outputFilename.empty()) {
          std::string fname = outputFilename + "-frame-00000.ppm";
          std::sprintf(&fname[0], "%s-frame-%05d.ppm", outputFilename.c_str(),
                       frameNumber);
          utility::writePPM(fname, app.fbSize.x, app.fbSize.y, img);

        }
        if (frameNumber >= framesToRender) {
          app.quit = true;
        }
        fb.unmap(img);
      }

      //const auto tfcnTimeStamp = transferFcn->childrenLastModified();

      if (showGUI) {
        ImGui_ImplGlfwGL3_NewFrame();
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                    1000.0f / ImGui::GetIO().Framerate,
                    ImGui::GetIO().Framerate);
		ImGui::Text("OSPRay render time %d ms/frame", renderTime);
		//tfnWidget->drawUi();

        ImGui::Render();
      }

      glfwSwapBuffers(window);

      glfwPollEvents();
      if (glfwWindowShouldClose(window)) {
        app.quit = true;
      }

	  /*
      tfnWidget->render();

      if (transferFcn->childrenLastModified() != tfcnTimeStamp) {
        transferFcn->child("valueRange").setValue(valueRange);
        transferFcn->updateChildDataValues();
        tfcnColors = transferFcn->child("colors").nodeAs<ospray::sg::DataVector3f>()->v;
        const auto &ospAlpha = transferFcn->child("opacities").nodeAs<ospray::sg::DataVector1f>()->v;
        tfcnAlphas.clear();
        std::copy(ospAlpha.begin(), ospAlpha.end(), std::back_inserter(tfcnAlphas));
        app.tfcnChanged = true;
      }
	  */

      const vec3f eye = windowState->camera.eyePos();
      const vec3f look = windowState->camera.lookDir();
      const vec3f up = windowState->camera.upDir();
      app.v[0] = vec3f(eye.x, eye.y, eye.z);
      app.v[1] = vec3f(look.x, look.y, look.z);
      app.v[2] = vec3f(up.x, up.y, up.z);
      app.cameraChanged = windowState->cameraChanged;
      windowState->cameraChanged = false;
      windowState->isImGuiHovered = ImGui::IsMouseHoveringAnyWindow();

	  /*
      if (frameNumber > 0) {
        std::cout << "Frame took: " << renderTime << "ms\n";
      }
	  */
    }
    // Send out the shared app state that the workers need to know, e.g. camera
    // position, if we should be quitting.
    MPI_Bcast(&app, sizeof(AppState), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (app.fbSizeChanged) {
      fb = FrameBuffer(app.fbSize, OSP_FB_SRGBA, fbFlags);
      fb.clear(fbFlags);
      camera.set("aspect", static_cast<float>(app.fbSize.x) / app.fbSize.y);
      camera.commit();

      arcballCamera.updateScreen(vec2i(app.fbSize.x, app.fbSize.y));
      app.fbSizeChanged = false;
      if (rank == 0) {
        glViewport(0, 0, app.fbSize.x, app.fbSize.y);
      }
    }
    if (app.tfcnChanged) {
      size_t sz = tfcnColors.size();
      MPI_Bcast(&sz, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
      if (rank != 0) {
        tfcnColors.resize(sz);
      }
      MPI_Bcast(tfcnColors.data(), sizeof(vec3f) * tfcnColors.size(), MPI_BYTE,
                0, MPI_COMM_WORLD);

      sz = tfcnAlphas.size();
      MPI_Bcast(&sz, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
      if (rank != 0) {
        tfcnAlphas.resize(sz);
      }
      MPI_Bcast(tfcnAlphas.data(), sizeof(float) * tfcnAlphas.size(), MPI_BYTE,
                0, MPI_COMM_WORLD);

      Data colorData(tfcnColors.size(), OSP_FLOAT3, tfcnColors.data());
      Data alphaData(tfcnAlphas.size(), OSP_FLOAT, tfcnAlphas.data());
      colorData.commit();
      alphaData.commit();

      for (auto &v : volumes) {
        v.tfcn.set("colors", colorData);
        v.tfcn.set("opacities", alphaData);
        v.tfcn.commit();
      }

      fb.clear(fbFlags);
      app.tfcnChanged = false;
    }

	if (autoRotate || autoRandomCamera) {
		if (autoRotate) {
			arcballCamera.rotate(vec2f(0), vec2f(0.01, 0));
			app.cameraChanged = true;
		} else if (autoRandomCamera) {
			arcballCamera.rotate(vec2f(0),
					vec2f(randomCamDistrib(rng), randomCamDistrib(rng)));
			app.cameraChanged = true;
		}

		const vec3f eye = arcballCamera.eyePos();
		const vec3f look = arcballCamera.lookDir();
		const vec3f up = arcballCamera.upDir();
		app.v[0] = vec3f(eye.x, eye.y, eye.z);
		app.v[1] = vec3f(look.x, look.y, look.z);
		app.v[2] = vec3f(up.x, up.y, up.z);
		app.cameraChanged = true;
	}

    ++frameNumber;
  }
  if (rank == 0) {
      ImGui_ImplGlfwGL3_Shutdown();
      glfwDestroyWindow(window);
  }

}

int main(int argc, char **argv) {
  parseArgs(argc, argv);

  // The application can be responsible for initializing and finalizing MPI,
  // or can let OSPRay's mpi_distributed device handle it. In the case that
  // the distributed device is responsible MPI will be initialized when the
  // device is created and finalized when it's destroyed.
  if (appInitMPI) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    assert(provided == MPI_THREAD_MULTIPLE);
  }

  runApp();

  ospShutdown();
  // If the app is responsible for setting up MPI we've also got
  // to finalize it at the exit
  if (appInitMPI) {
    MPI_Finalize();
  }
  return 0;
}

