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

#include <set>
#include <map>
#include <unordered_map>
#include <vector>
#include <random>
#include <array>
#include <mpiCommon/MPICommon.h>
#include <ospcommon/xml/XML.h>
#include <ospray/ospray_cpp/Data.h>
#include <ospray/ospray_cpp/TransferFunction.h>
#include <ospray_cpp/Volume.h>
#include "raw_reader.h"
#include "llnlrm_reader.h"
#include "generateSciVis.h"
#include "importCosmicWeb.h"

#include <vtkSmartPointer.h>
#include <vtkTriangle.h>
#include <vtkFlyingEdges3D.h>
#include <vtkImageData.h>

namespace gensv {

  bool computeDivisor(int x, int &divisor)
  {
    int upperBound = std::sqrt(x);
    for (int i = 2; i <= upperBound; ++i) {
      if (x % i == 0) {
        divisor = i;
        return true;
      }
    }
    return false;
  }

  // Compute an X x Y x Z grid to have num bricks,
  // only gives a nice grid for numbers with even factors since
  // we don't search for factors of the number, we just try dividing by two
  vec3i computeGrid(int num)
  {
    vec3i grid(1);
    int axis = 0;
    int divisor = 0;
    while (computeDivisor(num, divisor)) {
      grid[axis] *= divisor;
      num /= divisor;
      axis = (axis + 1) % 3;
    }
    if (num != 1) {
      grid[axis] *= num;
    }
    return grid;
  }

  /* This function generates the rank's local geometry within its
   * volume's bounding box. The bbox represents say its simulation
   * or owned data region.
   */
  ospray::cpp::Geometry makeSpheres(const box3f &bbox, const size_t numSpheres,
                                    const float sphereRadius,
                                    const bool transparent)
  {
    struct Sphere
    {
      vec3f org;
      int colorID{0};
    };

    auto numRanks = static_cast<float>(mpicommon::numGlobalRanks());
    auto myRank   = mpicommon::globalRank();

    containers::AlignedVector<Sphere> spheres(numSpheres);

    // To simulate loading a shared dataset all ranks generate the same
    // sphere data.
    std::mt19937 rng;
    rng.seed(0);

    // Generate spheres within this nodes volume, to keep the data disjoint.
    // We also leave some buffer space on the boundaries to avoid clipping
    // artifacts or needing duplication across nodes in the case a sphere
    // crosses a boundary. Note: Since we don't communicated ghost regions
    // among the nodes, we make sure not to generate any spheres which would
    // be clipped.
    std::uniform_real_distribution<float> dist_x(bbox.lower.x + sphereRadius,
                                                 bbox.upper.x - sphereRadius);
    std::uniform_real_distribution<float> dist_y(bbox.lower.y + sphereRadius,
                                                 bbox.upper.y - sphereRadius);
    std::uniform_real_distribution<float> dist_z(bbox.lower.z + sphereRadius,
                                                 bbox.upper.z - sphereRadius);
    for (auto &s : spheres) {
      s.org.x = dist_x(rng);
      s.org.y = dist_y(rng);
      s.org.z = dist_z(rng);
    }

    ospray::cpp::Data sphere_data(numSpheres * sizeof(Sphere), OSP_UCHAR,
                                  spheres.data());

    const float r = (numRanks - myRank) / numRanks;
    const float b = myRank / numRanks;
    const float g = myRank > numRanks / 2 ? 2 * r : 2 * b;
    vec4f color(r, g, b, 1.0);
    if (transparent) {
      color.w = 0.1;
    }
    ospray::cpp::Data color_data(1, OSP_FLOAT4, &color);

    ospray::cpp::Geometry geom("spheres");
    geom.set("offset_colorID", int(sizeof(vec3f)));
    geom.set("bytes_per_sphere", int(sizeof(Sphere)));
    geom.set("spheres", sphere_data);
    geom.set("radius", sphereRadius);
    geom.set("color", color_data);
    geom.commit();

    return geom;
  }

  CosmicWeb loadCosmicWeb(const std::string &basePath, const vec3i &cosmicWebGrid) {
    using namespace ospray::cpp;

    auto numRanks = static_cast<float>(mpicommon::numGlobalRanks());
    auto myRank   = mpicommon::globalRank();
    const vec3i grid = computeGrid(numRanks);
    const vec3i brickId(myRank % grid.x, (myRank / grid.x) % grid.y, myRank / (grid.x * grid.y));
    const vec3f brickDims = vec3f(cosmicWebGrid) / vec3f(grid);
    const vec3f gridOrigin = vec3f(brickId) * vec3f(brickDims);
    const box3f myBrick(gridOrigin, gridOrigin + brickDims);

    Material mat("scivis", "OBJMaterial");
    mat.set("Kd", vec3f(1));
    mat.set("d", 0.2f);
    mat.commit();

    CosmicWeb web;
    web.localBounds.lower = myBrick.lower * vec3f(768.f);
    web.localBounds.upper = myBrick.upper * vec3f(768.f);
    const float particleRadius = 0.015;
    // The region of particles we want to load (including ghost particles)
    const box3f loadRegion(web.localBounds.lower - vec3f(particleRadius),
                           web.localBounds.upper + vec3f(particleRadius));

    std::vector<vec3f> positions;
    for (int z = 0; z < cosmicWebGrid.z; ++z) {
      for (int y = 0; y < cosmicWebGrid.y; ++y) {
        for (int x = 0; x < cosmicWebGrid.x; ++x) {
          box3f cell(vec3f(x, y, z), (vec3f(x, y, z) + vec3f(1)));
          box3f cellBounds(cell.lower * 768.f, cell.upper * 768.f);
          if (touchingOrOverlapping(loadRegion, cellBounds)) {
            std::cout << "Loading brick " << cell << "\n";
            importCosmicWebBrick(basePath, cell.lower, loadRegion, positions);
            if (!positions.empty()) {
              std::cout << "Rank " << mpicommon::globalRank()
                << " loaded " << positions.size() << " particles from brick "
                << cell << "\n";
              Geometry g("spheres");
              g.set("bytes_per_sphere", int(sizeof(vec3f)));
              Data sphere_data(positions.size(), OSP_FLOAT3, positions.data());
              g.set("spheres", sphere_data);
              g.set("radius", particleRadius);
              g.setMaterial(mat);
              g.commit();
              web.bricks.push_back(g);
            }
          }
          positions.clear();
          std::cout << "Rank " << mpicommon::globalRank()
            << " cosmic web progress: "
            << 100.0 * static_cast<float>(x + 1 + cosmicWebGrid.x * (y + cosmicWebGrid.y * z))
              / (cosmicWebGrid.x * cosmicWebGrid.y * cosmicWebGrid.z)
            << "%\n";
        }
      }
    }
    std::cout << "Cosmic web bricks loaded" << std::endl;
    return web;
  }

  LoadedVolume::LoadedVolume()
    : volume(nullptr), tfcn("piecewise_linear"),
    id(mpicommon::globalRank())
  {
    const containers::AlignedVector<vec3f> colors = {
      vec3f(0, 0, 0.56),
      vec3f(0, 0, 1),
      vec3f(0, 1, 1),
      vec3f(0.5, 1, 0.5),
      vec3f(1, 1, 0),
      vec3f(1, 0, 0),
      vec3f(0.5, 0, 0)
    };
    const containers::AlignedVector<float> opacities = {0.01, 1.0};
    ospray::cpp::Data colorsData(colors.size(), OSP_FLOAT3, colors.data());
    ospray::cpp::Data opacityData(opacities.size(), OSP_FLOAT, opacities.data());
    colorsData.commit();
    opacityData.commit();

    tfcn.set("colors", colorsData);
    tfcn.set("opacities", opacityData);
  }

  enum GhostFace {
    NEITHER_FACE = 0,
    POS_FACE = 1,
    NEG_FACE = 1 << 1,
  };

  /* Compute which faces of this brick we need to specify ghost voxels for,
   * to have correct interpolation at brick boundaries. Returns mask of
   * GhostFaces for x, y, z.
   */
  std::array<int, 3> computeGhostFaces(const vec3i &brickId, const vec3i &grid) {
    std::array<int, 3> faces = {{NEITHER_FACE, NEITHER_FACE, NEITHER_FACE}};
    for (size_t i = 0; i < 3; ++i) {
      if (brickId[i] < grid[i] - 1) {
        faces[i] |= POS_FACE;
      }
      if (brickId[i] > 0) {
        faces[i] |= NEG_FACE;
      }
    }
    return faces;
  }

  LoadedVolume makeBrick(const size_t brickNum, const size_t numBricks) {
    LoadedVolume vol;
    vol.tfcn.set("valueRange", vec2f(0, numBricks - 1));
    vol.tfcn.commit();

    const vec3i volumeDims(128);
    const vec3i grid = computeGrid(numBricks);
    const vec3f gridSpacing = vec3f(1.f);
    const vec3i brickId(brickNum % grid.x, (brickNum / grid.x) % grid.y, brickNum / (grid.x * grid.y));
    const vec3f gridOrigin = vec3f(brickId) * gridSpacing * vec3f(volumeDims);
    const std::array<int, 3> ghosts = computeGhostFaces(brickId, grid);
    vec3i ghostDims(0);
    for (size_t i = 0; i < 3; ++i) {
      if (ghosts[i] & POS_FACE) {
        ghostDims[i] += 1;
      }
      if (ghosts[i] & NEG_FACE) {
        ghostDims[i] += 1;
      }
    }
    const vec3i fullDims = volumeDims + ghostDims;
    const vec3i ghostOffset(ghosts[0] & NEG_FACE ? 1 : 0,
                               ghosts[1] & NEG_FACE ? 1 : 0,
                               ghosts[2] & NEG_FACE ? 1 : 0);
    vol.ghostGridOrigin = gridOrigin - vec3f(ghostOffset) * gridSpacing;

    vol.volume = ospray::cpp::Volume("block_bricked_volume");
    vol.volume.set("voxelType", "uchar");
    vol.volume.set("dimensions", fullDims);
    vol.volume.set("transferFunction", vol.tfcn);
    vol.volume.set("gridSpacing", gridSpacing);
    vol.volume.set("gridOrigin", vol.ghostGridOrigin);

    containers::AlignedVector<unsigned char> volumeData(
      fullDims.x * fullDims.y * fullDims.z, 0
    );

    for (size_t i = 0; i < volumeData.size(); ++i)
      volumeData[i] = brickNum;

    vol.volume.setRegion(volumeData.data(), vec3i(0), fullDims);
    vol.volume.commit();

    vol.bounds = box3f(gridOrigin, gridOrigin + vec3f(volumeDims));
    return vol;
  }

  LoadedVolume makeVolume() {
    return makeBrick(mpicommon::globalRank(), mpicommon::numGlobalRanks());
  }

  containers::AlignedVector<LoadedVolume> makeVolumes(const size_t firstBrick,
                                                      const size_t numMine,
                                                      const size_t numBricks)
  {
    containers::AlignedVector<LoadedVolume> volumes;
    for (size_t i = firstBrick; i < firstBrick + numMine; ++i) {
      volumes.push_back(makeBrick(i, numBricks));
    }
    return volumes;
  }

  size_t sizeForDtype(const std::string &dtype)
  {
    if (dtype == "uchar" || dtype == "char") {
      return 1;
    }
    if (dtype == "ushort" || dtype == "short") {
      return 2;
    }
    if (dtype == "float") {
      return 4;
    }
    if (dtype == "double") {
      return 8;
    }
    throw std::runtime_error("Unrecognized data type!");
    return 0;
  }

  LoadedVolume loadVolumeBrick(RawReader &reader,
                               const vec3i &brickId,
                               const vec3sz &grid,
                               const vec3i &dimensions,
                               const std::string &dtype,
                               const vec2f &valueRange,
                               const std::vector<float> &isovalues,
                               const bool noVolume = false);

  LoadedVolume loadVolume(const FileName &file, const vec3i &dimensions,
                          const std::string &dtype, const vec2f &valueRange)
  {
    auto numRanks = static_cast<float>(mpicommon::numGlobalRanks());
    auto myRank   = mpicommon::globalRank();

    const vec3sz grid = vec3sz(computeGrid(numRanks));
    const vec3sz brickId(myRank % grid.x, (myRank / grid.x) % grid.y, myRank / (grid.x * grid.y));

    const size_t dtypeSize = sizeForDtype(dtype);
    RawReader reader(file, vec3sz(dimensions), dtypeSize);
    return loadVolumeBrick(reader, brickId, grid, dimensions, dtype,
                           valueRange, std::vector<float>{});
  }

  containers::AlignedVector<LoadedVolume>
    loadBrickedVolume(const FileName &file,
                      const vec3i &dimensions,
                      const std::string &dtype,
                      const vec2f &valueRange,
                      const size_t nbricks,
                      const size_t bricksPerRank,
                      const std::vector<float> &isovalues,
                      const bool noVolume)
  {
    const int numRanks = mpicommon::numGlobalRanks();
    const int myRank = mpicommon::globalRank();

    if (bricksPerRank > nbricks || bricksPerRank < nbricks / numRanks) {
      throw std::runtime_error("Invalid bricks-per-rank for config!");
    }

    const size_t dtypeSize = sizeForDtype(dtype);
    RawReader reader(file, vec3sz(dimensions), dtypeSize);
    const vec3sz grid = vec3sz(computeGrid(nbricks));

    containers::AlignedVector<LoadedVolume> bricks;
    if (bricksPerRank == nbricks) {
      for (size_t b = 0; b < nbricks; ++b) {
        const vec3sz brickId(b % grid.x,
                             (b / grid.x) % grid.y,
                             b / (grid.x * grid.y));
        LoadedVolume v = loadVolumeBrick(reader,
                                         brickId,
                                         grid,
                                         dimensions,
                                         dtype,
                                         valueRange,
                                         isovalues,
                                         noVolume);
        v.id = b;
        bricks.emplace_back(v);
      }
    } else {
      std::set<size_t> bprLoaded;
      for (size_t local = 0; local < bricksPerRank; ++local) {
        const size_t b = (myRank + local) % nbricks;
        if (bprLoaded.find(b) != bprLoaded.end()) {
          std::cout << "BPR Config would have loaded a brick twice!\n" << std::flush;
          throw std::runtime_error("BPR Config would have loaded a brick twice!");
        }
        bprLoaded.insert(b);
        const vec3sz brickId(b % grid.x,
                             (b / grid.x) % grid.y,
                             b / (grid.x * grid.y));
        LoadedVolume v = loadVolumeBrick(reader,
                                         brickId,
                                         grid,
                                         dimensions,
                                         dtype,
                                         valueRange,
                                         isovalues,
                                         noVolume);
        v.id = b;
        bricks.emplace_back(v);
      }
      std::cout << "Got " << bricks.size() << " bricks\n";
    }
    return bricks;
  }

  ospray::cpp::Geometry isosurfaceBrick(vtkSmartPointer<vtkImageData> imgData,
                                        const float isoval)
  {
    std::vector<vec3f> vertBuffer;
    std::vector<vec3i> indexBuffer;
    {
      vtkSmartPointer<vtkFlyingEdges3D> fedges = vtkSmartPointer<vtkFlyingEdges3D>::New();
      fedges->SetInputData(imgData);
      fedges->SetNumberOfContours(1);
      fedges->SetValue(0, isoval);
      fedges->SetComputeNormals(false);
      fedges->Update();
      vtkPolyData *isosurface = fedges->GetOutput();

      vertBuffer.reserve(isosurface->GetNumberOfCells());
      indexBuffer.reserve(isosurface->GetNumberOfCells());

      for (size_t i = 0; i < isosurface->GetNumberOfCells(); ++i) {
        vtkTriangle *tri = dynamic_cast<vtkTriangle*>(isosurface->GetCell(i));
        if (tri->ComputeArea() == 0.0) {
          continue;
        }
        vec3i tids;
        for (size_t v = 0; v < 3; ++v) {
          const double *pt = isosurface->GetPoint(tri->GetPointId(v));
          const vec3f vert(pt[0], pt[1], pt[2]);

          vertBuffer.push_back(vert);

          tids[v] = vertBuffer.size() - 1;
        }
        indexBuffer.push_back(tids);
      }
    }
    std::cout << "Isosurface at " << isoval << " has " << indexBuffer.size()
      << " triangles\n";
    ospray::cpp::Geometry triMesh("triangles");
    ospray::cpp::Data vertsData(vertBuffer.size(), OSP_FLOAT3, vertBuffer.data());
    ospray::cpp::Data indexData(indexBuffer.size(), OSP_INT3, indexBuffer.data());
    triMesh.set("vertex", vertsData);
    triMesh.set("index", indexData);
    triMesh.commit();
    return triMesh;
  }

  LoadedVolume loadVolumeBrick(RawReader &reader,
                               const vec3i &brickId,
                               const vec3sz &grid,
                               const vec3i &dimensions,
                               const std::string &dtype,
                               const vec2f &valueRange,
                               const std::vector<float> &isovalues,
                               const bool noVolume)
  {
    LoadedVolume vol;
    vol.tfcn.set("valueRange", valueRange);
    vol.tfcn.commit();

    const vec3sz brickDims = vec3sz(dimensions) / grid;
    const vec3f gridOrigin = vec3f(brickId) * vec3f(brickDims);

    const std::array<int, 3> ghosts = computeGhostFaces(vec3i(brickId), vec3i(grid));
    vec3sz ghostDims(0);
    for (size_t i = 0; i < 3; ++i) {
      if (ghosts[i] & POS_FACE) {
        ghostDims[i] += 1;
      }
      if (ghosts[i] & NEG_FACE) {
        ghostDims[i] += 1;
      }
    }
    const vec3sz fullDims = brickDims + ghostDims;
    const vec3i ghostOffset(ghosts[0] & NEG_FACE ? 1 : 0,
                               ghosts[1] & NEG_FACE ? 1 : 0,
                               ghosts[2] & NEG_FACE ? 1 : 0);
    vol.ghostGridOrigin = gridOrigin - vec3f(ghostOffset);

    if (!noVolume) {
      vol.volume = ospray::cpp::Volume("block_bricked_volume");
      vol.volume.set("voxelType", dtype.c_str());
      vol.volume.set("dimensions", vec3i(fullDims));
      vol.volume.set("transferFunction", vol.tfcn);
      vol.volume.set("gridOrigin", vol.ghostGridOrigin);
    } else {
      vol.volume = nullptr;
    }

    int vtkDataType = -1;
    if (dtype == "uchar") {
      vtkDataType = VTK_UNSIGNED_CHAR;
    } else if (dtype == "char") {
      vtkDataType = VTK_CHAR;
    } else if (dtype == "ushort") {
      vtkDataType = VTK_UNSIGNED_SHORT;
    } else if (dtype == "short") {
      vtkDataType = VTK_SHORT;
    } else if (dtype == "float") {
      vtkDataType = VTK_FLOAT;
    } else if (dtype == "double") {
      vtkDataType = VTK_DOUBLE;
    } else {
      throw std::runtime_error("Unsupported or unrecognized data type: " + dtype);
    }

    vtkSmartPointer<vtkImageData> imgData = vtkSmartPointer<vtkImageData>::New();
    imgData->SetDimensions(fullDims.x, fullDims.y, fullDims.z);
    imgData->AllocateScalars(vtkDataType, 1);
    imgData->SetOrigin(vol.ghostGridOrigin.x, vol.ghostGridOrigin.y, vol.ghostGridOrigin.z);

    reader.readRegion(brickId * brickDims - vec3sz(ghostOffset),
                      vec3sz(fullDims),
                      reinterpret_cast<uint8_t*>(imgData->GetScalarPointer()));

    if (!isovalues.empty()) {
      for (size_t i = 0; i < isovalues.size(); ++i) {
        vol.isosurfaces.push_back(isosurfaceBrick(imgData, isovalues[i]));
        vol.isosurfaceValueIndex.push_back(i);
      }
    }

    if (!noVolume) {
      vol.volume.setRegion(imgData->GetScalarPointer(), vec3i(0), vec3i(fullDims));
      vol.volume.commit();
    }

    vol.bounds = box3f(gridOrigin, gridOrigin + vec3f(brickDims));
    return vol;
  }

  containers::AlignedVector<LoadedVolume>
    loadRMBricks(const FileName &bobDir, const size_t bricksPerRank)
  {
    const int numRanks = mpicommon::numGlobalRanks();
    const int myRank = mpicommon::globalRank();
    const vec3sz grid = LLNLRMReader::blockGrid();
    const size_t nbricks = grid.x * grid.y * grid.z;

    if (bricksPerRank > nbricks || bricksPerRank < nbricks / numRanks) {
      throw std::runtime_error("Invalid bricks-per-rank for config!");
    }

    const vec2f valueRange(0, 255);
    const std::string dtype = "uchar";
    const size_t dtypeSize = sizeForDtype(dtype);
    LLNLRMReader reader(bobDir);

    containers::AlignedVector<LoadedVolume> bricks;
    // TODO: Uneven divisions? Maybe just yell at user?
    size_t bricksPerIter = std::max(size_t(1), size_t(nbricks / numRanks));
    for (size_t i = 0; i < bricksPerRank / bricksPerIter; ++i) {
      size_t brickRank = (myRank + i) % numRanks;
      for (size_t b = 0; b < nbricks; ++b) {
        if (b % numRanks == brickRank) {
          const vec3sz brickId(b % grid.x,
              (b / grid.x) % grid.y,
              b / (grid.x * grid.y));

          LoadedVolume vol;
          vol.tfcn.set("valueRange", valueRange);
          vol.tfcn.commit();

          const vec3sz brickDims = LLNLRMReader::blockSize();
          const vec3f gridOrigin = vec3f(brickId) * vec3f(brickDims);

          // TODO Ghost cells
          vec3sz ghostDims(0);
          //const std::array<int, 3> ghosts = {{0, 0, 0}};
          /*
          const std::array<int, 3> ghosts = computeGhostFaces(vec3i(brickId), vec3i(grid));
          for (size_t i = 0; i < 3; ++i) {
            if (ghosts[i] & POS_FACE) {
              ghostDims[i] += 1;
            }
            if (ghosts[i] & NEG_FACE) {
              ghostDims[i] += 1;
            }
          }
          */
          const vec3sz fullDims = brickDims + ghostDims;
          /*
          const vec3i ghostOffset(ghosts[0] & NEG_FACE ? 1 : 0,
              ghosts[1] & NEG_FACE ? 1 : 0,
              ghosts[2] & NEG_FACE ? 1 : 0);
              */
          //vol.ghostGridOrigin = gridOrigin - vec3f(ghostOffset);
          vol.ghostGridOrigin = gridOrigin - vec3f(0);

          vol.volume = ospray::cpp::Volume("block_bricked_volume");
          vol.volume.set("voxelType", dtype.c_str());
          vol.volume.set("dimensions", vec3i(fullDims));
          vol.volume.set("transferFunction", vol.tfcn);
          vol.volume.set("gridOrigin", vol.ghostGridOrigin);

          const size_t nbytes = fullDims.x * fullDims.y * fullDims.z * dtypeSize;
          containers::AlignedVector<char> volumeData(nbytes, 0);

          reader.loadBlock(b, volumeData);
          vol.volume.setRegion(volumeData.data(), vec3i(0), vec3i(fullDims));
          vol.volume.commit();

          vol.bounds = box3f(gridOrigin, gridOrigin + vec3f(brickDims));
          vol.id = b;
          bricks.emplace_back(vol);
        }
      }
    }
    return bricks;
  }

  LoadedVolume loadOSPBrick(const FileName &ospFile, const vec2f &valueRange) {
    auto doc = xml::readXML(ospFile);
    vec3i dimensions(0);
    box3f region;
    vec3f gridOrigin(0), gridSpacing(0);
    float samplingRate = 0.125;
    std::string dtype;
    FileName rawFilename;

    const xml::Node &volumeNode = doc->child[0];
    for (const xml::Node &e : volumeNode.child) {
      if (e.name == "dimensions") {
        std::stringstream str(e.content);
        str >> dimensions.x >> dimensions.y >> dimensions.z;
      } else if (e.name == "regionLower") {
        std::stringstream str(e.content);
        str >> region.lower.x >> region.lower.y >> region.lower.z;
      } else if (e.name == "regionUpper") {
        std::stringstream str(e.content);
        str >> region.upper.x >> region.upper.y >> region.upper.z;
      } else if (e.name == "ghostLower") {
        std::stringstream str(e.content);
        str >> gridOrigin.x >> gridOrigin.y >> gridOrigin.z;
      } else if (e.name == "spacing") {
        std::stringstream str(e.content);
        str >> gridSpacing.x >> gridSpacing.y >> gridSpacing.z;
      } else if (e.name == "filename") {
        rawFilename = ospFile.path() + e.content;
      } else if (e.name == "voxelType") {
        dtype = e.content;
      } else if (e.name == "samplingRate") {
        // Does the XML stuff trim out the spaces?
        std::stringstream str(e.content);
        str >> samplingRate;
      }
    }

    std::cout << "Reading brick from volume file:\n"
      << "Dims: " << dimensions << "\n"
      << "Region: " << region << "\n"
      << "gridOrigin: " << gridOrigin << "\n"
      << "spacing: " << gridSpacing << "\n"
      << "sampling rate: " << samplingRate << "\n"
      << "dtype: " << dtype << "\n"
      << "RAW file: " << rawFilename << "\n" << std::flush;

    LoadedVolume vol;
    vol.tfcn.set("valueRange", valueRange);
    vol.tfcn.commit();

    vol.ghostGridOrigin = gridOrigin;

    vol.volume = ospray::cpp::Volume("block_bricked_volume");
    vol.volume.set("voxelType", dtype.c_str());
    vol.volume.set("dimensions", dimensions);
    vol.volume.set("transferFunction", vol.tfcn);
    vol.volume.set("gridOrigin", vol.ghostGridOrigin);
    vol.volume.set("samplingRate", samplingRate);

    const size_t dtypeSize = sizeForDtype(dtype);
    const size_t nbytes = dimensions.x * dimensions.y * dimensions.z * dtypeSize;
    containers::AlignedVector<char> volumeData(nbytes, 0);

#if 0
    std::fill(reinterpret_cast<float*>(&volumeData[0]),
              reinterpret_cast<float*>(&volumeData[volumeData.size() - 1] + 1),
              float(mpicommon::globalRank()));
#else
    std::ifstream fin(rawFilename.c_str(), std::ios::binary);
    fin.read(volumeData.data(), nbytes);
#endif
    vol.volume.setRegion(volumeData.data(), vec3i(0), vec3i(dimensions));
    vol.volume.commit();

    vol.bounds = region;
    vol.id = mpicommon::globalRank();
    return vol;
  }

} // ::gensv

