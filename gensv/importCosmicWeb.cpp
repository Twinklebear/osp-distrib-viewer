#include <string>
#include <algorithm>
#include <limits>
#include <fstream>
#include "importCosmicWeb.h"

using namespace ospcommon;

#pragma pack(1)
struct CosmicWebHeader {
  // number of particles in this dat file
  int np_local;
  float a, t, tau;
  int nts;
  float dt_f_acc, dt_pp_acc, dt_c_acc;
  int cur_checkpoint, cur_projection, cur_halofind;
  float massp;
};
std::ostream& operator<<(std::ostream &os, const CosmicWebHeader &h) {
  os << "{\n\tnp_local = " << h.np_local
    << "\n\ta = " << h.a
    << "\n\tt = " << h.t
    << "\n\ttau = " << h.tau
    << "\n\tnts = " << h.nts
    << "\n\tdt_f_acc = " << h.dt_f_acc
    << "\n\tdt_pp_acc = " << h.dt_pp_acc
    << "\n\tdt_c_acc = " << h.dt_c_acc
    << "\n\tcur_checkpoint = " << h.cur_checkpoint
    << "\n\tcur_halofind = " << h.cur_halofind
    << "\n\tmassp = " << h.massp
    << "\n}";
  return os;
}

void importCosmicWebBrick(const std::string &basePath,
                          const vec3i &brick,
                          const box3f &filterBox,
                          std::vector<vec3f> &positions)
{
  // The cosmic web bricking is 8^3
  const int brickNumber = brick.x + 8 * (brick.y + 8 * brick.z);
  std::string brickName = "0.000xv000.dat";
  std::sprintf(&brickName[0], "0.000xv%03d.dat", brickNumber);

  const std::string file = basePath + "/" + brickName;
  std::ifstream fin(file.c_str(), std::ios::binary);

  if (!fin.good()) {
    throw std::runtime_error("could not open cosmic web file " + file);
  }

  CosmicWebHeader header;
  if (!fin.read(reinterpret_cast<char*>(&header), sizeof(CosmicWebHeader))) {
    throw std::runtime_error("Failed to read header");
  }

  // Each cell is 768x768x768 units
  const float step = 768.f;
  const vec3f offset(step * brick.x, step * brick.y, step * brick.z);

  positions.reserve(positions.size() + header.np_local);

  std::vector<char> fileData(header.np_local * 2 * sizeof(vec3f), 0);
  if (!fin.read(fileData.data(), fileData.size())) {
    throw std::runtime_error("Failed to read cosmic web file");
  }

  const vec3f *vecs = reinterpret_cast<const vec3f*>(fileData.data());
  for (int i = 0; i < header.np_local; ++i) {
    const vec3f v = vecs[i * 2] + offset;
    if (filterBox.contains(v)) {
      positions.push_back(v);
    }
  }
}

