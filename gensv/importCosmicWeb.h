#pragma once

#include <string>
#include <vector>

#include <ospcommon/vec.h>
#include <ospcommon/box.h>

void importCosmicWebBrick(const std::string &basePath,
                          const ospcommon::vec3i &brick,
                          const ospcommon::box3f &filterBox,
                          std::vector<ospcommon::vec3f> &positions);

