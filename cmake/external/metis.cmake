# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(ExternalProject)

set(METIS_PREFIX_DIR ${THIRD_PARTY_PATH}/metis)
set(METIS_INSTALL_DIR ${THIRD_PARTY_PATH}/install/metis)
set(METIS_INCLUDE_DIR
    "${METIS_INSTALL_DIR}/include"
    CACHE PATH "metis include directory." FORCE)
set(METIS_LIBRARIES
    "${METIS_INSTALL_DIR}/lib/libmetis.a"
    CACHE FILEPATH "metis library." FORCE)
set(METIS_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
include_directories(${METIS_INCLUDE_DIR})

ExternalProject_Add(
  extern_metis
  ${EXTERNAL_PROJECT_LOG_ARGS}
  PREFIX ${METIS_PREFIX_DIR}
  GIT_REPOSITORY "https://github.com/KarypisLab/METIS"
#   GIT_TAG v5.1.0
  UPDATE_COMMAND ""
  CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DWITH_BZ2=OFF
             -DWITH_GFLAGS=OFF
             -DCMAKE_CXX_FLAGS=${METIS_CMAKE_CXX_FLAGS}
             -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
  #    BUILD_BYPRODUCTS ${METIS_PREFIX_DIR}/src/extern_metis/libmetis.a
  BUILD_COMMAND $(MAKE) config shared=1 i64=1 && $(MAKE) install
  INSTALL_COMMAND
    mkdir -p ${METIS_INSTALL_DIR}/lib/ && cp
    ${METIS_PREFIX_DIR}/src/extern_metis/libmetis.a ${METIS_LIBRARIES}
    && cp -r ${METIS_PREFIX_DIR}/src/extern_metis/include
    ${METIS_INSTALL_DIR}/
  BUILD_IN_SOURCE 1
  BYPRODUCTS ${METIS_LIBRARIES})

add_dependencies(extern_metis snappy)

add_library(metis STATIC IMPORTED GLOBAL)
set_property(TARGET metis PROPERTY IMPORTED_LOCATION ${METIS_LIBRARIES})
add_dependencies(metis extern_metis)

list(APPEND external_project_dependencies metis)