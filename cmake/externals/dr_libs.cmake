FetchContent_Declare(dr_libs
    URL         https://github.com/mackron/dr_libs/archive/dbbd08d81fd2b084c5ae931531871d0c5fd83b87.zip
    URL_HASH    SHA1=84a2a31ef890b6204223b12f71d6e701c0edcd92
    SOURCE_SUBDIR not_set
    PATCH_COMMAND /opt/freeware/bin/patch -p1 --binary < ${CMAKE_CURRENT_SOURCE_DIR}/cmake/externals/dr_libs_aix.patch
)

FetchContent_MakeAvailable(dr_libs)
