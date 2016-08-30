cmake_minimum_required( VERSION 2.8 )
project( main )

FIND_PATH(IGRAPH_INCLUDE_DIR igraph/igraph.h
  PATHS
  $ENV{OSSIM_HOME}
  NO_DEFAULT_PATH
    PATH_SUFFIXES include
)

FIND_PATH(IGRAPH_INCLUDE_DIR igraph/igraph.h
  PATHS
  /home/covise/covise/extern_libs/include
  /usr/local/include
  /usr/local/include/<TAB>
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
)

FIND_LIBRARY(IGRAPH_LIBRARY 
  NAMES igraph
  PATHS $ENV{IGRAPH_HOME}
    NO_DEFAULT_PATH
    PATH_SUFFIXES lib64 lib
)
FIND_LIBRARY(IGRAPH_LIBRARY 
  NAMES igraph
  PATHS
    /home/covise/covise/extern_libs
    /usr/local/Cellar/igraph/<TAB> #brew
    /usr/local
    /usr
    /sw
    /opt/local
    /opt/csw
    /opt
    /usr/freeware
  PATH_SUFFIXES lib64 lib
)

SET(IGRAPH_FOUND "NO")
IF(IGRAPH_LIBRARY AND IGRAPH_INCLUDE_DIR)
  SET(IGRAPH_FOUND "YES")
ENDIF(IGRAPH_LIBRARY AND IGRAPH_INCLUDE_DIR)

#include_directories( /usr/local/include/igraph )
#include_directories( /usr/local/Cellar/igraph/0.7.1_3/lib/ )
#add_library(igraph STATIC IMPORTED)
#set_property(TARGET igraph PROPERTY
#             IMPORTED_LOCATION /usr/local/Cellar/igraph/0.7.1_3/lib/libigraph.dylib)

find_package( OpenCV REQUIRED )
if( OpenCV_FOUND )
list( APPEND ThirdParty_LIBS ${OpenCV_LIBS} )
    include_directories( ${OpenCV_INCLUDE_DIRS} )
endif( OpenCV_FOUND )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )

add_executable( image_graph main.cpp )

target_link_libraries( image_graph ${OpenCV_LIBS} )
#target_link_libraries( image_graph igraph )
