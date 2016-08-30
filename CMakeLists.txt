cmake_minimum_required( VERSION 2.8 )
project( main )

include_directories( /usr/local/include/igraph )
include_directories( /usr/local/Cellar/igraph/0.7.1_3/lib/ )
add_library(igraph STATIC IMPORTED)
set_property(TARGET igraph PROPERTY
             IMPORTED_LOCATION /usr/local/Cellar/igraph/0.7.1_3/lib/libigraph.dylib)

find_package( OpenCV REQUIRED )
if( OpenCV_FOUND )
list( APPEND ThirdParty_LIBS ${OpenCV_LIBS} )
    include_directories( ${OpenCV_INCLUDE_DIRS} )
endif( OpenCV_FOUND )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )

add_executable( image_graph main.cpp )

target_link_libraries( image_graph ${OpenCV_LIBS} )
target_link_libraries( image_graph igraph )
